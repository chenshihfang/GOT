# -*- coding: utf-8 -*-
import os
import json
import glob
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import deepspeed
    _HAVE_DS = True
except Exception:
    _HAVE_DS = False

# Keep DP checkpoint compatibility helpers if available
try:
    from ltr.admin import loading, multigpu
except Exception:
    loading = None
    multigpu = None


# ---------------- env helpers ----------------

def _env_rank():
    for k in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        if k in os.environ:
            try:
                return int(os.environ[k])
            except Exception:
                pass
    return 0


def _env_world_size():
    for k in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"):
        if k in os.environ:
            try:
                return int(os.environ[k])
            except Exception:
                pass
    return 1


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


class _NullTBWriter:
    """No-op tensorboard writer for non-zero ranks."""
    def write_info(self, *args, **kwargs): pass
    def write_epoch(self, *args, **kwargs): pass


def _read_bool(settings, name, default):
    return bool(getattr(settings, name, default))


def _read_int(settings, name, default):
    v = getattr(settings, name, default)
    try:
        return int(v)
    except Exception:
        return default


def _build_ds_config(settings):
    """Create DS config. For manual GA, force GAS=1 to match DP/DDP semantics.

    Respect settings.use_deepspeed_grad_clip to avoid double clipping.
    """
    zero_stage = getattr(settings, "deepspeed_zero_stage", None)
    if zero_stage is None:
        zero_stage = getattr(settings, "zero_stage", 0)

    per_gpu = _read_int(settings, "train_micro_batch_size_per_gpu",
                        getattr(settings, "batch_size", 1))

    # Decide DS gradient clipping from settings
    want_ds_clip = bool(getattr(settings, "use_deepspeed_grad_clip", False))
    max_norm = float(getattr(settings, "gradient_clipping", 0.0))
    clip = float(max_norm if want_ds_clip and max_norm > 0 else 0.0)

    cfg = {
        "train_micro_batch_size_per_gpu": max(1, per_gpu),
        "gradient_accumulation_steps": 1,  # manual GA in trainer
        "gradient_clipping": clip,
        "fp16": {"enabled": bool(getattr(settings, "deepspeed_fp16", False))},
        "bf16": {"enabled": bool(getattr(settings, "deepspeed_bf16", False))},
    }

    if int(zero_stage) == 0:
        cfg["zero_optimization"] = {"stage": 0}
    else:
        cfg["zero_optimization"] = {
            "stage": int(zero_stage),
            "contiguous_gradients": False,
            "reduce_scatter": False,
            "allgather_partitions": True,
            "overlap_comm": False,
            "ignore_unused_parameters": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        }
        cfg["use_multi_rank_bucket_allreduce"] = False

    return cfg


# ---------------- device picking ----------------

def _pick_device_from_local_rank():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        return torch.device("cpu"), local_rank

    n = torch.cuda.device_count()
    if n <= 0:
        return torch.device("cpu"), local_rank

    if local_rank < 0 or local_rank >= n:
        safe = local_rank % n
        print(f"[WARN] LOCAL_RANK={local_rank} out of range for {n} GPUs. Remap to {safe}.")
        local_rank = safe

    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank


# ---------------- trainer ----------------

class BaseTrainerDSA:
    """DeepSpeed/Native-DDP aware base trainer (manual GA parity).

    - Supports native PyTorch DDP (no DeepSpeed) OR DeepSpeed ZeRO-0.
    - LR scheduler stepped once per epoch (MultiStepLR).
    - Checkpoints saved/loaded in DP layout (interchangeable with DP/DDP).
    """

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        self.actor = actor              # expects .net (nn.Module)
        self.loaders = loaders
        self.optimizer = optimizer
        self.settings = settings
        self.lr_scheduler = lr_scheduler

        self.rank = _env_rank()
        self.world_size = _env_world_size()

        self.device = None
        self.engine = None
        self.use_deepspeed = _read_bool(settings, "use_deepspeed", True) and _HAVE_DS

        self.epoch = 0
        self.stats = {}

        self._checkpoint_dir = None
        self._update_settings_for_checkpoints(settings)

        self._setup_distributed_and_engine()


        # --- DoRA export controls (configurable from train settings) ---
        self.dora_save_mode = getattr(self.settings, "dora_save_mode", "none")      # "merged" | "adapters" | "both" | "none"
        self.dora_save_every_epoch = getattr(self.settings, "dora_save_every_epoch", False)
        self.dora_save_tag = getattr(self.settings, "dora_save_tag", "MERGED_INFER_ONLY")



        if is_dist_initialized():
            try:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            except Exception:
                pass

    def _find_vggt_backbone(self, net):
        # your models use either feature_extractor_VGGT or backbone_net_VGGT
        if hasattr(net, "feature_extractor_VGGT"):
            return getattr(net, "feature_extractor_VGGT")
        if hasattr(net, "backbone_net_VGGT"):
            return getattr(net, "backbone_net_VGGT")
        return None

    def _export_dora_merged(self, epoch: int):
        """Export full model with DoRA merged into VGGT weights."""
        import copy, os, torch
        if not self._is_master():
            return
        try:
            net = self._current_net()
            if net is None:
                return
            net_merged = copy.deepcopy(net).cpu()
            be = self._find_vggt_backbone(net_merged)
            if be is None or not hasattr(be, "merge_and_unload"):
                print("[merged] skip: VGGT not found or merge_and_unload missing", flush=True)
                return
            be.merge_and_unload()
            out_dir = self._project_directory()
            if out_dir is None:
                return
            net_type = type(net_merged).__name__
            out_path = os.path.join(out_dir, f"{net_type}_ep{epoch:04d}_{self.dora_save_tag}.pth")
            torch.save(net_merged.state_dict(), out_path)
            print(f"[merged] saved: {out_path}", flush=True)
        except Exception as e:
            print(f"[merged] export failed: {type(e).__name__}: {e}", flush=True)

    def _export_dora_adapters_only(self, epoch: int):
        """Export only DoRA/LoRA adapter weights from VGGT."""
        import copy, os, torch
        if not self._is_master():
            return
        try:
            net = self._current_net()
            if net is None:
                return
            net_cpu = copy.deepcopy(net).cpu()
            be = self._find_vggt_backbone(net_cpu)
            if be is None:
                print("[adapters] skip: VGGT not found", flush=True)
                return
            # keep only adapter tensors
            ad_state = {k: v for k, v in be.state_dict().items()
                        if "dora_" in k or "lora_" in k}
            if len(ad_state) == 0:
                print("[adapters] skip: no dora_/lora_ params found", flush=True)
                return
            out_dir = self._project_directory()
            if out_dir is None:
                return
            out_path = os.path.join(out_dir, f"VGGT_adapters_ep{epoch:04d}.pth")
            meta = {
                "module_name": "backbone_net_VGGT",  # where to load back at inference if needed
                "keys_prefix": "VGGT",               # free text tag
                "note": "DoRA/LoRA adapter tensors only",
            }
            torch.save({"adapters": ad_state, "meta": meta}, out_path)
            print(f"[adapters] saved: {out_path}", flush=True)
        except Exception as e:
            print(f"[adapters] export failed: {type(e).__name__}: {e}", flush=True)


    def _export_merged_for_inference(self):
        """Export a second, inference-ready checkpoint with DoRA/LoRA merged into base VGGT.
        Does not mutate the training model: works on a CPU clone.
        """
        try:
            if not self._is_master():
                return
            import copy, os, torch

            net = self._current_net()  # unwrap DS/DDP to get the real nn.Module
            if net is None:
                return

            # Clone to CPU so we do not disturb the training model
            net_merged = copy.deepcopy(net).cpu()

            # Locate VGGT backbone in your model (your code uses feature_extractor_VGGT)
            be = getattr(net_merged, "feature_extractor_VGGT", None)
            if be is None:
                be = getattr(net_merged, "backbone_net_VGGT", None)

            # If it is a PEFT-wrapped module with merge_and_unload(), fold and export
            if be is not None and hasattr(be, "merge_and_unload"):
                be.merge_and_unload()
            else:
                # Nothing to merge; skip silently
                return

            # Build output path under the same project/checkpoint directory
            directory = self._project_directory()
            if directory is None:
                return

            net_type = type(net_merged).__name__
            # Save only the model weights (typical for inference)
            out_path = os.path.join(directory, f"{net_type}_ep{self.epoch:04d}_MERGED_INFER_ONLY.pth")
            torch.save(net_merged.state_dict(), out_path)
            print(f"[merged] exported inference checkpoint: {out_path}", flush=True)

        except Exception as e:
            if self._is_master():
                print(f"[merged] export skipped: {type(e).__name__}: {e}", flush=True)


    # ---------------- helpers ----------------
    def _is_master(self) -> bool:
        return int(getattr(self, "rank", 0)) == 0

    def _dist_ready(self) -> bool:
        return is_dist_initialized()

    def _reduce_value_sum_and_count(self, value: float, count: int):
        if not self._dist_ready():
            return float(value), int(count)
        pair = torch.tensor([float(value), float(count)], dtype=torch.float32, device=self.device)
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        return float(pair[0].item()), int(pair[1].item())

    # ---------------- core hooks ----------------
    def _zero_grad(self):
        if self.engine is not None:
            self.engine.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)

    def _step(self):
        if self.engine is not None:
            self.engine.step()
        else:
            self.optimizer.step()

    def _backward(self, loss):
        if self.engine is not None:
            self.engine.backward(loss)
        else:
            loss.backward()

    # ---------------- lifecycle ----------------
    def train(self, max_epochs, load_latest=False, fail_safe=True):
        epoch = -1
        tries = 20 if fail_safe else 1
        for _ in range(tries):
            try:
                if load_latest:
                    self.load_checkpoint()

                # -------- ALIGN LR SCHEDULER ON RESUME (NO scheduler.step() here) --------
                if self.lr_scheduler is not None:
                    try:
                        # Make scheduler use the CURRENT optimizer LRs as its base
                        self.lr_scheduler.base_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
                    except Exception:
                        pass

                    # If we resume with self.epoch = E (next loop starts at E+1),
                    # we want the decay to happen at the end of epoch E when E is a milestone.
                    # MultiStepLR applies decay when (last_epoch+1) hits a milestone at step().
                    # So set last_epoch = E-1 when E>0; else keep -1.
                    if self.epoch > 0:
                        self.lr_scheduler.last_epoch = self.epoch - 1
                    else:
                        self.lr_scheduler.last_epoch = -1

                    if self._is_master():
                        try:
                            print(f"[lr][align] last_epoch={self.lr_scheduler.last_epoch} "
                                f"base_lrs={self.lr_scheduler.base_lrs}", flush=True)
                        except Exception:
                            pass

                # ----------------------- EPOCH LOOP -----------------------
                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch

                    # Train + validate for one epoch (child controls val_mode)
                    self.train_epoch()

                    # End-of-epoch: optimizer.step() has already happened inside the epoch.
                    # Now advance the scheduler EXACTLY ONCE (no explicit epoch arg).
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # Log AFTER stepping so TB shows post-decay LR
                    if hasattr(self, "_stats_new_epoch"):
                        self._stats_new_epoch()
                    if self._is_master() and hasattr(self, "_write_tensorboard"):
                        self._write_tensorboard()

                    # (optional) quick sanity print
                    if self._is_master() and self.lr_scheduler is not None:
                        try:
                            print(f"[lr][post-step] epoch={self.epoch} "
                                f"lrs={self.lr_scheduler.get_last_lr()}", flush=True)
                        except Exception:
                            pass

                    # Save checkpoint
                    if self._checkpoint_dir:
                        self.save_checkpoint()

                break
            except Exception:
                if not fail_safe:
                    raise
                self.epoch -= 1
                load_latest = True
                import traceback
                if self._is_master():
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')

        
        
        if self.dora_save_every_epoch and self._is_master():
            if self.dora_save_mode in ("merged", "both"):
                self._export_dora_merged(self.epoch)
            if self.dora_save_mode in ("adapters", "both"):
                self._export_dora_adapters_only(self.epoch)
                
        if self._is_master():
            print('Finished training!')



    def train_epoch(self):  # implemented by child
        raise NotImplementedError

    # ---------------- setup ----------------
    def _setup_distributed_and_engine(self):
        self.device, local_rank = _pick_device_from_local_rank()

        model = getattr(self.actor, "net", None)
        if model is None:
            raise AttributeError("actor must have attribute .net (nn.Module).")
        model.to(self.device)

        # DeepSpeed path
        if self.use_deepspeed:
            ds_config = getattr(self.settings, "deepspeed_config", None)
            if ds_config is None:
                ds_config = _build_ds_config(self.settings)
            # Always keep GAS=1 for manual GA
            ds_config["gradient_accumulation_steps"] = 1
            # Respect gradient_clipping as set in ds_config (built above)
            zero_stage = int(ds_config.get("zero_optimization", {}).get("stage", 0))
            ds_config["zero_optimization"] = {"stage": zero_stage}

            if self._is_master():
                print(json.dumps(ds_config, indent=2, sort_keys=True, default=str), flush=True)

            init_out = deepspeed.initialize(
                model=model,
                model_parameters=list(model.parameters()),
                optimizer=self.optimizer,   # AdamW unchanged
                lr_scheduler=None,          # epoch-based outside
                config=ds_config,
            )
            if isinstance(init_out, (list, tuple)):
                self.engine = init_out[0]
                if len(init_out) > 1 and init_out[1] is not None:
                    self.optimizer = init_out[1]
            else:
                self.engine = init_out
            return

        # Native PyTorch DDP (no DeepSpeed)
        if is_dist_initialized() and torch.cuda.is_available():
            ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            self.actor.net = DDP(model, **ddp_kwargs)
        self.engine = None

    # ---------------- checkpoints ----------------
    def _update_settings_for_checkpoints(self, settings=None):
        if settings is not None:
            self.settings = settings

        self._checkpoint_dir = None
        env = getattr(self.settings, 'env', None)
        workspace_dir = getattr(env, 'workspace_dir', None) if env is not None else None
        if workspace_dir is not None:
            workspace_dir = os.path.expanduser(workspace_dir)
            ckpt_root = os.path.join(workspace_dir, 'checkpoints')
            if not os.path.exists(ckpt_root) and self._is_master():
                os.makedirs(ckpt_root, exist_ok=True)
            self._checkpoint_dir = ckpt_root

    def _current_net(self):
        # Always return the real nn.Module (unwrap DS or DDP)
        if self.engine is not None:
            net = self.engine.module
        else:
            net = getattr(self.actor, 'net', None)
            if hasattr(net, 'module'):
                net = net.module
        return net

    def save_checkpoint(self):
        if (not self._is_master()) or self._checkpoint_dir is None:
            return

        net = self._current_net()
        if net is None:
            raise AttributeError('actor.net is required for checkpointing')

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'stats': getattr(self, 'stats', {}),
            'settings': self.settings,
        }

        directory = self._project_directory()
        if directory is None:
            return

        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        if loading is not None and hasattr(loading, 'torch_save'):
            loading.torch_save(state, tmp_file_path)
        else:
            torch.save(state, tmp_file_path)
        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        os.replace(tmp_file_path, file_path)
        if self._is_master():
            print('checkpoint saved: {}'.format(file_path))

    def _project_directory(self):
        proj = getattr(self.settings, 'project_path', 'default')
        directory = '{}/{}'.format(self._checkpoint_dir, proj) if self._checkpoint_dir else None
        if directory and self._is_master() and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        net = self._current_net()
        if net is None:
            raise AttributeError('actor.net is required for checkpointing')

        net_type = type(net).__name__

        if checkpoint is None:
            directory = self._project_directory()
            if directory is None:
                if self._is_master():
                    print('No checkpoint directory set')
                return
            import glob
            pattern = '{}/{}_ep*.pth.tar'.format(directory, net_type)
            checkpoint_list = sorted(glob.glob(pattern))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                if self._is_master():
                    print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            directory = self._project_directory()
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, checkpoint)
        elif isinstance(checkpoint, str):
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError('checkpoint must be None, int, or str')

        if loading is not None and hasattr(loading, 'torch_load_legacy'):
            checkpoint_dict = loading.torch_load_legacy(checkpoint_path)
        else:
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = list(checkpoint_dict.keys())
        if ignore_fields is None:
            ignore_fields = ['settings']
        ignore_fields = list(ignore_fields)
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key], strict=False)
            elif key == 'optimizer':
                opt_state = checkpoint_dict.get('optimizer', None)
                if opt_state is not None and self.optimizer is not None:
                    try:
                        self.optimizer.load_state_dict(opt_state)
                    except Exception:
                        if self._is_master():
                            print('Skip loading optimizer state due to incompatibility.')
            else:
                setattr(self, key, checkpoint_dict[key])

        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']
