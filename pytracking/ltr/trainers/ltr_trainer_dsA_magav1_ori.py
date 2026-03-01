# -*- coding: utf-8 -*-
import os
import time
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from ltr.trainers.base_trainer_dsA_magav1 import BaseTrainerDSA, _NullTBWriter

warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*indexing argument")


def freeze_batchnorm_layers(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def _to_device(sample, device):
    """Recursively move batch samples (tensors, dicts, lists/tuples) to a device."""
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    if isinstance(sample, dict):
        return {k: _to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, (list, tuple)):
        items = [_to_device(v, device) for v in sample]
        return type(sample)(items) if not isinstance(sample, tuple) else tuple(items)
    return sample


class LTRTrainer_magav1(BaseTrainerDSA):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None,
                 freeze_backbone_bn_layers=False, use_GradScaler=False):
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self.freeze_backbone_bn_layers = freeze_backbone_bn_layers
        self.use_GradScaler = use_GradScaler
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

        # Validation mode: "rank0" (parity) or "distributed"
        self.val_mode = str(getattr(self.settings, 'val_mode', 'rank0')).lower()
        assert self.val_mode in ('rank0', 'distributed')

        # Manual GA steps (trainer-level)
        self.ga_steps = int(getattr(self.settings, 'manual_gradient_accumulation_steps', 1))
        self.ga_steps = max(1, self.ga_steps)

        # Defaults
        default = {'print_interval': 1, 'print_stats': None, 'description': ''}
        for k, v in default.items():
            if getattr(self.settings, k, None) is None:
                setattr(self.settings, k, v)

        # Stats per loader
        self.stats = OrderedDict({loader.name: OrderedDict() for loader in self.loaders})

        # TensorBoard (rank 0 only)
        try:
            from ltr.admin.tensorboard import TensorboardWriter
            # Build a safe directory even if project_path/tensorboard_dir are missing
            proj = getattr(self.settings, "project_path", "default")
            tb_root = getattr(self.settings.env, "tensorboard_dir", ".")
            tb_dir = os.path.join(tb_root, proj)
            os.makedirs(tb_dir, exist_ok=True)  # ensure the folder exists

            if self._is_master():
                self.tensorboard_writer = TensorboardWriter(tb_dir, [l.name for l in self.loaders])
            else:
                self.tensorboard_writer = _NullTBWriter()
        except Exception as e:
            # Do NOT silently swallow: tell the user why TB is off.
            if self._is_master():
                print(f"[tensorboard] disabled: {type(e).__name__}: {e}")
            self.tensorboard_writer = _NullTBWriter()


    # ---------- epoch loop ----------
    def train_epoch(self):
        for loader in self.loaders:
            is_val = not loader.training

            # rank-0 only validation (parity)
            if is_val and self.val_mode == 'rank0' and not self._is_master():
                if self._dist_ready():
                    torch.distributed.barrier()
                continue

            if self.epoch % loader.epoch_interval == 0:
                self._cycle_loader(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

        # self._write_tensorboard()
        # self._stats_new_epoch()


    # ---------- inner loop ----------
    def _cycle_loader(self, loader):
        is_val = not loader.training

        if is_val:
            self.actor.eval()
            torch.set_grad_enabled(False)
        else:
            self.actor.train(True)
            torch.set_grad_enabled(True)

        # Freeze BN in backbone feature extractor only (match your DP)
        if self.freeze_backbone_bn_layers:
            fe = getattr(getattr(self.actor, "net", None), "feature_extractor", None)
            if fe is not None:
                freeze_batchnorm_layers(fe)

        self._init_timing()

        # Manual GA: zero once before loop; zero again only at boundary
        if loader.training:
            self._zero_grad()

        num_batches = len(loader)

        # Window accumulators for unbiased logging when GA > 1
        window_weighted_sum = None  # dict[name] -> weighted sum over window
        window_sample_count = 0     # total (global) samples over window

        for i, data in enumerate(loader, 1):  # i is 1-based
            if self.move_data_to_gpu:
                data = _to_device(data, self.device)

            # augment batch meta
            if isinstance(data, dict):
                data['epoch'] = self.epoch
                data['settings'] = self.settings

            # forward
            loss, stats = self.actor(data)

            # ---------- strict SUM/COUNT aggregation for logging ----------
            # Compute per-GPU batch size
            local_batch_sz = self._current_batch_size(data)
            cnt_local = float(local_batch_sz)

            # ---- FPS counters (per-GPU): count every iteration here
            # self.num_frames += int(cnt_local)
            # self.frames_since_prev += int(cnt_local)


            if is_val and self.val_mode == 'rank0':
                # Rank-0 only validation (no cross-rank reduction)
                global_cnt = cnt_local
                loss_log = float(loss.detach().item())
                reduced_stats = {}
                for name, val in stats.items():
                    v = float(val.detach().mean().item()) if isinstance(val, torch.Tensor) else float(val)
                    reduced_stats[name] = v
            else:
                # Distributed/sample-weighted reduction to global mean per stat
                loss_sum_local = float(loss.detach().item()) * cnt_local

                if self._dist_ready():
                    pair = torch.tensor([loss_sum_local, cnt_local], dtype=torch.float32, device=self.device)
                    torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
                    global_loss_sum = float(pair[0].item())
                    global_cnt = float(pair[1].item())
                    loss_log = global_loss_sum / max(global_cnt, 1.0)
                else:
                    global_cnt = cnt_local
                    loss_log = loss_sum_local / max(global_cnt, 1.0)

                reduced_stats = {}
                for name, val in stats.items():
                    v = float(val.detach().mean().item()) if isinstance(val, torch.Tensor) else float(val)
                    v_sum_local = v * cnt_local
                    if self._dist_ready():
                        pair = torch.tensor([v_sum_local, cnt_local], dtype=torch.float32, device=self.device)
                        torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
                        v_mean = float(pair[0].item() / max(pair[1].item(), 1.0))
                    else:
                        v_mean = v_sum_local / max(cnt_local, 1.0)
                    reduced_stats[name] = v_mean

            # Normalize TB loss key
            if 'loss' in reduced_stats and 'Loss/total' not in reduced_stats:
                reduced_stats['Loss/total'] = reduced_stats.pop('loss')
            if 'Loss/total' not in reduced_stats:
                reduced_stats['Loss/total'] = loss_log

            # ---------- training (manual GA) ----------
            if loader.training:
                # Exact averaging: full windows scale by ga_steps; final partial window scales by its true size
                last_window_size = num_batches % self.ga_steps
                is_in_last_partial = (last_window_size != 0) and (i > num_batches - last_window_size)
                divisor = float(last_window_size if is_in_last_partial else self.ga_steps)

                micro_loss = loss / divisor
                self._backward(micro_loss)


            # ---------- logging accumulation (build window sums) ----------
            if loader.training:
                if window_weighted_sum is None:
                    window_weighted_sum = {k: float(v) * global_cnt for k, v in reduced_stats.items()}
                else:
                    for k, v in reduced_stats.items():
                        window_weighted_sum[k] = window_weighted_sum.get(k, 0.0) + float(v) * global_cnt
                window_sample_count += int(global_cnt)

            # ---------- step / clip / zero ----------
            need_step = loader.training and ((i % self.ga_steps == 0) or (i == num_batches))
            if need_step:
                # Clip exactly once:
                # - If DS grad clipping is selected, do NOT clip here (DS will clip in engine.step()).
                # - If torch grad clipping is selected, clip here before step().
                use_torch_clip = bool(getattr(self.settings, "use_torch_grad_clip", False))
                if use_torch_clip:
                    max_norm = float(getattr(self.settings, "gradient_clipping",
                                             getattr(self.settings, "grad_clip_norm", 0.8)))
                    module = self.engine.module if self.engine is not None else self.actor.net
                    if hasattr(module, "module"):  # unwrap DDP
                        module = module.module
                    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, norm_type=2)

                self._step()
                self._zero_grad()

                # Flush window stats ONCE per optimizer step with proper weighting
                if window_weighted_sum is not None:
                    averaged_stats = {k: (window_weighted_sum[k] / max(window_sample_count, 1))
                                      for k in window_weighted_sum.keys()}
                    self._update_stats(averaged_stats, window_sample_count, loader)
                    window_weighted_sum, window_sample_count = None, 0

            # ---------- logging (per-iter prints) ----------
            if not loader.training:
                self._update_stats(reduced_stats, int(global_cnt), loader)
                if self._is_master() and (i % self.settings.print_interval == 0 or i == len(loader)):
                    self._print_stats(i, loader, loader.batch_size)
            else:
                if self._is_master() and (i % self.settings.print_interval == 0 or i == len(loader)):
                    self._print_stats(i, loader, loader.batch_size)

        # Final flush for any partial GA window so epoch stats are complete
        if loader.training and window_weighted_sum is not None:
            averaged_stats = {
                k: (window_weighted_sum[k] / max(window_sample_count, 1))
                for k in window_weighted_sum.keys()
            }
            self._update_stats(averaged_stats, window_sample_count, loader)

            # --- add this debug line here ---
            if self._is_master():
                print(f"[flush] epoch={self.epoch} final_window_samples={window_sample_count}")

            window_weighted_sum, window_sample_count = None, 0
            
        if is_val and self._dist_ready():
            torch.distributed.barrier()

    # ---------- helpers ----------
    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _current_batch_size(self, data):
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    return int(v.size(0))
        return int(getattr(self.settings, "batch_size", 1))

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        if self.stats.get(loader.name, None) is None:
            from ltr.admin.stats import AverageMeter
            self.stats[loader.name] = OrderedDict({n: AverageMeter() for n in new_stats.keys()})
        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                from ltr.admin.stats import AverageMeter
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time

        if self._is_master() and (i % self.settings.print_interval == 0 or i == len(loader)):
            print_str = f'[{loader.name}: {self.epoch}, {i} / {len(loader)}] '
            print_str += f'FPS: {average_fps:.1f} ({batch_fps:.1f})  ,  '
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, "avg"):
                    print_str += f'{name}: {val.avg:.5f}  ,  '
            print(print_str[:-5], flush=True)

    def _stats_new_epoch(self):
        # (optional) record LR here if desired, before rolling
        for loader in self.loaders:
            if loader.training and hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                    for i, lr in enumerate(lr_list):
                        var_name = f'LearningRate/group{i}'
                        if var_name not in self.stats[loader.name]:
                            from ltr.admin.stats import StatValue
                            self.stats[loader.name][var_name] = StatValue()
                        self.stats[loader.name][var_name].update(lr)
                except Exception:
                    pass

        # roll meters for the next epoch
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        # Write run metadata exactly once per run (even if starting at a later epoch)
        if not hasattr(self, "_tb_info_written") and hasattr(self, "tensorboard_writer"):
            try:
                self.tensorboard_writer.write_info(self.settings.module_name,
                                                self.settings.script_name,
                                                self.settings.description)
            except Exception:
                pass
            self._tb_info_written = True

        # Write epoch metrics
        try:
            self.tensorboard_writer.write_epoch(self.stats, self.epoch)
            if hasattr(self.tensorboard_writer, "flush"):
                self.tensorboard_writer.flush()
        except Exception:
            pass