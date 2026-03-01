#!/usr/bin/env python3
# ltr/run_training_dsA.py
import os, sys, shlex, socket, contextlib, subprocess, argparse, shutil, importlib, random

# Ensure repo root on sys.path so "import ltr" works for all ranks
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../ltr
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # project root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


if any(a.startswith("--local_rank") for a in os.sys.argv[1:]) or os.environ.get("LOCAL_RANK") is not None:
    try:
        import torch
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(lr)     # choose GPU for this rank first
            _ = torch.cuda.device(lr)     # touch context on the right GPU
    except Exception:
        pass
# ------------------------------------------------------------------------


# ---------- free port helpers ----------
def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True

def pick_master_port(start: int = 29500, end: int = 29999, host: str = "127.0.0.1") -> int:
    candidates = list(range(start, end + 1))
    random.shuffle(candidates)
    for p in candidates:
        if _port_is_free(p, host=host):
            return p
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
# ---------------------------------------

def _has_local_rank(argv):
    for a in argv:
        if a.startswith("--local_rank"):
            return True
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    return False

def _parse_worker_args(argv):
    parser = argparse.ArgumentParser(description="Worker shim for run_training.py", add_help=False)
    parser.add_argument("--local_rank", type=int, default=None)  # swallowed
    parser.add_argument("--backend", type=str, default=None, choices=["ddp", "ds0"])
    parser.add_argument("train_module", type=str)
    parser.add_argument("train_name", type=str)
    parser.add_argument("--cudnn_benchmark", type=int, default=1)
    args, _unknown = parser.parse_known_args(argv[1:])
    return args

def _run_worker(argv) -> int:
    args = _parse_worker_args(argv)
    from ltr import run_training as rt
    # Pass through the same args to your standard entry
    return rt.run_training(args.train_module, args.train_name, bool(args.cudnn_benchmark))

def _gpu_list_from_env():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip() == "":
        return []
    return [x for x in cvd.split(",") if x.strip() != ""]

def _resolve_backend(cli_backend: str, env_backend: str, train_module: str, train_name: str) -> str:
    # Priority: CLI > ENV > settings.BACKEND > default 'ds0'
    if cli_backend in ("ddp", "ds0"):
        return cli_backend
    if env_backend in ("ddp", "ds0"):
        return env_backend

    # Try to import the train settings module and read BACKEND if present
    try:
        mod_path = f"ltr.train_settings.{train_module}.{train_name}"
        cfg = importlib.import_module(mod_path)
        b = getattr(cfg, "BACKEND", None)
        if isinstance(b, str) and b.lower().strip() in ("ddp", "ds0"):
            return b.lower().strip()
    except Exception:
        pass

    return "ds0"

def _run_launcher(argv) -> int:
    if len(argv) < 3:
        print("Usage: CUDA_VISIBLE_DEVICES=0,1 python ltr/run_training_dsA.py <train_module> <train_name> [--backend ddp|ds0] [extra args]")
        return 1

    # Parse only to fetch --backend (do not consume other args)
    parsed = _parse_worker_args(argv)
    train_module = parsed.train_module
    train_name = parsed.train_name
    extra = argv[3:]  # keep user extras intact

    # rendezvous
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(pick_master_port())

    # Build env with PYTHONPATH containing repo root
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    if REPO_ROOT not in py_path.split(os.pathsep):
        env["PYTHONPATH"] = (REPO_ROOT if not py_path else REPO_ROOT + os.pathsep + py_path)

    # Backend choice (CLI > ENV > settings.BACKEND > default)
    backend = _resolve_backend(parsed.backend, env.get("LTR_BACKEND", "").lower().strip(), train_module, train_name)

    ds_cli = shutil.which("deepspeed")
    gpus = _gpu_list_from_env()
    nproc = len(gpus) if gpus else 1

    if backend == "ddp":
        # Force native torchrun even if deepspeed is installed
        cmd = [
            sys.executable, "-u", "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--master_addr", env["MASTER_ADDR"],
            "--master_port", env["MASTER_PORT"],
            "ltr/run_training_dsA.py",
            "--backend", "ddp",
            train_module,
            train_name,
            *extra,
        ]
        launcher = "torchrun"
    else:
        # backend == "ds0": prefer deepspeed CLI; fallback to torchrun
        if ds_cli is not None:
            cmd = [
                "deepspeed",
                "--master_port", env["MASTER_PORT"],
                "ltr/run_training_dsA.py",
                "--backend", "ds0",
                train_module,
                train_name,
                *extra,
            ]
            launcher = "deepspeed"
        else:
            cmd = [
                sys.executable, "-u", "-m", "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "--master_addr", env["MASTER_ADDR"],
                "--master_port", env["MASTER_PORT"],
                "ltr/run_training_dsA.py",
                "--backend", "ds0",
                train_module,
                train_name,
                *extra,
            ]
            launcher = "torchrun"

    print("\n[run_training_dsA] Launcher:", launcher)
    print("[run_training_dsA] Backend:", backend)
    print("[run_training_dsA] Command:")
    print(" ", " ".join(shlex.quote(c) for c in cmd))
    print("[run_training_dsA] MASTER_ADDR =", env.get("MASTER_ADDR"))
    print("[run_training_dsA] MASTER_PORT =", env.get("MASTER_PORT"))
    print("[run_training_dsA] CUDA_VISIBLE_DEVICES =", env.get("CUDA_VISIBLE_DEVICES"))
    print("[run_training_dsA] PYTHONPATH includes repo root =", REPO_ROOT in env.get("PYTHONPATH", "")); print()
    proc = subprocess.run(cmd, env=env)
    return proc.returncode

def main():
    # If this is a worker process launched by torchrun/deepspeed, it will pass --local_rank
    if _has_local_rank(sys.argv):
        rc = _run_worker(sys.argv)
    else:
        rc = _run_launcher(sys.argv)
    sys.exit(rc)

if __name__ == "__main__":
    main()
