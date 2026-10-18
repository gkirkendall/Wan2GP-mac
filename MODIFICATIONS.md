# Modifications in this fork (Wan2GP-mac)

This repository is a fork of [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP), modified to run better on Apple Silicon Macs (MPS).
It is **not** an official WanGP release, and its changes are **not endorsed** by the WanGP Licensor. The WanGP Community License 2.0 (`LICENSE.txt`) applies unchanged.
These notes record the modifications as required by section 4.1(d) of that license.

## 2026-10-18: merged upstream v17–v18 (`mac-v0.2.0`)

Upstream `main` was merged at `22d1f261` (39 commits: WanGP v17 and v18). There were no conflicts. The four changes from 2026-10-03 below carry over unchanged, and they now apply against `22d1f261`.
No new modifications were made.

Notes for Mac installs:
- mmgp is now bundled in the repository (`mmgp/`), not installed with pip. Existing environments should run `pip uninstall mmgp` and then `pip install -r requirements.txt`.
- mmgp's new allocator (`mmgp/allocator`) is CUDA-only and stays inactive on MPS. The WanGP UI starts cleanly on an M4 Max with torch 2.14.1.
- `shared/mps/test_mps_forward.py` (upstream) no longer matches the current `WanModel.forward` signature, and this was already the case before the merge. It fails before the forward pass runs, so its result says nothing about MPS support.

## 2026-10-03: Apple Silicon installer and dependency fixes (`mac-fixes` branch)

Changes against upstream `main` at `b8b18f81`:

1. **`setup_config.json`: pinned torch stack for the `mps` profile.**
   Before: the profile installed unpinned `torch torchvision torchaudio`, which resolved to a mismatched set (torch 2.14.1 with torchaudio 2.11.0, beside the
   `torchcodec==0.10.0` pin). Now: `torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0`.
   Measured on an M4 Max (128 GB): torch 2.14 runs MPS attention as a fused, memory-efficient kernel (no extra memory at 16k tokens, where torch 2.10 ran out of memory),
   and real models run **about 2.3–2.4x faster per step** than on torch 2.10 (Qwen Image Edit 2511, LTX-2.5), with 22–29% lower peak memory.
2. **`requirements.txt`: platform-specific torchcodec pin.** `torchcodec==0.10.0` is kept for non-macOS platforms (unchanged behavior),
   and macOS gets `torchcodec==0.17.0`, the build matching torch 2.14. torchcodec 0.17 ships wheels for **macOS 14 or later**.
3. **`setup.py:get_system_specs()`: memory detection on macOS.** `/proc/meminfo` and `nvidia-smi` do not exist on macOS, so the installer defaulted to
   16 GB RAM / 8 GB VRAM. It now reads `sysctl -n hw.memsize` and reports unified memory as both RAM and VRAM.
4. **`setup.py:create_wgp_config()`: memory profile for unified memory.** For the `MPS` hardware profile it selects profile 3 above 60 GB,
   profile 4 above 30 GB, and profile 5 otherwise. Profiles 1 and 2 keep a reserved-RAM copy of the weights next to the "VRAM" copy, which on unified memory
   holds the model twice in the same pool.

Notes:
- The fused MPS attention kernel in torch 2.14 requires **macOS 26.2 or later**. Older macOS versions fall back to PyTorch's previous kernels.
- No NVIDIA, AMD or Windows code paths were changed.
