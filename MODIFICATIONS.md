# Modifications in this fork (Wan2GP-mac)

This repository is a fork of [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP), modified to run better on Apple Silicon Macs (MPS).
It is **not** an official WanGP release, and its changes are **not endorsed** by the WanGP Licensor. The WanGP Community License 2.0 (`LICENSE.txt`) applies unchanged.
These notes record the modifications as required by section 4.1(d) of that license.

## 2026-10-22: the same time-axis pad fix for Wan Uni3C and the H3 control mask (`mac-v0.2.5`)

The MPS fault below is exact: `F.pad` on the time axis alone (spatial pad zero), at the front or the end, constant or replicate,
is wrong once a frame has 65,536 pixels or more (65,535: correct). Pads of space and time together are correct, so the Wan, Qwen
Image, LongCat and Hunyuan VAEs are not affected; the H3 transformer's frame pad (3D) and LongCat's latent pad are not either.
1. **`shared/kernels/vae_ops.py`: `pad_front_frames` became `pad_frames(x, front=0, back=0, mode="constant"|"replicate")`**
   (MPS: `torch.cat` of zero frames or edge-frame copies; other devices: `F.pad` as before). `pad_front_frames` remains, calling it.
2. **`models/wan/camera_control/uni3c.py`**: the condition chunks' leading empty frames (on the GPU, wrong on MPS from 480x832 up).
3. **`models/minimax_h3/pipeline.py` (`_resize_video_mask`)**: the control mask's trailing replicated frames. The mask is in RAM
   today, so this was correct already; the change keeps it correct if it moves to the GPU.
Tested against CPU `F.pad` for every form at 255x257, 256x256 and 576x1024 (lab `tools/test_pad_frames.py`: all pass). Uni3C
has not been run on a Mac.

## 2026-10-22: MiniMax H3 video VAE encodes pictures correctly on MPS (`mac-v0.2.4`)

1. **`shared/kernels/vae_ops.py`: new `pad_front_frames(x, front)`**, used by `frame_group_norm` and by
   **`models/minimax_h3/components/video_autoencoder.py`** (`MiniMaxH3VideoCausalConv3d.forward`) for the causal zero frames.
   PyTorch 2.14.1 on MPS corrupts a constant `F.pad` on the time axis of a 5D tensor once a frame exceeds about 4k pixels (the
   copied frames come back wrong; spatial padding is fine; 8x8 and 64x64 frames are fine, 290x514 is not). On MPS the zero frames are
   now joined with `torch.cat`; other devices keep `F.pad`, so NVIDIA and AMD behave as before.
   Effect: every H3 picture encode on the Mac was wrong (start/end frames, reference images, control videos). FL2VA rendered a
   flat brown field because it copies the start frame's latent into frame 0; Ref2VA only got a loose redraw of its references.
   Measured (512x288 start frame, encode then decode): latent cosine against a CPU fp32 encode went from −0.47 (fp32) / NaN (fp16)
   to 1.0000 untiled and 0.9867 tiled, the same as CPU tiled against CPU untiled; the decoded picture SSIM went from 0.19 to 0.53,
   equal to the CPU encode's.

## 2026-10-21: merged upstream `fdc63bdf`, H3 audio continuity (`mac-v0.2.3`)

Upstream's one commit (2026-10-19, 10 files) merged into `mac-fixes` with no conflicts; our MPS edits in `pipeline.py` and the
`torchcodec` pin are untouched. It brings: H3 sound that runs 0.1 s past each window so sliding windows and Continue Video join
cleanly (saved H3 videos keep that 0.1 s tail); an "End Image Strength" setting for H3 FL2VA and Ref2VA (default 1 = as before);
Discard Last Frames in steps of 1 for H3; a CUDA-only GGUF fix; `tensordict` dropped from `requirements.txt` (it stays installed in
`env_torch214`, unused). The changed files compile in `env_torch214`.

**Not yet tested by a render.** The planned test (lab PLAN.md, Phase U3: a two-window H3 FL2VA clip, listening at the join) is on
hold: the FL2VA path renders a flat brown field on MPS (MPT `tests/t-sf1/RESULT.md`, sf4, on the code before this merge), so that
test would show nothing about the sound. It runs after the FL2VA fix, or on Ref2VA if the owner prefers.

## 2026-10-18: MiniMax H3 runs on MPS (`mac-v0.2.2`)

1. **`models/minimax_h3/pipeline.py`: the pipeline's `device` property returns `mps` on Apple Silicon.**
   Before: without CUDA it returned the transformer's parameter device, which is the CPU under mmgp, so H3 tried to compute on the CPU.
   The `cuda` branch is still checked first, so NVIDIA and AMD behave as before.
2. **`models/minimax_h3/pdd.py` and `models/minimax_h3/lora_affine.py`: move to the CPU first, then cast to float64.**
   PyTorch 2.14.1 on MPS returns all zeros for a one-call `t.to(device="cpu", dtype=torch.float64)`; other dtypes, and moving then casting, are correct.
   With the device fix above, the PDD sigma schedule is built on MPS, and the zeroed copy failed the "strictly descending sigma boundaries" check.
   Values and dtypes are unchanged on every device.

Measured on an M4 Max (128 GB, torch 2.14.1, no `--preload`): MiniMax H3 Ref2VA Pruned PDD 8-step, 1024×576, 107 frames with 4 reference images,
finished in 39m 43s (about 4 min per denoising step), lowest free memory 38%, with video and stereo audio.

## 2026-10-18: MPS weight-transfer fix (`mac-v0.2.1`)

**`shared/mps/device_patch.py`: `torch.cuda.Event` and `torch.cuda.Stream` are real barriers on MPS.**
Before: the MPS shim replaced them with no-ops (`synchronize()` did nothing, `query()` always returned True). The mmgp bundled since v17 copies pageable
weights to the GPU through two reusable pinned staging slots and waits on an event before refilling a slot. With the no-op event, the CPU overwrote
a slot before MPS had copied it, so models ran with scrambled weights: LTX-2.5 produced noise after the first frames.
A 64 MB transfer through mmgp's `_PinnedStager` was corrupted 5 times out of 5.
Now: event `synchronize()`, `query()` and `wait()`, and stream `synchronize()` and `wait_event()`, call `torch.mps.synchronize()`. The transfer test is exact 5/5.
LTX-2.5 and Qwen Image Edit 2511 renders match their 2026-10-03 baselines. Denoising is slower than on `mac-v0.1.0` (LTX +27%, Qwen 2-step +61%),
but whole jobs are faster (LTX 530 s → 163 s, Qwen 736 s → 676 s) because model loading is much faster. This only affects MPS; `apply_mps_patch()` does not run elsewhere.

## 2026-10-18: merged upstream v17–v18 (`mac-v0.2.0`)

**Broken on MPS; use `mac-v0.2.1`.** See above.


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
