# FaceSwap_DreamID-V

Experimental DreamID-V B long-video runner with protected-context latent anchoring, generated overlap context, resumable windows and original audio. No decoded face pasteback or enhancer. This repository contains code only: no personal media, model weights, SSH keys, credentials or private render state.

## Environment
Linux or WSL2 with NVIDIA GPU passthrough; tested target RTX 5090 32GB. Keep the environment, caches, checkpoints and media on a spacious data disk. WSL ext4 on a data drive is preferred for this Linux pipeline. A resumed render may need around 35-50GB of free storage, plus a margin.

```bash
export FS_ROOT=/data/faceswap/experiments
export UV_CACHE_DIR=$FS_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$FS_ROOT/python
mkdir -p "$FS_ROOT"
python3 -m venv "$FS_ROOT/bootstrap"
"$FS_ROOT/bootstrap/bin/pip" install uv
UV="$FS_ROOT/bootstrap/bin/uv"
$UV python install 3.12
$UV venv --python 3.12 "$FS_ROOT/dreamidv/.venv"
$UV pip install --python "$FS_ROOT/dreamidv/.venv/bin/python" torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
$UV pip install --python "$FS_ROOT/dreamidv/.venv/bin/python" -r requirements.txt
```

FFmpeg/ffprobe must be installed; optional FS_FFMPEG overrides /usr/bin/ffmpeg. Do not install a Linux NVIDIA display driver in WSL: GPU passthrough uses the Windows host driver.

## Private state migration and resume
Transfer the private state archive separately over SSH and extract it under FS_ROOT. It must include the frozen upstream source, CP01 manifest/snapshots, source reference image, prepared crop/pose/mask/audio, window manifest and completed window directories. Obtain pretrained weights from the pinned repositories in the transferred metadata:

```bash
python3 download_weights.py
# Link the verified weights directory as checkpoints/CP01_before_hair_AB_20261030/weights.
python3 B5_job.py
```

B5_job skips windows with validated done.json markers. Copy all completed windows, especially the last completed raw.mp4: the next window uses its last nine frames as known context. Each window is 73 frames, stride 64, 20 sampling steps and guidance 4. The existing long-video task is 4500 frames at 15fps (300 seconds). Slide-only sections bypass generation. Inference requires substantial free VRAM; the coordinator waits for sufficient memory and retries after CUDA OOM.

Read dreamidv/B_5min_run01/status.json for progress. Final output is deliverables/B_5min.mp4, a 512x512 head-following crop with original audio, not a reconstructed full 1920x1080 frame. comparison_5min.mp4 provides the corresponding original crop. Temporal seams, identity, hairstyles and lip motion require visual review; the pipeline does not guarantee production quality.

## Upstream and scope
Frozen DreamID-V revision: 9b589940577559c91481fb3a13bae000a55f97a1, https://github.com/bytedance/DreamID-V (Apache-2.0). Weight revisions and SHA-256 values are preserved in private weights.json. Modified inference clamps non-editable latents to the known video trajectory each flow-matching step. CP01 rollback data is preserved separately and is never published. This repository does not redistribute or relicense pretrained weights or upstream code.
