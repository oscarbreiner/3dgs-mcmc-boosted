# ScanNet++ integration notes for 3dgs-mcmc-boosted

## Scope
This repo has native ScanNet++ loading in the main training pipeline.
No import from `scannetpp/` is required for `train.py` runtime.

## Runtime integration in this repo

ScanNet++ dataset detection is in `scene/__init__.py` and triggers when any of these exist under `--source_path`:
- `dslr/nerfstudio/transforms.json`
- `dslr/nerfstudio/transforms_undistorted.json`
- `iphone/nerfstudio/transforms.json`

The main loader is `readScanNetPPSceneInfo(...)` in `scene/dataset_readers.py`.
It:
- reads Nerfstudio transforms,
- prefers `dslr/resized_undistorted_images` when `--images images` is used,
- supports `init_type=random`, `init_type=sfm`, and `init_type=pc_aligned`.

## Dataset access

Use the official ScanNet++ dataset page for download instructions, split definitions, and release details:
- https://scannetpp.mlsg.cit.tum.de/scannetpp/

Dataset access is controlled by the ScanNet++ maintainers and may require requesting access.

## Required files by mode

Describe requirements as scene-relative paths (`<SCENE_DIR>/...`):

- Camera metadata and images (required for all modes):
- `dslr/nerfstudio/transforms_undistorted.json` or `dslr/nerfstudio/transforms.json`
- image directory referenced by `--images` (commonly `dslr/resized_undistorted_images`)

- Initialization-specific files:
- `init_type=sfm`: `dslr/colmap/points3D.bin` or `dslr/colmap/points3D.txt` (or prebuilt `points3D.ply`)
- `init_type=pc_aligned`: `scans/pc_aligned.ply`
- `init_type=random`: no preexisting point cloud file required

## Path and secret hygiene

Use repo-relative or environment-variable based paths in docs/commands:

```bash
export SCANNETPP_ROOT="${PWD}/datasets/scannetpp"
```

Then train with:

```bash
python train.py \
  --source_path "${SCANNETPP_ROOT}/data/<SCENE_ID>" \
  --images dslr/resized_undistorted_images \
  --init_type sfm \
  --eval \
  --cap_max <CAP>
```

## Term glossary

- `source_path`: one scene directory (`.../data/<SCENE_ID>`), not the dataset root.
- `transforms.json` / `transforms_undistorted.json`: camera intrinsics + poses in Nerfstudio format.
- `init_type=sfm`: initialize Gaussians from COLMAP sparse points (`points3D.*`).
- `init_type=random`: initialize Gaussians from random points in scene bounds.
- `init_type=pc_aligned`: initialize from ScanNet++ aligned point cloud (`scans/pc_aligned.ply`).
