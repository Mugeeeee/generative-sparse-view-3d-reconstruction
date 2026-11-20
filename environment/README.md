# Environment Setup

The project was tested on Linux with NVIDIA RTX 3090 GPUs.

## Tested System

| Component | Version |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.3.1+cu121 |
| CUDA toolkit | 12.2 |
| GPU architecture | 8.6 |
| NumPy | 1.26.1 |
| OpenCV | 4.11.0 |
| gsplat | 1.5.3 |
| diffusers | 0.25.1 |
| transformers | 4.38.0 |
| CuPy | 13.6.0 |

We used separate environments because the reconstruction, 3DGS, and GeoQuery pipelines have different CUDA extension requirements.

## Recommended Environments

### 1. VGGT environment

Used for:

- VGGT inference
- VGGT-Omega inference
- COLMAP model export
- Camera trajectory evaluation

Important packages include:

- torch
- numpy
- opencv-python
- trimesh
- pycolmap
- kornia
- LightGlue

Install the upstream VGGT and VGGT-Omega repositories from `third_party/repos/` according to their official instructions.

### 2. 3DGS environment

Used for:

- Gaussian Splatting training
- Rendering
- PSNR, SSIM, and LPIPS evaluation

The original 3DGS CUDA extensions must be compiled:

- simple-knn
- diff-gaussian-rasterization
- fused-ssim

Example build configuration:

    export CUDA_HOME=/usr/local/cuda-12.2
    export PATH="$CUDA_HOME/bin:$PATH"
    export TORCH_CUDA_ARCH_LIST=8.6
    export MAX_JOBS=4

Follow the installation instructions in the pinned Gaussian Splatting repository.

### 3. GeoQuery environment

Used for:

- gsplat training
- GeoQuery refinement
- SD-Turbo inference

Important packages include:

- torch 2.3.1
- gsplat 1.5.3
- diffusers 0.25.1
- transformers 4.38.0
- peft 0.9.0
- cupy-cuda12x 13.6.0
- ninja

Apply the included patches before running GeoQuery:

    cd third_party/repos/GeoQuery
    git apply ../../../patches/geoquery_offline_model.patch
    git apply ../../../patches/geoquery_torch23_compat.patch

For offline SD-Turbo loading:

    export SD_TURBO_PATH=/path/to/sd-turbo
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

For gsplat CUDA JIT compilation, ensure Ninja is installed:

    python -m pip install ninja

## Validation Commands

Check PyTorch and CUDA:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

Check the 3DGS CUDA extensions:

    python -c "import simple_knn, diff_gaussian_rasterization, fused_ssim; print('3DGS extensions OK')"

Check the GeoQuery environment:

    python -c "import torch, cupy, diffusers, transformers, gsplat; print('GeoQuery dependencies OK')"

## Important Notes

- Do not mix system Python packages with a project environment.
- Use `python -m pip` instead of calling `pip` directly.
- The CUDA toolkit used to build extensions should be compatible with the installed PyTorch CUDA runtime.
- Large environments, wheels, checkpoints, and compiled artifacts are not stored in this Git repository.
