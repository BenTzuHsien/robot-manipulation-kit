# robot-manipulation-kit
 
Template scripts, utilities, and reference solutions for a real-robot manipulation curriculum using a **6-DoF Lite6 robotic arm** and a **ZED 2i stereo camera**.

## Overview

![](assets/robotics_course_developer_teaser.gif)

This kit supports a 10-stage manipulation curriculum that progressively increases perception difficulty across two stages:
 
**Stage 1 — AprilTag-Based Manipulation (Checkpoints 1–5)**
Object pose estimation assisted by AprilTags. Tasks include basic grasping, pick and place, target selection, stacking, and sequential stacking.
 
**Stage 2 — Pure Vision Manipulation (Checkpoints 6–10)**
The same tasks are repeated without AprilTags, requiring a full RGB-D perception pipeline using point clouds and image segmentation.

## Hardware
 
| Component | Details |
|---|---|
| Robot Arm | UFactory Lite6 (6-DoF), controlled via xArm Python SDK |
| Camera | ZED 2i stereo camera (RGB + point cloud) |
| Arena Poster | Custom-printed poster with four reference AprilTags, used for camera-to-robot calibration. ([Download](assets/arena_poster.pdf)) |

## Dependencies
 
- **NVIDIA Driver** — 570-open
- **CUDA** — 12.8
- **TensorRT** — 10.9
- **[ZED SDK](https://www.stereolabs.com/developers/release)** — CUDA 12 / TensorRT 10 build
- **[xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)** — `pip install xarm-python-sdk`
- **[SAM3](https://github.com/facebookresearch/sam3)** — with CUDA 12.8 PyTorch build
- **Python packages** — `cupy-cuda12x`, `opencv-python`, `pyopengl`, `pyopengl-accelerate`, `scipy`, `pupil-apriltags`, `open3d`, `einops`, `decord`, `pycocotools`, `psutil`
- **Python** — 3.12 (via Mamba/Miniforge recommended)

## Checkpoints
 
Each checkpoint script (`checkpoint1.py` – `checkpoint10.py`) contains skeleton code with `TODO` sections for students to complete. Checkpoint 0 (`checkpoint0.py`) is fully provided and handles camera-to-robot calibration using four reference AprilTags on the arena poster.
 
Reference solutions are available in the `solution/` directory.
 
## Reference Solution Highlights
 
The reference solutions use SAM3-based segmentation and oriented bounding box fitting on point clouds for zero-shot target selection and 6D object pose estimation in Stage 2.
 
## License
 
MIT