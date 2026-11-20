# 数据集配置

由于文件较大并受到原始许可证约束，本仓库不直接分发原始数据集。

官方数据来源：

- Waymo Open Dataset: https://waymo.com/open/
- DL3DV: https://dl3dv-10k.github.io/DL3DV-10K/
- RealEstate10K: https://google.github.io/realestate10k/

项目期望的数据目录：

    datasets/raw/project4_3d/405841/FRONT/rgb/
    datasets/raw/project4_3d/DL3DV-2/rgb/
    datasets/raw/project4_3d/Re10k-1/images/

如果数据已经位于服务器其他目录，可以建立软链接：

    mkdir -p datasets/raw
    ln -s /absolute/path/to/project4_3d datasets/raw/project4_3d

VGGT和VGGT-Omega导出脚本要求每个处理后的场景至少包含：

    scene_name/images/

脚本会将估计得到的相机和点云写入：

    scene_name/sparse/

请勿向Git提交原始图片、模型权重、深度图、训练检查点或大型PLY文件。
