# 第三方依赖

本仓库只保存本项目编写的脚本、实验结果和补丁，不重复上传完整的第三方仓库。

需要使用以下项目：

- VGGT: https://github.com/facebookresearch/vggt
- VGGT-Omega: https://github.com/facebookresearch/vggt-omega
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- LightGlue: https://github.com/jytime/LightGlue
- GeoQuery: https://github.com/Xiaoc7/GeoQuery

各仓库在本项目中使用的提交版本记录在 revisions.tsv。

GeoQuery补丁说明：

1. geoquery_offline_model.patch
   允许通过环境变量 SD_TURBO_PATH 离线加载SD-Turbo。

2. geoquery_torch23_compat.patch
   使softsplat兼容PyTorch 2.3的AMP接口。

在GeoQuery仓库根目录应用补丁：

    git apply /path/to/project-repo/patches/geoquery_offline_model.patch
    git apply /path/to/project-repo/patches/geoquery_torch23_compat.patch

本仓库不包含模型权重。请从各项目的官方页面下载权重，并通过命令行参数或环境变量指定本地路径。
