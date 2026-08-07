# RTX 5090 服务器部署与洋红幕验证

本文面向 Linux + NVIDIA RTX 5090。默认只监听 `127.0.0.1`，通过 SSH 隧道访问，
避免把没有登录鉴权的本地工具直接暴露到公网。

## 1. 系统依赖

服务器需要：

- Python 3.10 或更新版本，并支持创建 `venv`。
- 可被 PyTorch 识别的 NVIDIA 驱动。
- `ffmpeg` 和 `ffprobe`。
- 首次加载 BiRefNet 时可访问 Hugging Face，或提前准备好模型缓存。

## 2. 安装 CUDA AI 环境

在项目目录运行：

```bash
./setup_ai_runtime.sh
```

脚本默认安装 PyTorch CUDA 12.8 wheel，并在结束时打印 `cuda_available` 和显卡名称；
如果 PyTorch 看不到 CUDA，脚本会失败，不会把 CPU 环境误报成 5090 环境可用。

默认运行时和模型缓存位于 `work/models/`。也可以放到独立磁盘：

```bash
export SPRITE_VIDEO_LAB_AI_ROOT=/data/sprite-video-lab-models
export SPRITE_VIDEO_LAB_WORK_DIR=/data/sprite-video-lab-work
./setup_ai_runtime.sh
```

如果服务器要求其他 PyTorch CUDA wheel，可在安装前覆盖：

```bash
export SPRITE_VIDEO_LAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

## 3. 启动与访问

服务器端：

```bash
./start_sprite_video_lab.sh
```

本机建立 SSH 隧道：

```bash
ssh -L 8894:127.0.0.1:8894 <user>@<server>
```

然后在本机浏览器打开 `http://127.0.0.1:8894`。

如需监听局域网或由带鉴权的反向代理接入，可显式设置
`SPRITE_VIDEO_LAB_HOST=0.0.0.0`；不要直接把该端口裸露到公网。

## 4. 洋红背景素材的首轮验证参数

建议先用已保留的 `*-generated-chroma.png` 或 `*-chroma-source.png` 单图验证：

1. 模式选择 `BiRefNet`。
2. 模型选择 `BiRefNet HR-matting`。
3. AI 运行方式选择 `cuda`。
4. AI 边缘精细度先选 `自动（质量优先）`；5090 上可再比较 2048 或 2560。
5. 背景取色先用自动取角落。生成图的洋红色未必是精确 `#ff00ff`，自动取色通常更准。
6. 去色溢出强度先设 `1.0`，Halo 先设 `0`；只有外轮廓仍宽时再试 Halo `1`。
7. 分别在浅色、深色和棋盘格背景检查树叶尖端、草边、木板缝、斜坡细边与半透明像素。

这条路径使用 BiRefNet 生成语义 Alpha，再根据实际洋红键色反解半透明边缘的前景 RGB，
最后把清理后的 RGB 向完全透明区域延展 2 px。它既处理“抠没抠对”，也处理缩放和纹理
过滤后重新出现洋红边的问题。

CorridorKey 只用于绿幕或蓝幕，不用于洋红幕。

## 5. 验证边界

安装脚本只验证 PyTorch 能看到 CUDA。首次加载 BiRefNet 还会下载模型，因此最终验收仍需：

- 页面运行状态显示 CUDA 可用。
- 单帧预览实际使用 `BiRefNet HR-matting` 和 `cuda`。
- 对同一张图比较 Chroma 与 BiRefNet 的透明边缘。
- 导出 PNG 后在目标游戏的实际采样设置下检查边缘；网页预览不能替代运行时纹理过滤验证。
