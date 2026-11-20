# 依赖与模型许可证清单

## 项目代码

AI Voice Studio 自行开发的源码采用 MIT 许可证，详见仓库根目录 `LICENSE`。

## 直接 Python 运行依赖

以下许可证依据 2026-11-20 开发环境的已安装发行版元数据核对。实际构建解析到的版本和许可证会由 `python -m voice_studio.compliance --output dependency-licenses.json` 导出，并随 CI/Windows 构建产物保存。

| 发行版 | 开发环境版本 | 元数据许可证 |
|---|---:|---|
| fastapi | 0.165.0 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| soundfile | 0.14.0 | BSD 3-Clause License |
| python-multipart | 0.0.34 | Apache-2.0 |
| httpx | 0.28.1 | BSD-3-Clause |

该表不是完整传递依赖清单。重新分发构建产物前，必须以对应构建生成的 JSON、wheel 自带许可证文件及上游发行说明为准。

## 外部 AI 引擎

GPT-SoVITS、Applio、Demucs、DiffSinger、OpenUtau、NNSVS 等不会作为本仓库运行依赖打入主程序。适配器只通过 HTTP 或独立进程调用用户另行配置的环境。第三方引擎源码继续受各自许可证约束。

## 模型、声库、数据和音频

源码许可证不覆盖模型权重、vocoder、声库、训练数据、参考录音、歌曲或伴奏。项目默认不分发这些内容。模型和数据登记必须记录来源与授权；`registered_unverified`、`trained_unverified` 只描述登记/流程状态，不代表拥有再分发权。

