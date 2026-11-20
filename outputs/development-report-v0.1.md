# AI Voice Studio V0.1 开发报告

日期：2026-11-20

## 已实现

- 可启动的 FastAPI 本地 Web 工作台。
- 项目创建与 SQLite 持久化。
- 音频上传、格式解码校验、WAV 输出、峰值归一化和 10ms 边界淡化。
- 全部上传入口使用分块写入和超限清理：音频单文件 512 MiB、训练数据集总量 4 GiB、MIDI/DS 文档 16 MiB、模型文件 4 GiB；双文件工作流与数据集导入发生中途失败时会清理已写入文件。
- 后台任务状态、日志、历史、取消请求、全部现有工作流按原始参数重试，以及服务重启后的未完成任务恢复。
- 输出音频下载与浏览器试听。
- RVC、GPT-SoVITS、唱声模型的基础登记：文件扩展名、大小、SHA-256、来源、许可证说明。
- TTS、音色转换、人声分离和唱声合成的统一引擎请求/结果协议。
- GPT-SoVITS API v2 `/tts` HTTP 适配器与统一 TTS 任务入口；需要外部服务和合法模型。
- Applio `core.py infer` 外部进程适配器与统一 RVC 任务入口；需要独立 Applio 环境和合法 RVC 模型。
- Applio 官方 `preprocess`、`extract`、`train`、`index` 四阶段训练工作流：授权音频数据集登记、SQLite 元数据、后台任务、`.pth`/`.index` 产物校验和自动模型登记；当前只通过模拟外部进程验证，未执行真实训练。
- Demucs 两轨分离适配器；代码接口已完成，权重不随项目分发。
- DiffSinger `scripts/infer.py acoustic` 外部进程适配器、DS 文件任务入口和图形界面提交控件；需要独立引擎环境、合法模型和 vocoder。
- 真实 WAV 人声/伴奏混合：长度对齐、独立增益和峰值防削波。
- AI 翻唱预设编排：歌曲导入、人声/伴奏分离、RVC 转换、转换前后客观质量比较、伴奏混合与成品试听；AI 阶段只通过模拟适配器验证，混合和质量分析是真实执行。
- 可选重叠分块转换：保持输入采样率、声道数和总帧数，交叉淡化拼接并记录边界最大采样跳变；只通过模拟 RVC 分块验证。
- 标准 MIDI 解析：PPQ、BPM、音符、速度、时长、休止和歌词元事件；当前不生成唱声。
- 单声部歌词对齐：中文逐字、英文按词、`-` 延音标记，歌词音节数与 MIDI 音符数严格校验；状态明确为未音素化、未合成。
- 音频质量筛查与处理前后比较：峰值、RMS、削波、静音占比、内部静音段、采样突变，以及单声部音频的有声比例、中位音高和音域半音数；结果不替代主观试听。
- Ubuntu/Windows GitHub Actions 测试配置。
- Windows PyInstaller one-folder 构建、打包后 API 冒烟测试和未签名构建产物上传工作流；首次远端构建、启动与上传均已成功。
- 直接 Python 运行依赖许可证清单与自动 JSON 导出；CI 和 Windows 构建保存实际解析版本，同时明确不覆盖传递依赖、外部 AI 引擎、模型、声库和数据授权。
- Bandit 源码扫描与 pip-audit 依赖审计工作流；两者输出 JSON 附件及日志摘要，发现项会阻断流程。提交 `a8ba77d` 的首次远端扫描已成功。

## 验证状态

- 真实音频处理：已验证。测试生成 WAV，经处理后可解码、采样率和长度保持正确。
- 服务启动：已验证。Uvicorn 启动成功，首页与能力 API 返回 HTTP 200。
- API 集成：已验证项目、上传、任务、输出、全部现有工作流重试、取消请求、模型登记和错误输入。
- 自动测试：63 项通过，并在 `-W error` 模式下无警告。
- 静态检查：Ruff 通过；`git diff --check` 通过。
- 真实 AI 推理：未验证，也未标记为完成。
- GitHub Actions：Ubuntu 与 Windows 两个任务均已远端运行并成功。
- Windows 构建：提交 `d041fea` 已完成 PyInstaller 打包、隐藏启动、能力 API HTTP 200 冒烟测试和产物上传。产物 `AI-Voice-Studio-windows-unsigned` 为 30,802,575 字节，摘要 `sha256:afb9ecd81208f671f042ea16e4be7ee7c84eb339ac86d221705ddda84e52f2d8`。

## 技术选型

- 主应用：Python + FastAPI + SQLite + NumPy/SoundFile。
- 引擎隔离：优先采用独立进程或 HTTP 服务适配，避免不同 CUDA/PyTorch 版本进入主进程。
- TTS：优先研究 GPT-SoVITS 独立 API。
- RVC：优先研究 Applio 的稳定 CLI/服务入口。
- 分离：评估 Demucs 的维护分支与模型权重许可。
- 原创唱声：优先研究 OpenVPI DiffSinger；OpenUtau 用于格式与工作流互操作参考。

## 尚未完成

- GPT-SoVITS、Applio RVC、Demucs 和 DiffSinger 的真实模型推理验证。
- 训练阶段内进度解析、可中断子进程取消、断点续训验证和真实 GPU 训练。
- 语言音素化、MIDI/歌词到 DiffSinger DS 自动转换与原创唱声生成。
- GPU 验证、主观试听、真实 AI 音质对比和 Windows 安装程序。
- 中文/英文歌词音素化、发音词典适配和真实唱声模型推理。
- Windows 11 多设备验收、安装程序和代码签名发布。

## 已知问题

- 当前取消属于协作式取消；短任务可能在取消请求到达前完成。
- 模型登记只验证元数据和文件扩展名，状态固定为 `registered_unverified`，不会声称模型可推理。
- 早期运行曾在 `C:\Users\hp\.ai-voice-studio` 创建项目 SQLite 和空数据目录；代码已改为项目内 `work/runtime`。自动删除该旧目录被执行环境的删除保护拒绝。

## 仓库状态

已核验账户内不存在同名仓库，并创建公开独立仓库：<https://github.com/ray153860/AI-Voice-Studio>。

本地完整提交历史已推送到 `main`。Windows 构建提交 `d041fea` 的常规 CI 与独立打包流程均成功：<https://github.com/ray153860/AI-Voice-Studio/actions/runs/44722571074>、<https://github.com/ray153860/AI-Voice-Studio/actions/runs/44722571001>。
