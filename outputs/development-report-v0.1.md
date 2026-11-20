# AI Voice Studio V0.1 开发报告

日期：2026-11-20

## 已实现

- 可启动的 FastAPI 本地 Web 工作台。
- 项目创建与 SQLite 持久化。
- 音频上传、格式解码校验、WAV 输出、峰值归一化和 10ms 边界淡化。
- 后台任务状态、日志、历史、取消请求、重试和服务重启后的未完成任务恢复。
- 输出音频下载与浏览器试听。
- RVC、GPT-SoVITS、唱声模型的基础登记：文件扩展名、大小、SHA-256、来源、许可证说明。
- TTS、音色转换、人声分离和唱声合成的统一引擎请求/结果协议。
- GPT-SoVITS API v2 `/tts` HTTP 适配器与统一 TTS 任务入口；需要外部服务和合法模型。
- Applio `core.py infer` 外部进程适配器与统一 RVC 任务入口；需要独立 Applio 环境和合法 RVC 模型。
- Demucs 两轨分离适配器；代码接口已完成，权重不随项目分发。
- 真实 WAV 人声/伴奏混合：长度对齐、独立增益和峰值防削波。
- 标准 MIDI 解析：PPQ、BPM、音符、速度、时长、休止和歌词元事件；当前不生成唱声。
- Ubuntu/Windows GitHub Actions 测试配置。

## 验证状态

- 真实音频处理：已验证。测试生成 WAV，经处理后可解码、采样率和长度保持正确。
- 服务启动：已验证。Uvicorn 启动成功，首页与能力 API 返回 HTTP 200。
- API 集成：已验证项目、上传、任务、输出、重试、取消、模型登记和错误输入。
- 自动测试：25 项通过。
- 静态检查：Ruff 通过；`git diff --check` 通过。
- 真实 AI 推理：未验证，也未标记为完成。
- Windows GitHub Actions：已配置，尚未在远端运行。

## 技术选型

- 主应用：Python + FastAPI + SQLite + NumPy/SoundFile。
- 引擎隔离：优先采用独立进程或 HTTP 服务适配，避免不同 CUDA/PyTorch 版本进入主进程。
- TTS：优先研究 GPT-SoVITS 独立 API。
- RVC：优先研究 Applio 的稳定 CLI/服务入口。
- 分离：评估 Demucs 的维护分支与模型权重许可。
- 原创唱声：优先研究 OpenVPI DiffSinger；OpenUtau 用于格式与工作流互操作参考。

## 尚未完成

- GPT-SoVITS、Applio RVC 和 Demucs 的真实模型推理验证，以及 DiffSinger 适配器。
- 训练进程、进度解析、取消和训练产物登记。
- MIDI/歌词/音素对齐与原创唱声生成。
- GPU 验证、主观试听、音质对比和 Windows 原生打包。
- 歌词到音素/音节的对齐和真实唱声模型推理。
- GitHub 账户内同名仓库核验、远端 CI 和发布。

## 已知问题

- 测试出现 Starlette 对当前 TestClient/httpx 组合的弃用警告，不影响现有测试结果。
- 当前取消属于协作式取消；短任务可能在取消请求到达前完成。
- 模型登记只验证元数据和文件扩展名，状态固定为 `registered_unverified`，不会声称模型可推理。
- 早期运行曾在 `C:\Users\hp\.ai-voice-studio` 创建项目 SQLite 和空数据目录；代码已改为项目内 `work/runtime`。自动删除该旧目录被执行环境的删除保护拒绝。

## 仓库状态

当前为本地 Git 仓库，尚未配置 GitHub 远端。没有可用 GitHub 账户连接，因此未核验或创建远端仓库，也没有声称已发布。
