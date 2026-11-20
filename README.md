# AI Voice Studio

模块化、本地优先的 AI 音声工作流桌面应用。当前为 V0.1 开发预览。

## 当前状态

- 已实现：本地 Web 工作台、项目/音频文件记录、WAV 校验与转换、真实音频处理与混合、含音高范围的客观质量筛查/前后对比、任务状态和日志、取消请求、全部现有工作流按原始参数重试、模型文件登记与校验、MIDI 解析，以及 GPT-SoVITS、Applio 推理/训练、Demucs、DiffSinger 独立引擎适配协议。
- 已通过真实音频处理验证：WAV 导入/校验、标准化处理、输出可解码。
- AI 翻唱预设已串联人声分离、RVC、质量比较和伴奏混合；编排与真实混音已验证，两个 AI 引擎仍仅通过模拟适配器验证。
- 翻唱转换支持可选的重叠分块与交叉淡化拼接，保持原始帧数并报告拼接边界最大采样跳变；当前只通过模拟 RVC 分块验证。
- 仅模拟适配器验证：TTS、RVC、分离、唱声合成接口。
- 尚未实现真实 AI 推理、真实模型训练、Windows 安装程序及代码签名；未签名 one-folder 构建已在 GitHub Windows runner 启动验证。
- Applio 训练已包含授权音频数据集登记、官方四阶段后台任务和训练产物自动登记；当前只通过模拟外部进程验证，没有执行真实训练。
- 原创演唱已提供单声部 MIDI 与歌词音节对齐、休止/节拍时间线和 DiffSinger DS 文件任务入口；当前尚未完成语言音素化、自动 DS 生成和真实唱声验证。

## 运行

Python 3.11+。在隔离开发环境安装 `pip install -e .` 后运行 `python -m voice_studio`，浏览器打开 `http://127.0.0.1:8765`。仅需 FastAPI、Uvicorn、NumPy、SoundFile；没有大型模型或 GPU 依赖。开发期数据默认写入仓库的 `work/runtime`；可通过 `AI_VOICE_STUDIO_DATA_DIR` 指定独立数据目录。

外部引擎分别通过 `GPT_SOVITS_URL`、`APPLIO_ROOT`/`APPLIO_PYTHON`、`DEMUCS_EXECUTABLE`、`DIFFSINGER_ROOT`/`DIFFSINGER_PYTHON` 配置。主应用不捆绑这些引擎、CUDA、模型或声库。

Windows 源码运行、隔离构建和验证边界见 [`docs/WINDOWS.md`](docs/WINDOWS.md)。GitHub Actions 会生成未签名 one-folder 构建并做本地 API 冒烟测试；这不等同于已完成所有 Windows 11 硬件兼容或代码签名验收。

## 许可

项目代码采用 MIT。模型权重、数据集与用户音频不随仓库分发；使用者须核实各自来源授权。直接依赖和模型授权边界见 [`docs/DEPENDENCY_LICENSES.md`](docs/DEPENDENCY_LICENSES.md)，每次 CI/Windows 构建还会保存实际解析版本的 JSON 报告。
