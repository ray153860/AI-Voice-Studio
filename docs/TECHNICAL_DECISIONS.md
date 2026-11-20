# 技术选型与集成决策（2026-11）

## 产品位置

AI Voice Studio 聚焦于把独立音频引擎组织成可复现的项目、任务和素材流程：统一输入输出、任务记录、音频预处理、模型来源与许可标记。它不重做底层神经网络，也不试图取代 Applio、OpenUtau 等专用创作工具的全部编辑能力。

## 首版架构

- FastAPI 本地 API 与轻量网页 UI；Python 负责工作流和音频 I/O。
- SQLite 持久化项目和任务元数据；音频/模型文件存放在用户数据目录，永不默认进入源码仓库。
- 独立进程/HTTP/CLI 引擎适配器，避免把不同 PyTorch/CUDA 栈导入主进程。
- V0.1 的 WAV 校验、峰值归一化和边界淡化是真实音频处理；AI 引擎尚未接入。

## 候选框架

| 能力 | 候选 | 现阶段结论 |
|---|---|---|
| TTS | GPT-SoVITS | MIT 源码仓库；已按 API v2 `/tts` POST 契约实现 HTTP 适配器。当前只通过模拟 HTTP 响应验证契约，未运行真实模型。具体版本、语言模型和权重仍逐项登记授权；CPU 性能与显存取决于模型和版本。 |
| RVC | Applio | 仓库 MIT；已按官方 `core.py infer` CLI 实现独立进程适配器和统一推理任务入口，并为 `preprocess`、`extract`、`train`、`index` 四阶段实现训练适配协议。当前只通过模拟进程验证命令和产物发现，未运行真实模型或训练；发布前仍需钉住可复现版本。 |
| 分离 | Demucs | 采用 adefossez 维护仓库的 CLI 契约；源码 MIT，但官方 README 明确说明预训练权重许可不清晰。本项目只提供独立 CLI 适配器，不打包或再分发权重。UVR 是桌面集成参考，不直接复刻其 UI。 |
| 歌声合成 | OpenVPI DiffSinger | Apache-2.0 源码，模型、声库各自授权。已按官方 `scripts/infer.py acoustic DS_FILE --exp EXP --out DIR --title TITLE` 契约实现独立进程适配器和 DS 文件任务入口；当前仅模拟进程验证，未运行真实模型。 |
| 歌声合成 | NNSVS | MIT，研究工具链，训练/配置和声库准备成本较高；作为替代引擎，不作为首发集成。 |
| 歌声工作台 | OpenUtau | MIT，成熟桌面编辑器和插件生态；项目定位不同，更适合作为兼容/互操作参考，非本软件内嵌 UI。 |
| 同类整合 | UtaiSynthesizer | 功能交叉明显，包含歌声合成、转换、模型管理。其文档清楚区分应用源码与独立权重许可，适合作为功能/许可管理参考；不复制 AGPL 或其组件。 |

## 集成顺序

1. 确立 Engine Adapter 契约、超时/取消、结构化进度、错误和模型声明。
2. 通过独立进程适配 Applio RVC，实测一段合法合成语音样本；比较分块边界连续性和峰值/响度，不将可播放等同于听感验收。
3. 增加分离器并组合翻唱预设工作流；当前编排、真实混音和客观指标已验证，分离/RVC 只通过模拟适配器验证。
4. 以 GPT-SoVITS 独立服务实现参考音频 TTS。
5. 歌声合成单列阶段：先以 DS 文件对接 DiffSinger，再处理歌词音素、MIDI 到 DS 的自动对齐；没有合规可用声库前只报告接口/模拟验证状态。

GPT-SoVITS 适配器通过 `GPT_SOVITS_URL` 配置服务地址。当前 API v2 接受服务器可访问的 `ref_audio_path`，因此本机进程可直接使用工作区路径；远程或容器服务必须显式共享参考音频目录。该限制未满足时不能宣称远程部署可用。

Applio 适配器通过 `APPLIO_ROOT` 和 `APPLIO_PYTHON` 指向独立引擎目录及其 Python。主应用以参数列表调用 `core.py infer`，不使用 shell，也不导入 Applio 的 PyTorch/CUDA 模块。模型必须先登记，并保留来源与许可信息。

Applio 训练适配器依次调用官方 `preprocess`、`extract`、`train`、`index` 命令，限制实验名以阻止路径穿越，并要求最终同时出现推理 `.pth` 与 `.index` 产物。主应用支持多音频数据集登记、SQLite 元数据、后台训练任务和产物复制/校验/登记；会排除 `G_*.pth`、`D_*.pth` 训练检查点，避免将其误认作推理模型。当前取消仍为协作式，阶段内进度解析和真实训练尚未验证。

Demucs 适配器通过 `DEMUCS_EXECUTABLE` 指向独立环境的命令入口，以 `--two-stems=vocals` 生成 vocals/no_vocals。主应用验证两个 WAV 后分别登记输出。混合阶段由本项目实现，负责长度补齐、独立增益和峰值防削波。

DiffSinger 适配器通过 `DIFFSINGER_ROOT` 和 `DIFFSINGER_PYTHON` 指向独立引擎目录及其 Python。任务输入是官方 JSON `.ds` 格式和已存在于该引擎环境中的实验/模型名称；主应用验证生成 WAV，但不捆绑检查点、声库或 vocoder。当前接口只通过模拟外部进程验证。

## 许可规则

AI Voice Studio 的 MIT 许可只覆盖本仓库自行编写代码。第三方源码、二进制、模型权重、声库和数据分别登记许可证、来源、校验值、用途限制与分发规则。默认不下载或分发任何模型权重；优先让用户从权利方来源自行取得。

## 证据链接

- [GPT-SoVITS API](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py) 与 [许可证](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
- [Applio](https://github.com/IAHispano/Applio)
- [Demucs](https://github.com/facebookresearch/demucs)
- [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger)
- [NNSVS](https://github.com/nnsvs/nnsvs)
- [OpenUtau](https://github.com/openutau/OpenUtau)
- [UtaiSynthesizer 权重/NOTICE 说明](https://github.com/yasoukyoku/UtaiSynthesizer/blob/main/NOTICE.md)
