# Windows 构建与运行

## 当前验证边界

GitHub Actions 在 `windows-latest` 上执行 Python 测试，并使用 PyInstaller 生成 one-folder 构建。构建工作流会隐藏启动程序、轮询 `http://127.0.0.1:8765/api/capabilities`，确认 HTTP 200 后停止进程并上传未签名构建产物。

这证明构建产物可在 GitHub 的 Windows runner 启动，不等同于已在所有 Windows 11 设备、显卡驱动或 AI 引擎组合上完成兼容性验收。

## 源码运行

在隔离环境安装项目后执行：

```powershell
python -m voice_studio
```

或使用安装后生成的命令：

```powershell
ai-voice-studio
```

服务只监听 `127.0.0.1:8765`。浏览器访问该地址即可使用工作台。

## 构建

构建依赖被放在独立的 `build` 可选依赖组中：

```powershell
python -m pip install -e '.[build]'
pyinstaller --noconfirm --clean --onedir --name AI-Voice-Studio --collect-all soundfile --collect-all uvicorn --collect-all fastapi src/voice_studio/launcher.py
```

公开 CI 产物未签名，不应绕过 Windows 安全提示。正式签名需要独立代码签名证书和用户授权，目前不属于已完成能力。

外部 GPT-SoVITS、Applio、Demucs、DiffSinger 环境和模型不会被打入主程序，仍须按各自许可证和兼容版本独立配置。
