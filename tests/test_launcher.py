from voice_studio import launcher


def test_launcher_binds_only_to_localhost(monkeypatch):
    captured = {}

    def fake_run(application, **options):
        captured["application"] = application
        captured["options"] = options

    monkeypatch.setattr(launcher.uvicorn, "run", fake_run)

    launcher.main()

    assert captured["application"] is launcher.app
    assert captured["options"] == {"host": "127.0.0.1", "port": 8765}
