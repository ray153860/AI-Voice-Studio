from voice_studio.store import Store


def test_startup_recovery_marks_active_jobs_failed(tmp_path):
    store = Store(tmp_path / "studio.sqlite3")
    store.save_job({"id": "running", "status": "running", "log": []})
    store.save_job({"id": "done", "status": "completed", "log": []})

    assert store.recover_incomplete_jobs() == ["running"]
    assert store.get_job("running")["status"] == "failed"
    assert store.get_job("done")["status"] == "completed"
