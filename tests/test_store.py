from voice_studio.store import Store


def test_startup_recovery_marks_active_jobs_failed(tmp_path):
    store = Store(tmp_path / "studio.sqlite3")
    store.save_job({"id": "running", "status": "running", "log": []})
    store.save_job({"id": "done", "status": "completed", "log": []})

    assert store.recover_incomplete_jobs() == ["running"]
    assert store.get_job("running")["status"] == "failed"
    assert store.get_job("done")["status"] == "completed"


def test_training_dataset_metadata_is_persistent(tmp_path):
    database = tmp_path / "studio.sqlite3"
    store = Store(database)
    dataset = {"id": "dataset-1", "name": "Authorized voice", "files": 2, "license": "owned"}

    store.save_dataset(dataset)

    reopened = Store(database)
    assert reopened.get_dataset("dataset-1") == dataset
    assert reopened.list_datasets() == [dataset]


def test_audio_asset_metadata_is_persistent_and_filtered_by_project(tmp_path):
    database = tmp_path / "studio.sqlite3"
    store = Store(database)
    project_asset = {"id": "asset-1", "project_id": "project-1", "name": "voice.wav"}
    other_asset = {"id": "asset-2", "project_id": "project-2", "name": "song.wav"}

    store.save_audio_asset(project_asset)
    store.save_audio_asset(other_asset)

    reopened = Store(database)
    assert reopened.get_audio_asset("asset-1") == project_asset
    assert reopened.list_audio_assets("project-1") == [project_asset]
    assert reopened.list_audio_assets() == [other_asset, project_asset]
