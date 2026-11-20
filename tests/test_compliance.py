import json

import pytest

from voice_studio.compliance import dependency_metadata, write_dependency_report


def test_dependency_report_records_resolved_version_and_license(tmp_path):
    output = tmp_path / "licenses.json"

    write_dependency_report(output, ["fastapi", "uvicorn"])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert [item["name"] for item in report["packages"]] == ["fastapi", "uvicorn"]
    assert all(item["version"] and item["license"] for item in report["packages"])
    assert report["scope"] == "resolved direct Python dependencies; transitive and model licenses remain separate"


def test_dependency_metadata_rejects_missing_distribution():
    with pytest.raises(RuntimeError, match="not installed"):
        dependency_metadata(["definitely-not-an-installed-package"])
