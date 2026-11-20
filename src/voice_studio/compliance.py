import argparse
import json
from importlib import metadata
from pathlib import Path

RUNTIME_DISTRIBUTIONS = ["fastapi", "uvicorn", "numpy", "soundfile", "python-multipart", "httpx"]


def dependency_metadata(distributions: list[str]) -> list[dict[str, str]]:
    packages = []
    for name in distributions:
        try:
            details = metadata.metadata(name)
            version = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"dependency distribution is not installed: {name}") from error
        license_name = details.get("License-Expression") or details.get("License")
        if not license_name:
            classifiers = details.get_all("Classifier", [])
            licenses = [value.removeprefix("License :: OSI Approved :: ") for value in classifiers if value.startswith("License :: OSI Approved :: ")]
            license_name = "; ".join(licenses) or "UNKNOWN"
        packages.append({"name": name, "version": version, "license": license_name})
    return packages


def write_dependency_report(output: Path, distributions: list[str] | None = None) -> None:
    report = {
        "scope": "resolved direct Python dependencies; transitive and model licenses remain separate",
        "packages": dependency_metadata(distributions or RUNTIME_DISTRIBUTIONS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export resolved direct dependency license metadata")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_dependency_report(arguments.output)


if __name__ == "__main__":
    main()
