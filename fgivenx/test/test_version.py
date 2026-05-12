import pathlib
from fgivenx import _version


def test_read_version_fallback_to_metadata(monkeypatch):
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)
    monkeypatch.setattr(_version, "version", lambda pkg: "9.9.9")
    assert _version._read_version() == "9.9.9"
