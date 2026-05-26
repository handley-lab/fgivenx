from pathlib import Path
from importlib.metadata import version


def _read_version():
    readme = Path(__file__).resolve().parent.parent / "README.rst"
    if readme.exists():
        for line in readme.read_text().splitlines():
            if ":Version:" in line:
                return line.split(":")[2].strip()
    return version("fgivenx")


__version__ = _read_version()
