"""Fetch GNU-Unifont (full BMP coverage) for the glyph content stencil.

PIL's default font renders many BMP glyphs — including Swedish ``Ä Å Ö ä å ö`` — as ``.notdef`` boxes,
which silently corrupts the content-conditioning signal (measured: 16/101 chars in the GT corpus). The
renderers auto-resolve ``<repo>/assets/unifont.otf`` (or ``$DIFFU_UNIFONT_PATH``); run this once to fetch it.

    python scripts/get_unifont.py
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

INDEX = "https://unifoundry.com/pub/unifont/"
ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    index = urllib.request.urlopen(INDEX, timeout=30).read().decode()  # noqa: S310 (trusted host)
    version = sorted(set(re.findall(r"unifont-(\d+\.\d+\.\d+)", index)), key=lambda v: [int(p) for p in v.split(".")])[-1]
    url = f"{INDEX}unifont-{version}/font-builds/unifont-{version}.otf"  # plane-0 (BMP) build
    out = ASSETS / "unifont.otf"
    print(f"downloading GNU-Unifont {version} -> {out}")
    urllib.request.urlretrieve(url, out)  # noqa: S310 (trusted host)
    print(f"done: {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
