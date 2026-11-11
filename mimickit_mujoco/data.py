"""Download only the pinned G1 subset from the official MimicKit data bundle."""

import hashlib
import http.cookiejar
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import ROOT

OFFICIAL_URL = "https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa"


def install_archive(archive, root=ROOT):
    root = Path(root)
    manifest = json.loads((ROOT / "data/official_g1_manifest.json").read_text())
    # Validate every selected file before changing the checkout. Paths come
    # exclusively from our manifest, never from arbitrary archive members.
    with zipfile.ZipFile(archive) as z:
        payloads = {}
        for path, digest in manifest.items():
            name = "MimicKit_Data/" + path.removeprefix("data/")
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Official asset checksum changed: {path}; review the upstream revision first")
            dest = root / path
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Preserving locally modified asset: {dest}")
            payloads[dest] = data
        for dest, data in payloads.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    print(f"Verified and installed {len(payloads)} official G1 files")


def fetch_data(archive=None):
    if archive is not None:
        install_archive(archive)
        return
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    print("Opening the official MimicKit public data share...")
    with opener.open(OFFICIAL_URL, timeout=60) as r:
        page = r.read().decode("utf-8")
    match = re.search(r'"\.downloadUrl"\s*:\s*("[^"\n]+")', page)
    if not match:
        raise RuntimeError("Official share did not expose a download URL; use a downloaded archive with --archive")
    url = json.loads(match[1])
    print("Downloading official bundle (~384 MiB); installing only G1 assets and motions...")
    with tempfile.TemporaryDirectory(prefix="mimickit-data-") as tmp:
        archive = Path(tmp) / "MimicKit_Data.zip"
        with opener.open(url, timeout=60) as source, archive.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        install_archive(archive)
