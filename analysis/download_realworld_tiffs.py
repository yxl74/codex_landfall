#!/usr/bin/env python3
"""
Download a curated real-world TIFF corpus (benign) from public sources.

Targets multiple TIFF "styles":
- classic TIFF test images (varied compressions, photometrics)
- GeoTIFF samples (varied raster data types / tags)
- GDAL autotest GeoTIFF/TIFF fixtures (broad edge-case coverage)

Outputs under data/benign_data/ so it is automatically picked up by the anomaly
feature extractor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_UA = "Mozilla/5.0 (codex_landfall dataset downloader)"


@dataclass(frozen=True)
class Downloaded:
    url: str
    path: Path
    sha256: str
    size: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_bytes(url: str, user_agent: str) -> bytes:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req) as resp:
        return resp.read()


def download(url: str, out_path: Path, user_agent: str, skip_existing: bool) -> Downloaded:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return Downloaded(url=url, path=out_path, sha256=sha256_file(out_path), size=out_path.stat().st_size)
    data = fetch_bytes(url, user_agent)
    out_path.write_bytes(data)
    return Downloaded(url=url, path=out_path, sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def extract_tar_tiffs(tar_gz: Path, dest_root: Path, strip_prefix: Optional[str] = None) -> List[Path]:
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with tarfile.open(tar_gz, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix) :]
            name = name.lstrip("/")
            if not name:
                continue
            if not name.lower().endswith((".tif", ".tiff")):
                continue
            out_path = dest_root / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            out_path.write_bytes(f.read())
            extracted.append(out_path)
    return extracted


def iter_geotiff_sample_urls(base: str, user_agent: str) -> Iterable[str]:
    # base is an Apache directory index like https://download.osgeo.org/geotiff/samples/
    html = fetch_bytes(base, user_agent).decode("utf-8", errors="ignore")
    dirs = re.findall(r'href="([^"]+/)"', html)
    for d in dirs:
        if d in ("../",):
            continue
        sub_url = urljoin(base, d)
        sub = fetch_bytes(sub_url, user_agent).decode("utf-8", errors="ignore")
        files = re.findall(r'href="([^"]+\.(?:tif|tiff))"', sub, flags=re.IGNORECASE)
        for fn in files:
            yield urljoin(sub_url, fn)


def safe_relpath_from_url(url: str, base: str) -> Path:
    # Keep directory structure after base.
    rel = url[len(base) :] if url.startswith(base) else url.split("://", 1)[-1].replace("/", "_")
    rel = rel.lstrip("/")
    return Path(rel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default="data/benign_data/real_world_tiff",
        help="Destination under data/benign_data/",
    )
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--user-agent", default=DEFAULT_UA)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    downloads: List[Downloaded] = []

    # 1) libtiff test images tarball (varied compressions)
    libtiff_url = "https://download.osgeo.org/libtiff/pics-3.8.0.tar.gz"
    libtiff_tar = out_root / "_sources" / "libtiff" / "pics-3.8.0.tar.gz"
    downloads.append(download(libtiff_url, libtiff_tar, args.user_agent, args.skip_existing))
    extracted_libtiff = extract_tar_tiffs(libtiff_tar, out_root / "libtiff_pics", strip_prefix="pics-3.8.0/")

    # 2) GeoTIFF samples directory (real-ish geospatial rasters with different data types)
    geotiff_base = "https://download.osgeo.org/geotiff/samples/"
    geotiff_files = list(iter_geotiff_sample_urls(geotiff_base, args.user_agent))
    extracted_geotiff: List[Path] = []
    for url in geotiff_files:
        rel = safe_relpath_from_url(url, geotiff_base)
        dst = out_root / "geotiff_samples" / rel
        downloads.append(download(url, dst, args.user_agent, args.skip_existing))
        extracted_geotiff.append(dst)

    # 3) GDAL autotest tarball (broad fixture coverage)
    gdal_url = "https://download.osgeo.org/gdal/3.8.0/gdalautotest-3.8.0.tar.gz"
    gdal_tar = out_root / "_sources" / "gdal" / "gdalautotest-3.8.0.tar.gz"
    downloads.append(download(gdal_url, gdal_tar, args.user_agent, args.skip_existing))
    extracted_gdal = extract_tar_tiffs(gdal_tar, out_root / "gdal_autotest", strip_prefix="gdalautotest-3.8.0/")

    # Write a manifest for provenance/debugging.
    manifest = out_root / "_manifest.csv"
    rows = []
    for d in downloads:
        rows.append(
            {
                "url": d.url,
                "path": str(d.path),
                "sha256": d.sha256,
                "bytes": str(d.size),
            }
        )
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "path", "sha256", "bytes"])
        w.writeheader()
        w.writerows(rows)

    print("Output:", out_root)
    print("Manifest:", manifest)
    print("Counts:")
    print(" - libtiff_pics:", len(extracted_libtiff))
    print(" - geotiff_samples:", len(extracted_geotiff))
    print(" - gdal_autotest:", len(extracted_gdal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
