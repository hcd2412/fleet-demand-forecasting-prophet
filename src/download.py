from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DownloadConfig:
    dataset_id: str = "4b4i-vvec"
    start: str = "2023-01-01T00:00:00"
    end: str = "2023-03-31T23:59:59"
    select_cols: tuple[str, ...] = ("tpep_pickup_datetime",)
    limit: int = 50_000
    out_csv: Path = PROJECT_ROOT / "data/raw/yellow_taxi_2023_jan_mar.csv"


def build_socrata_csv_url(cfg: DownloadConfig) -> str:
    base = f"https://data.cityofnewyork.us/resource/{cfg.dataset_id}.csv"
    params = {
        "$select": ",".join(cfg.select_cols),
        "$where": (
            f"tpep_pickup_datetime >= '{cfg.start}' "
            f"AND tpep_pickup_datetime <= '{cfg.end}'"
        ),
        "$limit": str(cfg.limit),
    }
    return base + "?" + urlencode(params)


def download_csv(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 0
    page = 0
    wrote_header = False

    while True:
        paged_url = url + f"&$offset={offset}"
        print(f"[download] page={page} offset={offset} GET {paged_url}")

        r = requests.get(paged_url, timeout=60)
        r.raise_for_status()

        text = r.text.strip()
        if not text or text == "tpep_pickup_datetime":
            print("[download] done")
            break

        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            print("[download] done")
            break

        df.to_csv(
            out_path,
            index=False,
            mode=("w" if not wrote_header else "a"),
            header=(not wrote_header),
        )
        wrote_header = True

        rows = len(df)
        print(f"[download] wrote rows={rows} -> {out_path}")

        offset += rows
        page += 1

        if page > 500:
            raise RuntimeError("Pagination exceeded 500 pages.")


def quick_sanity_check(path: Path) -> None:
    df = pd.read_csv(path, nrows=5)
    print("[check] head:")
    print(df.head())
    print("[check] columns:", list(df.columns))


def main() -> None:
    cfg = DownloadConfig()
    url = build_socrata_csv_url(cfg)
    download_csv(url, cfg.out_csv)
    quick_sanity_check(cfg.out_csv)


if __name__ == "__main__":
    main()
