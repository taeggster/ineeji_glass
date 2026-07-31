"""Dump the parquet files in data/ to CSV, verbatim.

No resampling, no cleaning, no column changes -- one CSV per parquet, same
rows, same columns, same values.

    python data.py                 # convert every .parquet in data/
    python data.py data/foo.parquet  # or just the ones you name
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent / "data"
BATCH_ROWS = 100_000  # stream so the 124-column file never lands in RAM whole


def to_csv(src: Path) -> Path:
    dst = src.with_suffix(".csv")
    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    done = 0

    with open(dst, "w", newline="", encoding="utf-8") as fh:
        header = True
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            batch.to_pandas().to_csv(fh, index=False, header=header)
            header = False
            done += batch.num_rows
            print(f"  {done:,}/{total:,} rows", end="\r", flush=True)

    size_mb = dst.stat().st_size / 1e6
    print(f"  {done:,} rows -> {dst.name} ({size_mb:,.0f} MB)")
    return dst


def main(argv: list[str]) -> int:
    sources = [Path(a) for a in argv] or sorted(DATA_DIR.glob("*.parquet"))
    if not sources:
        print(f"no .parquet files found in {DATA_DIR}", file=sys.stderr)
        return 1

    for src in sources:
        if not src.exists():
            print(f"missing: {src}", file=sys.stderr)
            return 1
        print(src)
        to_csv(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
