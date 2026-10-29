"""
Reconstruct the buoys for notebook 10: every offshore source, with and without wind.

Fills the reconstruction cache (``tmp/reconstruct/``) that notebook 10 reads.
Run from the case folder on the compute node, one setup after another::

    nohup python reconstruct_buoys.py > logs/reconstruct_buoys.log 2>&1 &
    python reconstruct_buoys.py --offshore WHACS_cal --workers 8

Each setup prints its resource proposal (``mm.reconstruction_resources``);
``--workers`` / ``--threads`` override it.
"""

from __future__ import annotations

import argparse
import logging
import time

from utils import metamodel as mm

PERIOD = slice("1979-01-01", "2018-12-31")  # CODEC water level ends in 2018
MDA_N = 550
METAMODEL = "buoys"
OFFSHORE = ["WHACS_cal", "ERA5_cal"]

logger = logging.getLogger("hywaves.reconstruct_buoys")


def run_setup(
    offshore_name: str, wind: bool, workers: int | None, threads: int | None
) -> None:
    """Reconstruct one setup over ``PERIOD`` (spectral, CODEC water level, no spectra saved)."""
    kwargs = dict(
        metamodel=METAMODEL,
        spectral=True,
        offshore_name=offshore_name,
        wl="forcing",
        save_spectra=False,
        wind_correction=wind,
    )
    print(f"\n=== {offshore_name} · wind {'on' if wind else 'off'}", flush=True)
    res = mm.reconstruction_resources(PERIOD, MDA_N, **kwargs)
    if workers is not None:
        res.workers = workers
    if threads is not None:
        res.threads = threads
    t0 = time.perf_counter()
    mm.reconstruct(PERIOD, MDA_N, resources=res, **kwargs)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--offshore", nargs="+", default=OFFSHORE, help="offshore_waves entries"
    )
    parser.add_argument(
        "--wind", nargs="+", choices=["off", "on"], default=["off", "on"]
    )
    parser.add_argument("--workers", type=int, help="override the proposed sum workers")
    parser.add_argument("--threads", type=int, help="override the proposed threads")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    mm.require_models(METAMODEL, MDA_N)
    for offshore_name in args.offshore:
        for wind in args.wind:
            run_setup(offshore_name, wind == "on", args.workers, args.threads)


if __name__ == "__main__":
    main()
