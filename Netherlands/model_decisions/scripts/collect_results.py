"""Tidy tables (``results/*.csv``) from the printed outputs in ``results/raw/``.

The scores were computed by the scripts in ``scripts/swan`` and ``scripts/snapwave``
against the production runs of 2026-11 (since replaced), so they are kept as
printed and parsed here. Run from ``Netherlands/model_decisions/``:

    python scripts/collect_results.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"

CELL = re.compile(r"([+-]?\d+\.\d+)/\s*(\d+\.\d+)\s*\(\s*(\d+)\)")
RATIO_ROW = re.compile(r"^(\d\.\d\d-\d\.\d\d)\s+(deep|<18m)\s+(.*)$")

#: raw file -> one entry per score table in it (in order): experiment, hours, labels.
EXPERIMENTS: dict[str, list[dict]] = {
    "results_2026-11-02.txt": [
        {
            "experiment": "swan_source_terms",
            "hours": "39 test hours of 2013",
            "labels": {
                "SnapWave": "SnapWave production (old WHACS_cal)",
                "base": "SWAN Westhuysen (old WHACS_cal)",
                "komen": "SWAN Komen (old WHACS_cal)",
                "st6": "SWAN ST6 (old WHACS_cal)",
            },
        }
    ],
    "results_2026-11-02_round2.txt": [
        {
            "experiment": "swan_forcing_numerics",
            "hours": "39 test hours of 2013",
            "labels": {
                "SnapWave": "SnapWave production (old WHACS_cal)",
                "komen": "SWAN Komen (old WHACS_cal)",
                "era5cal": "SWAN Komen (ERA5_cal)",
                "whacs_map": "SWAN Komen (WHACS_cal)",
                "wl_field": "SWAN Komen (old WHACS_cal), water-level field",
                "bkd": "SWAN Komen (old WHACS_cal), BREAKING BKD",
            },
        }
    ],
    "results_2026-11-02_round3.txt": [
        {
            "experiment": "swan_friction",
            "hours": "39 test hours of 2013",
            "labels": {
                "SnapWave": "SnapWave production (old WHACS_cal)",
                "komen": "SWAN Komen (old WHACS_cal)",
                "era5cal": "SWAN Komen (ERA5_cal)",
                "era5cal_fric": "SWAN Komen (ERA5_cal), friction 0.019",
                "whacs_map": "SWAN Komen (WHACS_cal)",
                "whacs_map_fric": "SWAN Komen (WHACS_cal), friction 0.019",
            },
        }
    ],
    "results_2026-11-02_round3_wind.txt": [
        {
            "experiment": "swan_wind",
            "hours": "39 test hours of 2013",
            "labels": {
                "no wind": "SWAN no wind (WHACS_cal)",
                "wind": "SWAN Komen (WHACS_cal)",
                "wind+fric": "SWAN Komen (WHACS_cal), friction 0.019",
            },
        },
        {
            "experiment": "swan_no_wind_500h",
            "hours": "500 hours of 2013 (stratified by offshore Hs)",
            "labels": {
                "no wind": "SWAN no wind (WHACS_cal)",
                "SnapWave": "SnapWave production (old WHACS_cal)",
            },
        },
    ],
    "results_2026-11-03_round4.txt": [
        {
            "experiment": "swan_120h",
            "hours": "120 other hours of 2013",
            "labels": {
                "whacs_038": "SWAN Komen (WHACS_cal), friction 0.038",
                "whacs": "SWAN Komen (WHACS_cal), friction 0.019",
                "era5": "SWAN Komen (ERA5_cal), friction 0.019",
            },
        }
    ],
    "results_2026-11-03_wind110.txt": [
        {
            "experiment": "swan_wind_input",
            "hours": "39 test hours of 2013",
            "labels": {
                "SnapWave": "SnapWave production (old WHACS_cal)",
                "whacs_map_fric": "SWAN Komen (WHACS_cal), friction 0.019",
                "whacs_fric_agrow": "SWAN Komen AGROW (WHACS_cal), friction 0.019",
                "whacs_fric_wind110": "SWAN Komen (WHACS_cal), friction 0.019, wind x 1.1",
                "era5cal_fric": "SWAN Komen (ERA5_cal), friction 0.019",
                "era5cal_fric_wind110": "SWAN Komen (ERA5_cal), friction 0.019, wind x 1.1",
            },
        }
    ],
    "snapwave_results_2026-11-03.txt": [
        {
            "experiment": "snapwave_setup",
            "hours": "39 test hours of 2013",
            "labels": {
                "base": "SnapWave (WHACS_cal), AoI rectangle",
                "enc": "SnapWave (WHACS_cal), open-boundary enclosure",
                "enc_upw": "SnapWave (WHACS_cal), enclosure + upwindref 1",
                "enc_upw_fw02": "SnapWave (WHACS_cal), enclosure + upwindref 1, fw 0.02",
                "enc_upw_fw04": "SnapWave (WHACS_cal), enclosure + upwindref 1, fw 0.04",
                "enc_upw_nowind": "SnapWave no wind (WHACS_cal), enclosure + upwindref 1",
                "whacs_map": "SWAN Komen (WHACS_cal)",
            },
        }
    ],
}


def score_tables(text: str) -> list[pd.DataFrame]:
    """Every ``label  bias/rmse (n) ...`` table of a printed output, in order."""

    lines = text.splitlines()
    tables, i = [], 0
    while i < len(lines):
        if CELL.search(lines[i]) and i >= 1 and not CELL.search(lines[i - 1]):
            names = re.split(r"\s{2,}", lines[i - 1].strip())
            rows = []
            while i < len(lines) and (CELL.search(lines[i]) or not lines[i].strip()):
                if lines[i].strip():
                    cells = CELL.findall(lines[i])
                    label = lines[i][: CELL.search(lines[i]).start()].strip()
                    rows += [
                        {"variant": name, "group": label, "bias": float(b), "rmse": float(r), "n": int(n)}
                        for name, (b, r, n) in zip(names, cells)
                    ]
                i += 1
            tables.append(pd.DataFrame(rows))
        i += 1
    return tables


def buoy_scores() -> pd.DataFrame:
    """Hs vs 27 buoys, bias/RMSE (m) per experiment, variant and group."""

    frames = []
    for name, experiments in EXPERIMENTS.items():
        tables = score_tables((RAW / name).read_text())
        for meta, table in zip(experiments, tables):
            table = table[table["variant"].isin(meta["labels"])].copy()
            table.insert(0, "experiment", meta["experiment"])
            table.insert(1, "hours", meta["hours"])
            table.insert(3, "label", table["variant"].map(meta["labels"]))
            table["source"] = name
            frames.append(table)
    return pd.concat(frames, ignore_index=True)


def ratio_rows(text: str, names: list[str]) -> list[dict]:
    """``steepness  where  v1 v2 ...`` rows after the ``steepness  where`` header, columns *names*."""

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(r"steepness\s+where", line))
    rows = []
    for line in lines[start + 1 :]:
        match = RATIO_ROW.match(line)
        if not match:
            break
        for name, value in zip(names, match.group(3).split()):
            rows.append({"steepness": match.group(1), "where": match.group(2), "variant": name, "ratio": float(value)})
    return rows


def metamodel_ratios() -> pd.DataFrame:
    """Hs ratios on the 40 metamodel test cases (energy weighted, sites with >= 25 % of the offshore Hs)."""

    frames = []
    # round 1: SWAN default and OFF WCAP against SnapWave production
    lines = (RAW / "results_2026-11-02.txt").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("steepness"))
    rows = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) != 6:
            break
        for (where, variant), value in zip(
            [("deep", "SWAN default"), ("deep", "SWAN OFF WCAP"), ("<18m", "SWAN default"), ("<18m", "SWAN OFF WCAP")],
            parts[2:],
        ):
            rows.append({"steepness": parts[0], "where": where, "variant": variant, "ratio": float(value)})
    frames.append(pd.DataFrame(rows).assign(experiment="swan_whitecapping", reference="SnapWave production"))

    names = ["SWAN default", "SWAN OFF WCAP", "SWAN OFF WCAP, friction 0.019", "SWAN OFF WCAP, no triads"]
    table = pd.DataFrame(ratio_rows((RAW / "results_2026-11-02_round2.txt").read_text(), names))
    frames.append(table.assign(experiment="swan_friction_triads", reference="SnapWave production"))

    names = ["SnapWave production", "SnapWave enclosure", "SnapWave enclosure, niter 200",
             "SnapWave enclosure + upwindref 1, niter 200", "SnapWave enclosure, sector 360, niter 200"]
    table = pd.DataFrame(ratio_rows((RAW / "snapwave_results_2026-11-03.txt").read_text(), names))
    frames.append(table.assign(experiment="snapwave_setup", reference="SWAN OFF WCAP (friction 0.038)"))
    return pd.concat(frames, ignore_index=True)


def snapwave_unstable() -> pd.DataFrame:
    """The 24 sampled non-converging hours under each SnapWave variant."""

    lines = (RAW / "snapwave_results_2026-11-03.txt").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("variant"))
    columns = ["variant", "unstable_of_24", "median_error", "median_pct_converged", "median_hs_max",
               "max_hs_max", "median_sites_above_3m", "median_seconds"]
    rows = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) != len(columns):
            break
        rows.append(dict(zip(columns, [parts[0], *map(float, parts[1:])])))
    return pd.DataFrame(rows)


def snapwave_convergence() -> pd.DataFrame:
    """Final SnapWave error of the 8760 production dynamic hours of 2013 (wind, old setup), binned."""

    hours = pd.read_csv(ROOT / "inputs" / "snapwave_production_2013_hours.csv", parse_dates=["time"])
    bins = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, np.inf]
    hours["error_bin"] = pd.cut(hours["err"], bins, include_lowest=True).astype(str)
    hours["hs_max_over_offshore"] = hours["hsmax"] / hours["hsoff"]
    return (
        hours.groupby("error_bin", sort=False)
        .agg(hours=("err", "size"), median_pct_converged=("ok", "median"),
             median_hs_max_over_offshore=("hs_max_over_offshore", "median"),
             median_offshore_hs=("hsoff", "median"), median_tp=("tpmean", "median"),
             median_u10=("u10", "median"))
        .reindex(pd.cut(pd.Series([*bins[1:-1], 10.0]) - 1e-9, bins, include_lowest=True).astype(str))
        .rename_axis("error_bin")
        .reset_index()
    )


def swan_directions() -> pd.DataFrame:
    """5 vs 10 degree directions: Hs ratio per metamodel test case (``cases_2/dir72`` vs ``cases/no_wcap``)."""

    text = (RAW / "results_2026-11-03_wind110.txt").read_text()
    rows = re.findall(r"^(\d+) (\d+\.\d+)$", text, flags=re.M)
    return pd.DataFrame(rows, columns=["case", "ratio"]).astype({"case": int, "ratio": float})


def main() -> None:
    tables = {
        "buoy_scores": buoy_scores(),
        "metamodel_ratios": metamodel_ratios(),
        "snapwave_unstable_hours": snapwave_unstable(),
        "snapwave_convergence_2013": snapwave_convergence(),
        "swan_directions": swan_directions(),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")
    check = tables["buoy_scores"]
    assert not check.duplicated(["experiment", "variant", "group"]).any()
    assert np.isfinite(check[["bias", "rmse"]]).all().all()


if __name__ == "__main__":
    main()
