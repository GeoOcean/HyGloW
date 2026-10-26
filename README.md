<p align="center">
  <img src="assets/logo.svg" alt="HyGloW — Hybrid Global Wave downscaling" width="100%">
</p>

HyGloW turns an offshore wave hindcast into nearshore waves for any coast:
a few hundred runs of a nearshore wave model (SnapWave or SWAN) train a
metamodel that reconstructs decades of hourly waves, bulk or full spectra.
How it works: [`hywaves/README.md`](hywaves/README.md).

## Install

A Python ≥ 3.11 environment with `bluemath_tk` (from its `develop` branch),
`bluemesh2d`, `wavespectra`, `xarray`, `geopandas`, `rioxarray`:

```bash
pip install "git+https://github.com/GeoOcean/BlueMath_tk.git@develop"
```

and the wave model, [SnapWave](https://github.com/danoroelvink/snapwave/) or
SWAN, on your PATH (or on a SLURM cluster), from notebook 05 on.

## Make your case

```bash
python -m hywaves start --name MyCase
```

1. Fill in `MyCase/MyCase.json` (replace the `TODO`s).
2. Run the notebooks in order; `MyCase/README.md` says what each one needs.
3. `python -m hywaves validate MyCase/` tells you what is missing at any time.

To see a finished case first: [Netherlands](Netherlands/) (Belgian and Dutch
coasts, 10 goals, 38 buoys, wind and water level). The results of every case
are in the GUI: `cd GUI && python app.py` (http://localhost:8050).

## Share your results

`python -m hywaves publish MyCase/` writes `MyCase/portal/`, the small bundle
the GUI reads; commit it and open a pull request, or keep the case in your
own repository and add it to `GUI/catalog/cases.yaml`.

## Repository

| Path | |
| --- | --- |
| `hywaves/` | The method, the case template and the command line |
| `Netherlands/`, `NewSouthWales/`, `Samoa/` | Cases |
| `GUI/` | Results explorer |
| `deck/` | Slides, published on GitHub Pages |

Generic code lives in [bluemath_tk](https://github.com/GeoOcean/BlueMath_tk);
examples in [BlueMath](https://github.com/GeoOcean/BlueMath).

Built by [GeoOcean](https://github.com/GeoOcean) (Universidad de Cantabria).
Contact: Javier Tausía ([@tausiaj](https://github.com/tausiaj),
[tausiaj@unican.es](mailto:tausiaj@unican.es)). Licensed under
[GPLv3](LICENSE).
