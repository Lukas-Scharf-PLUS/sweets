# sweets
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isce-framework/sweets/main.svg)](https://results.pre-commit.ci/latest/github/isce-framework/sweets/main)

End-to-end InSAR workflow that turns a single AOI + date range into unwrapped interferograms, a displacement timeseries and a velocity raster.

## What sweets gives you

- Cloud-optimized data downloads
  - Sentinel-1 burst-subsetting via [`burst2safe`](https://github.com/ASFHyP3/burst2safe)
  - OPERA CSLC-S1 and NISAR GSLC AOI-based subsetting
- Geocoding
- DEM + water-mask prep
- Geometry stitching
- Phase-linking, unwrapping, timeseries network inversion, and velocity estimation via [dolphin](https://github.com/isce-framework/dolphin)

Interchangeable input sources, accessible through the same `Workflow` object:

| `--source`         | What it is                                                                                                                                                                                                                                                                                                                                                                                                                      | Tools used                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `safe` *(default)* | Sentinel-1 Level-1 bursts that intersect your AOI via `burst2safe`. Geocoded SLCs (GSLCs) created by COMPASS.                                                                                                                                                                                                                                                                                                                   | burst2safe, s1-reader, COMPASS |
| `opera-cslc`       | Pre-made [OPERA L2 CSLC-S1 HDF5s](https://www.jpl.nasa.gov/go/opera/products/cslc-product) from ASF DAAC, plus their matching CSLC-STATIC layers for geometry. Faster than `safe` when OPERA has produced for your AOI (CONUS, see [OPERA product suite](https://www.jpl.nasa.gov/go/opera/products/cslc-product-suite/) for coverage) if you are fine with their processing parameters (e.g., 5 meter x 10 meter UTM posting). | opera-utils                    |
| `nisar-gslc`       | [NISAR L2 GSLC products](https://nisar.jpl.nasa.gov/) geocoded in UTM.                                                                                                                                                                                                                                                                                                                                                          | opera-utils                    |

An optional `--do-tropo` flag adds a post-dolphin OPERA L4 TROPO-ZENITH
correction step for `safe` and `opera-cslc` (not yet wired for NISAR).

## Install

We recommend [pixi](https://pixi.sh/) for managing local environments.

```bash
git clone https://github.com/isce-framework/sweets.git && cd sweets
pixi install
pixi shell
```

That drops you into an env where `sweets`, `dolphin` and `compass` are all on
the `PATH` and sweets is installed in editable mode.

Conda/mamba is also supported using the included environment file:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPERA_UTILS=0.25.9 conda env create -f environment.yml
conda activate sweets_fork
sweets --help
```

The Conda environment targets Python 3.12 and CPU builds of `isce3` and
`jaxlib` for consistent solves across machines. The version override is
needed because the pinned Sweets-compatible `opera-utils` fork reports a
development version below the minimum declared by Dolphin 0.42.8. Use Pixi's
`gpu` environment if you need the CUDA-enabled `isce3` build.

**GPU mode (Linux + CUDA 12+ only).** There's a `gpu` pixi environment that
swaps the CPU `isce3` build for `isce3-cuda`, which accelerates COMPASS
geocoding, cross-multiplication and resampling on the GPU. dolphin itself
runs phase-linking on the GPU through JAX+CUDA regardless of environment.

```bash
pixi shell -e gpu        # activate once
sweets run config.yaml   # now uses isce3-cuda
```

macOS can't install this environment (conda-forge doesn't ship osx-arm64
builds of `isce3-cuda`); `pixi shell` without `-e gpu` is the right default
for every non-CUDA machine.

A [conda-forge feedstock](https://github.com/conda-forge/sweets-feedstock)
exists for `sweets`. It currently tracks the pre-v0.2 API; we'll bump it to
the new release once the fork dependencies (`s1-reader`, `COMPASS`,
`opera-utils`, `dolphin`, `spurt`) are merged upstream and a v0.2 release is
cut.

## Usage

### Command line

```bash
sweets config --help
sweets run --help
```

To configure a workflow the minimum inputs are an AOI (`--bbox` or `--wkt`), a
date range (`--start` / `--end`), and — for the Sentinel-1 path — the relative
orbit (`--track`). `--out-dir` is where raw downloads land; `--work-dir` holds
the workflow outputs.

Raw Sentinel-1 bursts (default):

```bash
sweets config \
  --bbox=-102.96 31.22 -101.91 31.56 \
  --start 2021-06-05 --end 2021-06-22 \
  --track 78 \
  --swaths IW2 \
  --out-dir ./data \
  --work-dir ./pecos_demo \
  --output pecos_demo/sweets_config.yaml

sweets run pecos_demo/sweets_config.yaml
```

Example config for Vienna:
```bash
sweets config \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --track 73 \
  --bbox \
    16.18099875016676 48.11744311620298 \
    16.587587299348154 48.324307364563005 \
  --work-dir ./ \
  --out-dir ./downloads \
  --output ./sweets_config.yaml \
  --n-workers 2 \
  --threads-per-worker 4 \
  --dolphin.worker-settings.block-shape 128 128 \
  --dolphin.output-options.strides.y 3 \
  --dolphin.output-options.strides.x 6 \
  --dolphin.unwrap-options.unwrap-method WHIRLWIND
```

Pre-made OPERA CSLCs (faster for CONUS, no COMPASS needed):

```bash
sweets config \
  --bbox=-102.96 31.22 -101.91 31.56 \
  --start 2021-06-05 --end 2021-06-22 \
  --source opera-cslc \
  --out-dir ./data \
  --work-dir ./pecos_opera \
  --output pecos_opera/sweets_config.yaml

sweets run pecos_opera/sweets_config.yaml
```

With the optional tropospheric correction step:

```bash
sweets config --source opera-cslc --do-tropo ... --output pecos_opera/sweets_config.yaml
sweets run pecos_opera/sweets_config.yaml
```

NISAR GSLCs:

```bash
sweets config \
  --bbox=-121.10 36.55 -120.95 36.70 \
  --start 2025-10-01 --end 2025-12-15 \
  --source nisar-gslc \
  --track 42 --frame 70 \
  --out-dir ./data \
  --work-dir ./salinas_nisar \
  --output salinas_nisar/nisar.yaml

sweets run salinas_nisar/nisar.yaml
```

```bash
sweets run sweets_config.yaml 2>&1 | tee sweets_run.log
```

### Starting at a later step

`sweets run` accepts `--starting-step N` so you can skip earlier stages if the
outputs are already on disk:




```bash
sweets run config.yaml --starting-step 2  # skip download, run geocode + dolphin
sweets run config.yaml --starting-step 3  # just (re-)run dolphin
```

### Dolphin configuration options

Sweets uses Dolphin's native workflow section names directly under `dolphin`.
The same options from `pixi run dolphin config --help` are available, including
options for SNAPHU, SPURT, Whirlwind, and other unwrap backends. Sweets
continues to supply the input CSLCs, mask, and work directory itself.

For example, choose Whirlwind and adjust its Goldstein filter in a
`sweets_config.yaml`:

```yaml
dolphin:
  unwrap_options:
    unwrap_method: whirlwind
    whirlwind_options:
      goldstein_alpha: 0.4
      connect_gaps: true
```

The same settings can be passed when generating a config with the CLI:

```bash
sweets config \
  ... \
  --dolphin.unwrap-options.unwrap-method WHIRLWIND \
  --dolphin.unwrap-options.whirlwind-options.goldstein-alpha 0.4 \
  --dolphin.unwrap-options.whirlwind-options.connect-gaps
```

Use `sweets config --help` for the Sweets-prefixed flags, or
`pixi run dolphin config --help` for Dolphin's native option names and
descriptions.

### Configuration from Python

You can also build a `Workflow` directly:

```python
from sweets.core import Workflow
from sweets.download import BurstSearch

w = Workflow(
    bbox=(-102.96, 31.22, -101.91, 31.56),
    search=BurstSearch(
        track=78,
        start="2021-06-05",
        end="2021-06-22",
        swaths=["IW2"],
        out_dir="data",
    ),
    work_dir="pecos_demo",
)
w.to_yaml("pecos_demo/sweets_config.yaml")
w.run()
```

The same pattern works for `OperaCslcSearch` and `NisarGslcSearch` — pick
whichever variant matches your data source. `Workflow.search` is a
discriminated union keyed on a `kind` field, so a YAML file with
`search: {kind: opera-cslc, ...}` round-trips correctly.

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0
licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
