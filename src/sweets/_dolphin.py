"""Adapter that runs ``dolphin.workflows.displacement`` over a stack of CSLCs.

The job here is small: take the geocoded SLCs that COMPASS produced, build a
:class:`dolphin.workflows.config.DisplacementWorkflow` config from a handful
of sweets-friendly knobs, and call :func:`dolphin.workflows.displacement.run`.

dolphin owns everything from this point on:

- phase linking (sequential estimator over ministacks)
- interferogram network selection
- stitching across bursts
- unwrapping (SNAPHU / SPURT / WHIRLWIND)
- displacement timeseries inversion
- velocity estimation

This module is intentionally a thin shim — do not re-implement any of the
above here.
"""

from __future__ import annotations

from copy import copy
from math import cos, floor, log2, radians
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from dolphin.workflows.config import DisplacementWorkflow
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)
from pydantic_core import PydanticUndefined

from loguru import logger

from ._log import log_runtime

if TYPE_CHECKING:
    from dolphin.workflows.displacement import OutputPaths


class _DolphinConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _DolphinOptionsBase(_DolphinConfigSection):
    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_options(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for key, option in value.items():
            if isinstance(option, BaseModel):
                value[key] = option.model_dump(mode="python")

        legacy_fields = {
            "half_window": ("phase_linking", "half_window"),
            "strides": ("output_options", "strides"),
            "ministack_size": ("phase_linking", "ministack_size"),
            "use_evd": ("phase_linking", "use_evd"),
            "max_bandwidth": ("interferogram_network", "max_bandwidth"),
            "unwrap_method": ("unwrap_options", "unwrap_method"),
            "n_parallel_unwrap": ("unwrap_options", "n_parallel_jobs"),
            "snaphu_ntiles": ("unwrap_options", "snaphu_options", "ntiles"),
            "snaphu_parallel_tiles": (
                "unwrap_options",
                "snaphu_options",
                "n_parallel_tiles",
            ),
            "snaphu_tile_overlap": (
                "unwrap_options",
                "snaphu_options",
                "tile_overlap",
            ),
            "snaphu_cost": ("unwrap_options", "snaphu_options", "cost"),
            "gpu_enabled": ("worker_settings", "gpu_enabled"),
            "n_parallel_bursts": ("worker_settings", "n_parallel_bursts"),
            "block_shape": ("worker_settings", "block_shape"),
        }
        legacy_values = {
            key: value.pop(key) for key in tuple(legacy_fields) if key in value
        }
        run_timeseries = value.pop("run_timeseries", None)
        value.pop("nearest_n_coherence", None)

        legacy_workflow_options = value.pop("workflow_options", {})
        if not isinstance(legacy_workflow_options, dict):
            msg = "Legacy dolphin.workflow_options must be a mapping."
            raise ValueError(msg)
        protected_fields = {"cslc_file_list", "work_directory", "mask_file"}
        conflicting_fields = protected_fields.intersection(legacy_workflow_options)
        if conflicting_fields:
            fields = ", ".join(sorted(conflicting_fields))
            msg = f"workflow_options cannot override Sweets-managed fields: {fields}"
            raise ValueError(msg)

        for key, legacy_value in legacy_values.items():
            path = legacy_fields[key]
            if key == "snaphu_ntiles" and legacy_value == "auto":
                continue
            if key in {"half_window", "strides"}:
                legacy_value = {"y": legacy_value[0], "x": legacy_value[1]}
            elif key in {"snaphu_ntiles", "snaphu_tile_overlap", "block_shape"}:
                legacy_value = tuple(legacy_value)
            _set_nested_option(value, path, legacy_value)

        if run_timeseries is not None:
            _set_nested_option(
                value, ("timeseries_options", "run_inversion"), run_timeseries
            )
            _set_nested_option(
                value, ("timeseries_options", "run_velocity"), run_timeseries
            )

        _merge_nested_config(value, legacy_workflow_options)
        return value


_SWEETS_DEFAULTS: dict[tuple[str, ...], Any] = {
    ("worker_settings", "gpu_enabled"): True,
    ("worker_settings", "threads_per_worker"): 2,
    ("worker_settings", "block_shape"): (256, 256),
    ("phase_linking", "max_num_compressed"): 5,
    ("phase_linking", "output_reference_idx"): 0,
    ("interferogram_network", "max_bandwidth"): 3,
    ("unwrap_options", "n_parallel_jobs"): 2,
    ("unwrap_options", "snaphu_options", "ntiles"): None,
    ("unwrap_options", "snaphu_options", "tile_overlap"): (400, 400),
    ("unwrap_options", "snaphu_options", "single_tile_reoptimize"): True,
    ("timeseries_options", "apply_mask_to_timeseries"): True,
    ("output_options", "strides", "y"): 3,
    ("output_options", "strides", "x"): 6,
}


def _create_dolphin_config_model(
    model: type[BaseModel],
    name: str,
    *,
    excluded_fields: frozenset[str] = frozenset(),
    path: tuple[str, ...] = (),
    root_model: bool = False,
) -> type[BaseModel]:
    """Mirror a Dolphin model while applying Sweets' workflow defaults."""
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        if field_name in excluded_fields:
            continue

        annotation = field_info.annotation
        copied_field = copy(field_info)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_model = _create_dolphin_config_model(
                annotation,
                f"{name}_{field_name}",
                path=(*path, field_name),
            )
            copied_field.default = PydanticUndefined
            copied_field.default_factory = nested_model
            fields[field_name] = (nested_model, copied_field)
        else:
            default_path = (*path, field_name)
            if default_path in _SWEETS_DEFAULTS:
                copied_field.default = _SWEETS_DEFAULTS[default_path]
                copied_field.default_factory = None
            if default_path == ("unwrap_options", "snaphu_options", "ntiles"):
                annotation = Optional[annotation]
            fields[field_name] = (annotation, copied_field)

    base_model = _DolphinOptionsBase if root_model else _DolphinConfigSection
    return create_model(name, __base__=base_model, **fields)


_DolphinOptions = _create_dolphin_config_model(
    DisplacementWorkflow,
    "DolphinOptions",
    excluded_fields=frozenset({"cslc_file_list", "work_directory", "mask_file"}),
    root_model=True,
)


def _merge_nested_config(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        current = config.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_nested_config(current, value)
        else:
            config[key] = value


def _set_nested_option(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _estimate_snaphu_tiles_from_bounds(
    bounds: tuple[float, float, float, float], strides: tuple[int, int]
) -> tuple[int, int]:
    """Estimate a square SNAPHU tile grid from geographic bounds."""
    west, south, east, north = bounds
    center_lat = (south + north) / 2
    lat_size_m = abs(north - south) * 111_320.0
    lon_size_m = abs(east - west) * 111_320.0 * abs(cos(radians(center_lat)))
    # For ~1000x1000 outputs, use 2x2 tiles. Every doubling adds one tile.
    approx_rows = lat_size_m / (10.0 * strides[0])
    approx_cols = lon_size_m / (5.0 * strides[1])
    max_side = max(approx_rows, approx_cols)
    tile_count = 1 if max_side < 1000 else floor(log2(max_side / 1000.0)) + 2
    return (tile_count, tile_count)


DolphinOptions = _DolphinOptions


def build_displacement_config(
    cslc_files: list[Path],
    work_directory: Path,
    *,
    options: Optional[DolphinOptions] = None,
    mask_file: Optional[Path] = None,
    bounds: Optional[tuple[float, float, float, float]] = None,
    subdataset: str = "/data/VV",
    wavelength: Optional[float] = None,
):
    """Build a :class:`DisplacementWorkflow` config from sweets options.

    Parameters
    ----------
    cslc_files
        Geocoded SLC files (COMPASS HDF5 outputs or OPERA CSLC HDF5s or
        NISAR GSLC HDF5s).
    work_directory
        Where dolphin will write its scratch and output products.
    options
        Sweets-side knobs. Defaults if None.
    mask_file
        Optional water/validity mask. Convention: 1 = good, 0 = bad, dtype uint8.
    bounds
        Optional crop bounds (left, bottom, right, top) in EPSG:4326.
    subdataset
        HDF5 dataset path for the complex SLC inside each input file.
        Defaults to ``/data/VV`` (COMPASS / OPERA CSLC layout); callers
        with NISAR GSLCs pass e.g.
        ``/science/LSAR/GSLC/grids/frequencyA/HH``.

    Returns
    -------
    dolphin.workflows.config.DisplacementWorkflow
        Configured workflow ready to be passed to :func:`run_displacement`.

    """
    if options is None:
        options = DolphinOptions()

    work_directory = Path(work_directory).resolve()
    work_directory.mkdir(parents=True, exist_ok=True)

    config = options.model_dump(mode="python")
    input_options = config["input_options"]
    if input_options["subdataset"] is None:
        input_options["subdataset"] = subdataset
    if input_options["wavelength"] is None and wavelength is not None:
        input_options["wavelength"] = wavelength

    output_options = config["output_options"]
    if output_options["bounds"] is None and bounds is not None:
        output_options["bounds"] = list(bounds)
        output_options["bounds_epsg"] = 4326

    snaphu_options = config["unwrap_options"]["snaphu_options"]
    if snaphu_options["ntiles"] is None:
        strides = config["output_options"]["strides"]
        snaphu_options["ntiles"] = (
            (1, 1)
            if bounds is None
            else _estimate_snaphu_tiles_from_bounds(
                bounds, (strides["y"], strides["x"])
            )
        )

    config.update(
        {
            "cslc_file_list": [Path(path).resolve() for path in cslc_files],
            "work_directory": work_directory,
            "mask_file": mask_file,
        }
    )
    return DisplacementWorkflow.model_validate(config)


@log_runtime
def run_displacement(
    cslc_files: list[Path],
    work_directory: Path,
    *,
    options: Optional[DolphinOptions] = None,
    mask_file: Optional[Path] = None,
    bounds: Optional[tuple[float, float, float, float]] = None,
    config_yaml: Optional[Path] = None,
    subdataset: str = "/data/VV",
    wavelength: Optional[float] = None,
) -> "OutputPaths":
    """Build the dolphin config and run the displacement workflow.

    Parameters
    ----------
    cslc_files
        Geocoded SLCs from COMPASS, OPERA, or NISAR.
    work_directory
        dolphin work / output directory.
    options
        Sweets-side knobs (defaults to :class:`DolphinOptions`).
    mask_file
        Optional water/validity mask.
    bounds
        Optional crop bounds in EPSG:4326.
    config_yaml
        If given, the resolved dolphin config is dumped to this YAML path
        before running, for reproducibility.
    subdataset
        HDF5 dataset path for the complex SLC inside each input file;
        forwarded to :func:`build_displacement_config`.

    Returns
    -------
    dolphin.workflows.displacement.OutputPaths
        Dataclass of output paths produced by dolphin.

    """
    from dolphin.workflows.displacement import run

    cfg = build_displacement_config(
        cslc_files=cslc_files,
        work_directory=work_directory,
        options=options,
        mask_file=mask_file,
        bounds=bounds,
        subdataset=subdataset,
        wavelength=wavelength,
    )
    if config_yaml is not None:
        cfg.to_yaml(config_yaml)
        logger.info(f"Wrote dolphin config: {config_yaml}")

    logger.info(
        f"Running dolphin displacement on {len(cfg.cslc_file_list)} CSLCs"
        f" in {cfg.work_directory}"
    )
    return run(cfg)
