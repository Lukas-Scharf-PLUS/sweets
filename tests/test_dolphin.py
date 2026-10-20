import pytest

from sweets._dolphin import (
    DolphinOptions,
    _estimate_snaphu_tiles_from_bounds,
    build_displacement_config,
)


def test_estimate_snaphu_tiles_small_bounds():
    bounds = (-102.2, 32.15, -102.1, 32.22)
    assert _estimate_snaphu_tiles_from_bounds(bounds, (6, 12)) == (1, 1)


def test_estimate_snaphu_tiles_thresholds():
    assert _estimate_snaphu_tiles_from_bounds((0.0, 0.0, 0.01, 0.539), (6, 12)) == (
        2,
        2,
    )
    assert _estimate_snaphu_tiles_from_bounds((0.0, 0.0, 0.01, 1.078), (6, 12)) == (
        3,
        3,
    )


def test_estimate_snaphu_tiles_respects_strides():
    bounds = (0.0, 0.0, 0.01, 0.539)
    assert _estimate_snaphu_tiles_from_bounds(bounds, (12, 24)) == (1, 1)


def test_build_displacement_config_uses_native_options(tmp_path):
    options = DolphinOptions(
        phase_linking={"write_crlb": False, "max_num_compressed": 8},
        timeseries_options={"method": "L2"},
        unwrap_options={
            "unwrap_method": "whirlwind",
            "whirlwind_options": {"goldstein_alpha": 0.4},
        },
    )

    config = build_displacement_config(
        [tmp_path / "20200101_input.h5"], tmp_path / "work", options=options
    )

    assert config.phase_linking.write_crlb is False
    assert config.phase_linking.max_num_compressed == 8
    assert config.phase_linking.ministack_size == options.phase_linking.ministack_size
    assert config.timeseries_options.method == "L2"
    assert config.unwrap_options.unwrap_method.value == "whirlwind"
    assert config.unwrap_options.whirlwind_options.goldstein_alpha == 0.4
    assert config.output_options.strides.y == 3
    assert config.output_options.strides.x == 6


def test_build_displacement_config_rejects_sweets_managed_overrides(tmp_path):
    with pytest.raises(ValueError, match="work_directory"):
        DolphinOptions(workflow_options={"work_directory": "/other/path"})


def test_dolphin_options_migrates_legacy_sweets_field_names():
    options = DolphinOptions(
        half_window=(20, 10),
        strides=(4, 8),
        unwrap_method="spurt",
        run_timeseries=False,
        workflow_options={"phase_linking": {"beta": 0.01}},
    )

    assert options.phase_linking.half_window.y == 20
    assert options.phase_linking.half_window.x == 10
    assert options.output_options.strides.y == 4
    assert options.output_options.strides.x == 8
    assert options.unwrap_options.unwrap_method.value == "spurt"
    assert options.timeseries_options.run_inversion is False
    assert options.timeseries_options.run_velocity is False
    assert options.phase_linking.beta == 0.01
