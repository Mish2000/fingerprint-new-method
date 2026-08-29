from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from fingerprint_new_method.experiment004 import (
    PATTERN_QUOTAS,
    RESUME_STATE_KEYS,
    affine_matrix,
    aggregate_point_metrics,
    assign_grouped_split,
    blend_tiles,
    blinded_gate_b_decision,
    canonical_json_bytes,
    canonical_json_sha256,
    deduplicate_points,
    edge_mask,
    extract_peaks,
    gate_a_decision,
    gaussian_target,
    optimal_point_matching,
    paired_bootstrap_median,
    point_metrics,
    repeatability,
    threshold_cardinality_curve,
    tile_origins,
    transform_points,
    validate_resume_state,
)
from fingerprint_new_method.experiment004_transfer import (
    Registration,
    estimate_ridge_period,
    map_plain_to_roll,
    negative_anatomical_position,
    normalize_ridge_scale,
    registration_from_npz,
    registration_to_npz,
    score_registered_detections,
)


def test_grouped_split_has_frozen_pattern_quotas_and_no_group_leakage() -> None:
    groups = [
        (pattern, index)
        for pattern, quotas in PATTERN_QUOTAS.items()
        for index in range(1, sum(quotas.values()) + 1)
    ]
    first = assign_grouped_split(groups)
    second = assign_grouped_split(reversed(groups))
    assert first == second
    assert len(first) == 148
    for pattern, quotas in PATTERN_QUOTAS.items():
        observed = {
            partition: sum(
                assignment["partition"] == partition
                for (candidate_pattern, _), assignment in first.items()
                if candidate_pattern == pattern
            )
            for partition in ("train", "validation", "test")
        }
        assert observed == quotas
    image_roles = {
        (pattern, index, run): first[(pattern, index)]["partition"]
        for pattern, index in groups
        for run in range(1, 6)
    }
    for pattern, index in groups:
        assert len({image_roles[(pattern, index, run)] for run in range(1, 6)}) == 1


def test_annotation_deduplication_preserves_first_occurrence() -> None:
    unique, duplicates = deduplicate_points([(4, 5), (7, 8), (4, 5), (9, 1), (7, 8)])
    assert unique == [(4, 5), (7, 8), (9, 1)]
    assert duplicates == 2


def test_affine_coordinate_transform_matches_frozen_matrix() -> None:
    parameters = {
        "angle_degrees": 0.0,
        "translate_x": 3.0,
        "translate_y": -2.0,
        "scale": 1.0,
        "contrast": 1.0,
        "brightness": 0.0,
    }
    matrix = affine_matrix(32, 32, parameters)
    transformed, valid = transform_points([[4, 5], [31, 31]], matrix, width=32, height=32)
    np.testing.assert_allclose(transformed, [[7, 3]])
    assert valid.tolist() == [True, False]


def test_gaussian_target_is_centered_clipped_and_uses_maximum() -> None:
    target = gaussian_target((15, 15), [[7, 8], [7, 8], [0, 0]], sigma=1.5)
    assert target.shape == (15, 15)
    assert target[8, 7] == pytest.approx(1.0)
    assert target[0, 0] == pytest.approx(1.0)
    assert target.max() == pytest.approx(1.0)
    assert target[8, 8] == pytest.approx(np.exp(-1 / (2 * 1.5**2)), rel=1e-6)


def test_local_maxima_plateau_and_nms_are_deterministic() -> None:
    heatmap = np.zeros((12, 12), dtype=np.float32)
    heatmap[2:4, 4:6] = 0.9
    heatmap[8, 8] = 0.8
    heatmap[8, 10] = 0.7
    peaks = extract_peaks(heatmap, threshold=0.5, nms_radius=2)
    np.testing.assert_array_equal(peaks.coordinates, [[4, 2], [8, 8]])
    np.testing.assert_allclose(peaks.scores, [0.9, 0.8])


def _brute_force_matching_count_and_cost(predictions: np.ndarray, truth: np.ndarray, tolerance: float) -> tuple[int, float]:
    best_count = -1
    best_cost = float("inf")
    for size in range(min(len(predictions), len(truth)) + 1):
        for prediction_indices in itertools.combinations(range(len(predictions)), size):
            for truth_indices in itertools.permutations(range(len(truth)), size):
                distances = [
                    float(np.linalg.norm(predictions[prediction_index] - truth[truth_index]))
                    for prediction_index, truth_index in zip(prediction_indices, truth_indices, strict=True)
                ]
                if all(distance <= tolerance for distance in distances):
                    cost = sum(distances)
                    if size > best_count or (size == best_count and cost < best_cost):
                        best_count, best_cost = size, cost
    return best_count, best_cost


def test_point_matching_maximizes_cardinality_before_distance() -> None:
    predictions = np.asarray([[1.0, 0.0], [0.0, 0.0], [8.0, 8.0]])
    truth = np.asarray([[0.0, 0.0], [2.0, 0.0], [9.0, 8.0]])
    matches = optimal_point_matching(predictions, truth, tolerance=1.5)
    expected_count, expected_cost = _brute_force_matching_count_and_cost(predictions, truth, 1.5)
    assert len(matches) == expected_count == 3
    assert sum(item[2] for item in matches) == pytest.approx(expected_cost)
    assert {(item[0], item[1]) for item in matches} == {(0, 1), (1, 0), (2, 2)}


def test_precision_recall_f1_and_tolerance_sensitivity() -> None:
    predictions = np.asarray([[0, 0], [4, 0], [20, 20]], dtype=np.float32)
    truth = np.asarray([[1, 0], [7, 0]], dtype=np.float32)
    metrics_2 = point_metrics(predictions, truth, 2)
    metrics_4 = point_metrics(predictions, truth, 4)
    metrics_6 = point_metrics(predictions, truth, 6)
    assert metrics_2["true_positives"] == 1
    assert metrics_4["true_positives"] == 2
    assert metrics_6["true_positives"] == 2
    assert metrics_4["precision"] == pytest.approx(2 / 3)
    assert metrics_4["recall"] == pytest.approx(1.0)
    assert metrics_4["f1"] == pytest.approx(0.8)
    aggregate = aggregate_point_metrics([metrics_4, metrics_4])
    assert aggregate["f1"] == pytest.approx(0.8)
    assert aggregate["false_positives_per_image"] == pytest.approx(1.0)


def test_incremental_threshold_curve_matches_exact_cardinality() -> None:
    coordinates = np.asarray([[1, 0], [0, 0], [8, 8], [30, 30]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    truth = np.asarray([[0, 0], [2, 0], [9, 8]], dtype=np.float32)
    thresholds = [0.65, 0.75, 0.85, 0.95]
    curve = threshold_cardinality_curve(coordinates, scores, truth, thresholds, tolerance=1.5)
    for threshold in thresholds:
        selected = coordinates[scores >= threshold]
        exact = optimal_point_matching(selected, truth, 1.5)
        assert curve[threshold]["true_positives"] == len(exact)
        assert curve[threshold]["predictions"] == len(selected)


def test_edge_annotations_remain_identifiable_without_removal() -> None:
    mask = edge_mask([[8, 100], [9, 100], [511, 200], [200, 200]])
    assert mask.tolist() == [True, False, True, False]


def test_tiled_blending_reconstructs_global_coordinates() -> None:
    height, width = 700, 730
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    source = (x_grid + 3 * y_grid).astype(np.float32)
    tiles = []
    for y_value in tile_origins(height):
        for x_value in tile_origins(width):
            tiles.append((y_value, x_value, source[y_value : y_value + 512, x_value : x_value + 512]))
    reconstructed = blend_tiles(source.shape, tiles)
    np.testing.assert_allclose(reconstructed, source, rtol=1e-6, atol=1e-4)


def test_ridge_period_estimator_on_dataset_independent_sinusoid() -> None:
    y_grid, x_grid = np.mgrid[0:512, 0:512]
    image = np.clip(127 + 90 * np.cos(2 * np.pi * x_grid / 12.0), 0, 255).astype(np.uint8)
    estimate = estimate_ridge_period(image)
    assert estimate.status == "OK"
    assert estimate.period_px == pytest.approx(12.0, abs=1.0)
    assert estimate.accepted_tiles >= 5


def test_ridge_scale_transform_uses_only_estimated_period() -> None:
    _, x_grid = np.mgrid[0:512, 0:512]
    image = np.clip(127 + 90 * np.cos(2 * np.pi * x_grid / 24.0), 0, 255).astype(np.uint8)
    scaled = normalize_ridge_scale(image, target_period_px=12.0)
    assert scaled.status == "OK"
    assert scaled.source_period_px == pytest.approx(24.0, abs=1.0)
    assert scaled.scale_factor == pytest.approx(0.5, abs=0.03)
    assert scaled.image is not None
    assert scaled.image.shape[0] == pytest.approx(256, abs=16)


@pytest.mark.parametrize("position, expected", [(1, 2), (9, 10), (10, 1)])
def test_sd300_negative_pair_generation(position: int, expected: int) -> None:
    assert negative_anatomical_position(position) == expected


def test_overlap_filtering_mapping_and_repeatability() -> None:
    overlap = np.zeros((20, 20), dtype=bool)
    overlap[2:18, 2:18] = True
    registration = Registration(
        status="VALID",
        summary={},
        homography=np.eye(3),
        forward_flow=np.zeros((20, 20, 2), dtype=np.float32),
        overlap_plain=overlap,
        overlap_roll=overlap.copy(),
    )
    plain = np.asarray([[5, 5], [10, 10], [0, 0]], dtype=np.float32)
    roll = np.asarray([[6, 5], [10, 11], [19, 19]], dtype=np.float32)
    mapped = map_plain_to_roll(registration, plain[:2])
    np.testing.assert_allclose(mapped, plain[:2])
    score = score_registered_detections(registration, plain, roll, tolerance=2)
    assert score["matched"] == 2
    assert score["plain_in_overlap"] == 2
    assert score["roll_in_overlap"] == 2
    assert score["repeatability"] == pytest.approx(1.0)
    assert repeatability(0, 0, 0) == {"status": "ZERO_DETECTIONS_BOTH", "repeatability": None}


def test_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_median([0.1, 0.2, 0.3, -0.1], resamples=500, seed=123)
    second = paired_bootstrap_median([0.1, 0.2, 0.3, -0.1], resamples=500, seed=123)
    assert first == second
    assert first["estimate"] == pytest.approx(0.15)


def test_gate_decisions_respect_frozen_boundaries() -> None:
    strong_inputs = {
        "median_f1_4px": 0.80,
        "median_precision_4px": 0.75,
        "median_recall_4px": 0.75,
        "absolute_f1_advantage": 0.10,
        "run_median_f1_4px": {f"R{index}": 0.70 for index in range(1, 6)},
        "seed_f1_spread": 0.05,
        "all_seeds_above_baseline": True,
    }
    assert gate_a_decision(strong_inputs) == "STRONG_PASS"
    assert gate_a_decision({**strong_inputs, "median_f1_4px": 0.65}) == "CONDITIONAL"
    assert gate_a_decision({**strong_inputs, "median_f1_4px": 0.649999}) == "FAIL"
    assert gate_a_decision({**strong_inputs, "all_seeds_above_baseline": False, "median_f1_4px": 0.79}) == "FAIL"

    pending = blinded_gate_b_decision(
        mated_valid_registrations=15,
        paired_fingers=15,
        median_delta=0.10,
        bootstrap_lower=0.001,
        positive_seed_count=2,
    )
    assert pending["outcome_before_unblind"] == "PENDING_EXPERIMENT_001_UNBLIND"
    failed = blinded_gate_b_decision(
        mated_valid_registrations=15,
        paired_fingers=15,
        median_delta=0.02,
        bootstrap_lower=-0.01,
        positive_seed_count=1,
    )
    assert failed["outcome_before_unblind"] == "TRANSFER_FAIL"


def test_manifest_json_and_registration_round_trip(tmp_path: Path) -> None:
    value = {"z": [3, 2, 1], "a": "שלום", "nested": {"ok": True}}
    payload = canonical_json_bytes(value)
    assert json.loads(payload) == value
    assert canonical_json_sha256(value) == canonical_json_sha256(json.loads(payload))

    registration = Registration(
        status="VALID",
        summary={"inliers": 12},
        homography=np.eye(3),
        forward_flow=np.zeros((4, 5, 2), dtype=np.float32),
        overlap_plain=np.ones((4, 5), dtype=bool),
        overlap_roll=np.ones((6, 7), dtype=bool),
    )
    path = tmp_path / "registration.npz"
    registration_to_npz(path, registration)
    loaded = registration_from_npz(path)
    assert loaded.status == registration.status
    assert loaded.summary == registration.summary
    np.testing.assert_array_equal(loaded.homography, registration.homography)
    np.testing.assert_array_equal(loaded.forward_flow, registration.forward_flow)
    np.testing.assert_array_equal(loaded.overlap_plain, registration.overlap_plain)
    np.testing.assert_array_equal(loaded.overlap_roll, registration.overlap_roll)


def _resume_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "schema_version": 1,
        "seed": 40401,
        "epoch": 7,
        "model_configuration": {"maximum_epochs": 80, "batch_size": 2},
        **{key: 0 for key in RESUME_STATE_KEYS},
    }
    state.update(overrides)
    return state


def test_resume_state_is_rejected_unless_it_continues_the_same_frozen_run() -> None:
    configuration = {"maximum_epochs": 80, "batch_size": 2}
    accepted = _resume_state()
    assert validate_resume_state(accepted, seed=40401, model_configuration=configuration) is accepted

    for state, seed in (
        (_resume_state(), 40402),
        (_resume_state(seed=40403), 40401),
        (_resume_state(schema_version=2), 40401),
        (_resume_state(epoch=0), 40401),
        (_resume_state(epoch=81), 40401),
        (_resume_state(epoch=3.0), 40401),
        (_resume_state(model_configuration={"maximum_epochs": 80, "batch_size": 8}), 40401),
    ):
        with pytest.raises(RuntimeError):
            validate_resume_state(state, seed=seed, model_configuration=configuration)

    for key in RESUME_STATE_KEYS:
        truncated = _resume_state()
        del truncated[key]
        with pytest.raises(RuntimeError):
            validate_resume_state(truncated, seed=40401, model_configuration=configuration)
        assert validate_resume_state(
            truncated, seed=40401, model_configuration=configuration, require_full=False
        ) is truncated
