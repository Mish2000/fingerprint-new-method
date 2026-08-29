from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from fingerprint_new_method.experiment003 import (
    deduplicate_points,
    deterministic_rank,
    maximum_weight_assignment,
    parse_annotated_filename,
    parse_final_filename,
    scaled_tolerance,
    select_stratified,
    transform_points,
    write_csv,
    write_json,
)


def test_parse_final_filename() -> None:
    parsed = parse_final_filename("148_2_5.png")
    assert (parsed.identity, parsed.capture_group, parsed.instance) == (148, 2, 5)
    for invalid in ("0_1_1.png", "1_3_1.png", "1_1_6.png", "1_1_1.jpg", "x_1_1.png"):
        with pytest.raises(ValueError):
            parse_final_filename(invalid)


def test_parse_annotated_filename() -> None:
    parsed = parse_annotated_filename("73_right_loop.jpg")
    assert (parsed.local_index, parsed.pattern) == (73, "right_loop")
    for invalid in ("0_whorl.jpg", "1_unknown.jpg", "1_whorl.png", "whorl_1.jpg"):
        with pytest.raises(ValueError):
            parse_annotated_filename(invalid)


def test_deterministic_stratified_sampling_is_order_independent() -> None:
    records = [
        {"canonical_annotated_id": f"R1/{index}_whorl", "run": "R1", "pattern": "whorl"}
        for index in range(1, 6)
    ]
    forward = select_stratified(records, seed="seed", quotas={("R1", "whorl"): 3})
    reverse = select_stratified(list(reversed(records)), seed="seed", quotas={("R1", "whorl"): 3})
    assert [row["canonical_annotated_id"] for row in forward] == [row["canonical_annotated_id"] for row in reverse]
    assert all(len(row["selection_rank_sha256"]) == 64 for row in forward)
    assert deterministic_rank("seed", "R1", "R1/1_whorl") == deterministic_rank("seed", "R1", "R1/1_whorl")


def test_deduplicate_points_preserves_first_occurrence() -> None:
    assert deduplicate_points([(2, 3), (1, 1), (2, 3), [4, 5]]) == [(2, 3), (1, 1), (4, 5)]


def test_affine_and_projective_coordinate_transforms() -> None:
    points = [(0, 0), (2, 4)]
    affine = [[2, 0, 1], [0, 0.5, -1]]
    assert np.allclose(transform_points(points, affine), [[1, -1], [5, 1]])
    projective = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert np.allclose(transform_points(points, projective), points)


def test_scaled_tolerance_uses_area_equivalent_scale_and_half_up_rounding() -> None:
    assert scaled_tolerance([[0.625, 0, 0], [0, 0.4, 0]], 4) == 2
    assert scaled_tolerance([[0.625, 0, 0], [0, 0.625, 0]], 4) == 3
    assert scaled_tolerance([[0.1, 0, 0], [0, 0.1, 0]], 3) == 1


def test_maximum_weight_assignment_is_optimal_and_deterministic() -> None:
    weights = [[10, 2, 1], [9, 8, 1], [1, 2, 7]]
    assert maximum_weight_assignment(weights) == [0, 1, 2]
    assert maximum_weight_assignment([[1, 1], [1, 1]]) == [0, 1]
    with pytest.raises(ValueError):
        maximum_weight_assignment([[1], [2]])


def test_serialization_is_stable(tmp_path) -> None:
    json_path = tmp_path / "value.json"
    csv_path = tmp_path / "value.csv"
    write_json(json_path, {"b": 2, "a": "שלום"})
    write_csv(csv_path, [{"b": 2, "a": 1}], fieldnames=["a", "b"])
    assert json_path.read_text(encoding="utf-8") == '{\n  "a": "שלום",\n  "b": 2\n}\n'
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": "שלום", "b": 2}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"a": "1", "b": "2"}]
