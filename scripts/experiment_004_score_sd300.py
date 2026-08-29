"""Score frozen SD300 detections against pre-detector ridge registrations."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fingerprint_new_method.experiment004 import (
    ARTIFACT_ROOT,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    PROJECT_ROOT,
    TRAINING_SEEDS,
    blinded_gate_b_decision,
    extract_peaks,
    paired_bootstrap_median,
    ridge_area_mask,
    sha256_file,
    write_csv,
    write_json,
)
from fingerprint_new_method.experiment004_transfer import (
    SCALE_FACTOR_BANDS,
    registration_from_npz,
    score_registered_detections,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _preprocessing_row(manifest: dict[str, Any], ppi: int, source_id: str) -> dict[str, Any]:
    return manifest["records"][f"{ppi}|{source_id}"]


def _detections(
    heatmaps: dict[str, Any],
    *,
    ppi: int,
    seed: int,
    source_id: str,
    threshold: float,
    radius: int,
) -> np.ndarray:
    row = heatmaps["records"][f"{ppi}|{seed}|{source_id}"]
    heatmap = np.load(PROJECT_ROOT / row["relative_local_path"], allow_pickle=False)
    return extract_peaks(heatmap, threshold=threshold, nms_radius=radius).coordinates


def _density(detections: np.ndarray, preprocessing_row: dict[str, Any]) -> dict[str, Any]:
    image = np.load(PROJECT_ROOT / preprocessing_row["relative_local_path"], allow_pickle=False)
    ridge_pixels = int(np.count_nonzero(ridge_area_mask(image)))
    return {
        "detections": len(detections),
        "detections_per_megapixel": float(len(detections) / (image.size / 1_000_000.0)),
        "ridge_area_pixels": ridge_pixels,
        "detections_per_estimated_ridge_megapixel": (
            float(len(detections) / (ridge_pixels / 1_000_000.0)) if ridge_pixels else None
        ),
    }


def _existing_rows(path: Path, ppi: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if int(row["ppi"]) != ppi]


def _pair_band_status(
    preprocessing: dict[str, Any],
    ppi: int,
    first_source_id: str,
    second_source_id: str,
    analysis: str,
) -> str:
    """Both images of a pair must pass an analysis' ridge-scale band."""

    statuses = {
        _preprocessing_row(preprocessing, ppi, source_id).get("analysis_status", {}).get(analysis, "PREPROCESSING_FAILURE")
        for source_id in (first_source_id, second_source_id)
    }
    return "OK" if statuses == {"OK"} else "PREPROCESSING_FAILURE"


def _analysis_registration_status(row: dict[str, Any], analysis: str, pair_type: str) -> str:
    if analysis == "frozen" and row[f"frozen_{pair_type}_pair"] != "OK":
        return "INVALID"
    return str(row[f"{pair_type}_registration_status"])


def _analysis_delta(row: dict[str, Any], analysis: str) -> float | None:
    if analysis == "frozen" and (row["frozen_mated_pair"] != "OK" or row["frozen_non_mated_pair"] != "OK"):
        return None
    return row["delta"]


def score_plain_roll(ppi: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_manifest = _read_json(ARTIFACT_ROOT / "model_manifest.json")
    pairs = _read_json(ARTIFACT_ROOT / "sd300_pair_manifest.json")
    preprocessing = _read_json(ARTIFACT_ROOT / "sd300_preprocessing_manifest.json")
    heatmaps = _read_json(ARTIFACT_ROOT / "sd300_heatmap_manifest.json")
    frozen = model_manifest["frozen_inference"]
    records = [row for row in pairs["records"] if int(row["ppi"]) == ppi]
    result_rows: list[dict[str, Any]] = []
    detection_cache: dict[tuple[int, str], np.ndarray] = {}

    for seed in TRAINING_SEEDS:
        for record in records:
            source_ids = {
                "plain": record["plain_source_id"],
                "mated": record["mated_roll_source_id"],
                "non_mated": record["non_mated_roll_source_id"],
            }
            for source_id in source_ids.values():
                cache_key = (seed, source_id)
                if cache_key not in detection_cache:
                    preprocessing_row = _preprocessing_row(preprocessing, ppi, source_id)
                    detection_cache[cache_key] = (
                        _detections(
                            heatmaps,
                            ppi=ppi,
                            seed=seed,
                            source_id=source_id,
                            threshold=frozen["threshold"],
                            radius=frozen["nms_radius_px"],
                        )
                        if preprocessing_row["status"] == "OK"
                        else np.empty((0, 2), dtype=np.float32)
                    )
            plain = detection_cache[(seed, source_ids["plain"])]
            mated = detection_cache[(seed, source_ids["mated"])]
            non_mated = detection_cache[(seed, source_ids["non_mated"])]
            registration_root = PROJECT_ROOT / "artifacts" / "experiment-004" / "local-large" / "registrations"
            mated_registration = registration_from_npz(
                registration_root / str(ppi) / f"{record['record_id']}-mated.npz"
            )
            non_mated_registration = registration_from_npz(
                registration_root / str(ppi) / f"{record['record_id']}-non_mated.npz"
            )
            mated_score = score_registered_detections(mated_registration, plain, mated, tolerance=4.0)
            non_mated_score = score_registered_detections(
                non_mated_registration,
                plain,
                non_mated,
                tolerance=4.0,
            )
            delta = (
                float(mated_score["repeatability"] - non_mated_score["repeatability"])
                if mated_score["repeatability"] is not None and non_mated_score["repeatability"] is not None
                else None
            )
            frozen_pair = {
                pair_type: _pair_band_status(preprocessing, ppi, source_ids["plain"], source_ids[pair_type], "frozen")
                for pair_type in ("mated", "non_mated")
            }
            plain_density = _density(plain, _preprocessing_row(preprocessing, ppi, source_ids["plain"]))
            mated_density = _density(mated, _preprocessing_row(preprocessing, ppi, source_ids["mated"]))
            non_mated_density = _density(
                non_mated,
                _preprocessing_row(preprocessing, ppi, source_ids["non_mated"]),
            )
            result_rows.append(
                {
                    "record_id": record["record_id"],
                    "sample_index": record["sample_index"],
                    "subject_id": record["subject_id"],
                    "anatomical_position": record["anatomical_position"],
                    "ppi": ppi,
                    "seed": seed,
                    "frozen_mated_pair": frozen_pair["mated"],
                    "frozen_non_mated_pair": frozen_pair["non_mated"],
                    "mated_registration_status": mated_registration.status,
                    "non_mated_registration_status": non_mated_registration.status,
                    "mated_score_status": mated_score["status"],
                    "non_mated_score_status": non_mated_score["status"],
                    "mated_repeatability": mated_score["repeatability"],
                    "non_mated_repeatability": non_mated_score["repeatability"],
                    "delta": delta,
                    "mated_matches": mated_score["matched"],
                    "mated_plain_in_overlap": mated_score["plain_in_overlap"],
                    "mated_roll_in_overlap": mated_score["roll_in_overlap"],
                    "non_mated_matches": non_mated_score["matched"],
                    "non_mated_plain_in_overlap": non_mated_score["plain_in_overlap"],
                    "non_mated_roll_in_overlap": non_mated_score["roll_in_overlap"],
                    "plain_detections": plain_density["detections"],
                    "plain_detections_per_megapixel": plain_density["detections_per_megapixel"],
                    "plain_detections_per_estimated_ridge_megapixel": plain_density[
                        "detections_per_estimated_ridge_megapixel"
                    ],
                    "mated_roll_detections": mated_density["detections"],
                    "mated_roll_detections_per_megapixel": mated_density["detections_per_megapixel"],
                    "mated_roll_detections_per_estimated_ridge_megapixel": mated_density[
                        "detections_per_estimated_ridge_megapixel"
                    ],
                    "non_mated_roll_detections": non_mated_density["detections"],
                    "non_mated_roll_detections_per_megapixel": non_mated_density["detections_per_megapixel"],
                    "non_mated_roll_detections_per_estimated_ridge_megapixel": non_mated_density[
                        "detections_per_estimated_ridge_megapixel"
                    ],
                }
            )
        print(f"ppi={ppi} scored seed={seed}", flush=True)

    return result_rows, {analysis: _summarize(result_rows, ppi, analysis) for analysis in ("frozen", "conformant")}


def _summarize(result_rows: Sequence[dict[str, Any]], ppi: int, analysis: str) -> dict[str, Any]:
    """Aggregate one ridge-scale analysis from the single shared scoring pass."""

    per_seed: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        values = [
            float(delta)
            for row in result_rows
            if row["seed"] == seed and (delta := _analysis_delta(row, analysis)) is not None
        ]
        per_seed[str(seed)] = {
            "paired_valid": len(values),
            "median_delta": float(np.median(values)) if values else None,
            "positive_delta_count": sum(value > 0 for value in values),
            "bootstrap": paired_bootstrap_median(values, seed=BOOTSTRAP_SEED) if values else None,
        }
    finger_medians: list[dict[str, Any]] = []
    for sample_index in range(1, 21):
        values = [
            float(delta)
            for row in result_rows
            if int(row["sample_index"]) == sample_index and (delta := _analysis_delta(row, analysis)) is not None
        ]
        finger_medians.append(
            {
                "sample_index": sample_index,
                "seed_values": values,
                "available_seeds": len(values),
                "median_delta": float(np.median(values)) if len(values) >= 2 else None,
            }
        )
    primary_values = [row["median_delta"] for row in finger_medians if row["median_delta"] is not None]
    first_seed_rows = [row for row in result_rows if row["seed"] == TRAINING_SEEDS[0]]
    return {
        "ppi": ppi,
        "analysis": analysis,
        "scale_factor_band": list(SCALE_FACTOR_BANDS[analysis]),
        "decides_gate_b": analysis == "frozen",
        "mated_valid_registrations": sum(
            _analysis_registration_status(row, analysis, "mated") == "VALID" for row in first_seed_rows
        ),
        "non_mated_valid_registrations": sum(
            _analysis_registration_status(row, analysis, "non_mated") == "VALID" for row in first_seed_rows
        ),
        "pairs_excluded_by_scale_band": sum(
            row["frozen_mated_pair"] != "OK" or row["frozen_non_mated_pair"] != "OK" for row in first_seed_rows
        )
        if analysis == "frozen"
        else 0,
        "per_seed": per_seed,
        "finger_median_deltas": finger_medians,
        "primary_paired_fingers": len(primary_values),
        "primary_median_delta": float(np.median(primary_values)) if primary_values else None,
        "primary_bootstrap": paired_bootstrap_median(primary_values, seed=BOOTSTRAP_SEED) if primary_values else None,
    }


def _score_cross_resolution(
    pairs: dict[str, Any],
    preprocessing: dict[str, Any],
    heatmaps: dict[str, Any],
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_1000 = {int(row["sample_index"]): row for row in pairs["records"] if int(row["ppi"]) == 1000}
    records_2000 = {int(row["sample_index"]): row for row in pairs["records"] if int(row["ppi"]) == 2000}
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        for sample_index in range(1, 21):
            first = records_1000[sample_index]
            second = records_2000[sample_index]
            for impression, key in (("plain", "plain_source_id"), ("roll", "mated_roll_source_id")):
                source_1000 = first[key]
                source_2000 = second[key]
                detections_1000 = _detections(
                    heatmaps,
                    ppi=1000,
                    seed=seed,
                    source_id=source_1000,
                    threshold=frozen["threshold"],
                    radius=frozen["nms_radius_px"],
                )
                detections_2000 = _detections(
                    heatmaps,
                    ppi=2000,
                    seed=seed,
                    source_id=source_2000,
                    threshold=frozen["threshold"],
                    radius=frozen["nms_radius_px"],
                )
                path = (
                    PROJECT_ROOT
                    / "artifacts"
                    / "experiment-004"
                    / "local-large"
                    / "registrations"
                    / "cross-resolution"
                    / f"S{sample_index:02d}-{impression}.npz"
                )
                registration = registration_from_npz(path)
                score = score_registered_detections(
                    registration,
                    detections_1000,
                    detections_2000,
                    tolerance=4.0,
                )
                rows.append(
                    {
                        "sample_index": sample_index,
                        "seed": seed,
                        "impression": impression,
                        "registration_status": registration.status,
                        "frozen_pair": "OK"
                        if all(
                            _preprocessing_row(preprocessing, resolution, source_id)
                            .get("analysis_status", {})
                            .get("frozen")
                            == "OK"
                            for resolution, source_id in ((1000, source_1000), (2000, source_2000))
                        )
                        else "PREPROCESSING_FAILURE",
                        **score,
                    }
                )
    values = [row["repeatability"] for row in rows if row["repeatability"] is not None]
    frozen_values = [
        row["repeatability"]
        for row in rows
        if row["repeatability"] is not None and row["frozen_pair"] == "OK"
    ]
    return rows, {
        "valid_scores": len(values),
        "median_repeatability": float(np.median(values)) if values else None,
        "frozen_band_valid_scores": len(frozen_values),
        "frozen_band_median_repeatability": float(np.median(frozen_values)) if frozen_values else None,
        "by_impression": {
            impression: {
                "valid_scores": sum(row["impression"] == impression and row["repeatability"] is not None for row in rows),
                "median_repeatability": (
                    float(np.median([row["repeatability"] for row in rows if row["impression"] == impression and row["repeatability"] is not None]))
                    if any(row["impression"] == impression and row["repeatability"] is not None for row in rows)
                    else None
                ),
            }
            for impression in ("plain", "roll")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi", required=True, type=int, choices=(1000, 2000))
    arguments = parser.parse_args()
    ppi = arguments.ppi
    rows, summaries = score_plain_roll(ppi)
    summary = summaries["frozen"]
    csv_path = ARTIFACT_ROOT / "sd300_repeatability.csv"
    combined = _existing_rows(csv_path, ppi) + rows
    write_csv(csv_path, combined, list(combined[0]))
    transfer_path = ARTIFACT_ROOT / "sd300_transfer_summary.json"
    transfer = (
        _read_json(transfer_path)
        if transfer_path.is_file()
        else {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "BLINDED",
            "experiment_001_ratings_opened": False,
            "ppi": {},
        }
    )
    transfer["ppi"][str(ppi)] = summary
    variant = transfer.setdefault(
        "exploratory_scale_variant",
        {
            "status": "EXPLORATORY_SENSITIVITY",
            "frozen_at_utc": _read_json(ARTIFACT_ROOT / "scale_guard_contingency.json")["frozen_at_utc"],
            "amendment": "docs/experiments/004-scale-guard-contingency-amendment.md",
            "decides_gate_b": False,
            "ppi": {},
        },
    )
    variant["ppi"][str(ppi)] = summaries["conformant"]

    if ppi == 1000:
        median_delta = summary["primary_median_delta"]
        lower = summary["primary_bootstrap"]["lower"] if summary["primary_bootstrap"] else None
        positive_seeds = sum(
            row["median_delta"] is not None and row["median_delta"] > 0 for row in summary["per_seed"].values()
        )
        transfer["blinded_gate_b"] = blinded_gate_b_decision(
            mated_valid_registrations=summary["mated_valid_registrations"],
            paired_fingers=summary["primary_paired_fingers"],
            median_delta=median_delta,
            bootstrap_lower=lower,
            positive_seed_count=positive_seeds,
        )
        exploratory = summaries["conformant"]
        variant["blinded_gate_b_if_applied"] = blinded_gate_b_decision(
            mated_valid_registrations=exploratory["mated_valid_registrations"],
            paired_fingers=exploratory["primary_paired_fingers"],
            median_delta=exploratory["primary_median_delta"],
            bootstrap_lower=exploratory["primary_bootstrap"]["lower"] if exploratory["primary_bootstrap"] else None,
            positive_seed_count=sum(
                row["median_delta"] is not None and row["median_delta"] > 0
                for row in exploratory["per_seed"].values()
            ),
        )
        write_json(transfer_path, transfer)
        marker = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "ppi": 1000,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "repeatability_csv_sha256": sha256_file(csv_path),
            "transfer_summary_sha256": sha256_file(transfer_path),
            "ratings_opened": False,
        }
        write_json(ARTIFACT_ROOT / "sd300_1000_outputs_frozen.json", marker)
    else:
        pairs = _read_json(ARTIFACT_ROOT / "sd300_pair_manifest.json")
        preprocessing = _read_json(ARTIFACT_ROOT / "sd300_preprocessing_manifest.json")
        heatmaps = _read_json(ARTIFACT_ROOT / "sd300_heatmap_manifest.json")
        frozen = _read_json(ARTIFACT_ROOT / "model_manifest.json")["frozen_inference"]
        cross_rows, cross_summary = _score_cross_resolution(pairs, preprocessing, heatmaps, frozen)
        cross_path = ARTIFACT_ROOT / "sd300_cross_resolution_repeatability.csv"
        write_csv(cross_path, cross_rows, list(cross_rows[0]))
        transfer["cross_resolution_1000_to_2000"] = cross_summary
        write_json(transfer_path, transfer)
        marker = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "repeatability_csv_sha256": sha256_file(csv_path),
            "cross_resolution_csv_sha256": sha256_file(cross_path),
            "transfer_summary_sha256": sha256_file(transfer_path),
            "ratings_opened": False,
        }
        write_json(ARTIFACT_ROOT / "sd300_outputs_frozen.json", marker)
    print(json.dumps({"status": "SD300_SCORED_AND_FROZEN", "ppi": ppi, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
