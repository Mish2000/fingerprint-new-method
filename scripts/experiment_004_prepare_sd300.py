"""Freeze SD300 pair mapping, ridge-scale preprocessing, and ridge-only registration."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fingerprint_new_method.experiment004 import (
    ARTIFACT_ROOT,
    EXPERIMENT_ID,
    LOCAL_ROOT,
    PROJECT_ROOT,
    preprocess_image,
    resolve_source_id,
    sha256_bytes,
    sha256_file,
    write_csv,
    write_json,
)
from fingerprint_new_method.experiment004_transfer import (
    CONFORMANT_SCALE_FACTOR_BAND,
    SCALE_FACTOR_BANDS,
    Registration,
    build_sd300_pair_manifest,
    normalize_ridge_scale,
    register_ridge_images,
    registration_to_npz,
    scale_factor_status,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_path(ppi: int, source_id: str) -> Path:
    digest = sha256_bytes(source_id.encode())[:16]
    return LOCAL_ROOT / "sd300-preprocessed" / str(ppi) / f"{digest}.npy"


def _invalid_registration(reason: str, plain_shape: tuple[int, int], roll_shape: tuple[int, int]) -> Registration:
    return Registration(
        status="INVALID",
        summary={"reason": reason, "mutual_matches": 0, "ransac_inliers": 0},
        homography=None,
        forward_flow=None,
        overlap_plain=np.zeros(plain_shape, dtype=bool),
        overlap_roll=np.zeros(roll_shape, dtype=bool),
    )


def _pair_analysis_status(first: dict[str, Any], second: dict[str, Any], analysis: str) -> str:
    """Both images must pass an analysis' scale band for the pair to enter it."""

    statuses = {entry.get("analysis_status", {}).get(analysis, "PREPROCESSING_FAILURE") for entry in (first, second)}
    return "OK" if statuses == {"OK"} else "PREPROCESSING_FAILURE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi", required=True, type=int, choices=(1000, 2000))
    arguments = parser.parse_args()
    ppi = arguments.ppi
    gate_a = _read_json(ARTIFACT_ROOT / "test_metrics.json")["gate_a"]
    if gate_a == "FAIL":
        raise RuntimeError("Gate A failed; SD300 must not be accessed")
    model_manifest = _read_json(ARTIFACT_ROOT / "model_manifest.json")
    if ppi == 2000 and not (ARTIFACT_ROOT / "sd300_1000_outputs_frozen.json").is_file():
        raise RuntimeError("2000-ppi sensitivity requires frozen 1000-ppi outputs")
    if (ARTIFACT_ROOT / "sd300_outputs_frozen.json").exists():
        raise RuntimeError("All SD300 outputs are already frozen")

    pair_manifest_path = ARTIFACT_ROOT / "sd300_pair_manifest.json"
    if pair_manifest_path.is_file():
        pair_manifest = _read_json(pair_manifest_path)
    else:
        selection = _read_json(ARTIFACT_ROOT.parent / "experiment-001" / "selection" / "selection_manifest.json")
        pair_manifest = build_sd300_pair_manifest(selection)
        pair_manifest["frozen_at_utc"] = datetime.now(UTC).isoformat()
        pair_manifest["configuration_id"] = model_manifest["configuration_id"]
        write_json(pair_manifest_path, pair_manifest)

    selected_records = [record for record in pair_manifest["records"] if int(record["ppi"]) == ppi]
    if len(selected_records) != 20:
        raise RuntimeError(f"Expected 20 pair records at {ppi} ppi, found {len(selected_records)}")
    source_ids = sorted(
        {
            record[key]
            for record in selected_records
            for key in ("plain_source_id", "mated_roll_source_id", "non_mated_roll_source_id")
        }
    )
    cache_manifest_path = ARTIFACT_ROOT / "sd300_preprocessing_manifest.json"
    cache_manifest = (
        _read_json(cache_manifest_path)
        if cache_manifest_path.is_file()
        else {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "configuration_id": model_manifest["configuration_id"],
            "target_ridge_period_px": model_manifest["frozen_inference"]["ridge_scale"]["target_period_px"],
            "records": {},
        }
    )
    target_period = float(cache_manifest["target_ridge_period_px"])
    for index, source_id in enumerate(source_ids, start=1):
        key = f"{ppi}|{source_id}"
        if key in cache_manifest["records"]:
            continue
        source_path = resolve_source_id(source_id)
        gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            cache_manifest["records"][key] = {
                "ppi": ppi,
                "source_id": source_id,
                "status": "SOURCE_DECODE_FAILURE",
                "analysis_status": {name: "PREPROCESSING_FAILURE" for name in SCALE_FACTOR_BANDS},
            }
            continue
        # The wide band is materialized once; the frozen primary analysis is a strict
        # subset of it because an accepted image gets the identical resize either way.
        scaled = normalize_ridge_scale(gray, target_period, band=CONFORMANT_SCALE_FACTOR_BAND)
        row: dict[str, Any] = {
            "ppi": ppi,
            "source_id": source_id,
            "source_sha256": sha256_file(source_path),
            "source_shape": list(gray.shape),
            "status": scaled.status,
            "analysis_status": {
                name: scale_factor_status(scaled.scale_factor, band) for name, band in SCALE_FACTOR_BANDS.items()
            },
            "source_period_px": scaled.source_period_px,
            "target_period_px": scaled.target_period_px,
            "scale_factor": scaled.scale_factor,
            "ridge_period_estimate": scaled.estimate.as_dict(),
        }
        if scaled.image is not None:
            processed = np.rint(preprocess_image(scaled.image) * 255.0).astype(np.uint8)
            output_path = _cache_path(ppi, source_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, processed, allow_pickle=False)
            row.update(
                {
                    "normalized_shape": list(processed.shape),
                    "relative_local_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                }
            )
        cache_manifest["records"][key] = row
        print(f"ppi={ppi} preprocessing={index}/{len(source_ids)} status={row['status']}", flush=True)
    cache_manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(cache_manifest_path, cache_manifest)

    registration_rows: list[dict[str, Any]] = []
    summary_path = ARTIFACT_ROOT / "sd300_registration_summary.csv"
    if summary_path.is_file():
        import csv

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            registration_rows.extend(dict(row) for row in csv.DictReader(handle) if str(row["ppi"]) != str(ppi))
    for record_index, record in enumerate(selected_records, start=1):
        plain_entry = cache_manifest["records"][f"{ppi}|{record['plain_source_id']}"]
        for pair_type, roll_key in (("mated", "mated_roll_source_id"), ("non_mated", "non_mated_roll_source_id")):
            roll_entry = cache_manifest["records"][f"{ppi}|{record[roll_key]}"]
            registration_path = LOCAL_ROOT / "registrations" / str(ppi) / f"{record['record_id']}-{pair_type}.npz"
            if plain_entry["status"] != "OK" or roll_entry["status"] != "OK":
                plain_shape = tuple(plain_entry.get("normalized_shape", (1, 1)))
                roll_shape = tuple(roll_entry.get("normalized_shape", (1, 1)))
                registration = _invalid_registration("PREPROCESSING_FAILURE", plain_shape, roll_shape)
            else:
                plain = np.load(PROJECT_ROOT / plain_entry["relative_local_path"], allow_pickle=False)
                roll = np.load(PROJECT_ROOT / roll_entry["relative_local_path"], allow_pickle=False)
                registration = register_ridge_images(
                    plain.astype(np.float32) / np.float32(255.0),
                    roll.astype(np.float32) / np.float32(255.0),
                )
            registration_to_npz(registration_path, registration)
            registration_rows.append(
                {
                    "record_id": record["record_id"],
                    "sample_index": record["sample_index"],
                    "ppi": ppi,
                    "pair_type": pair_type,
                    "frozen_analysis_status": _pair_analysis_status(plain_entry, roll_entry, "frozen"),
                    "conformant_analysis_status": _pair_analysis_status(plain_entry, roll_entry, "conformant"),
                    "subject_id": record["subject_id"],
                    "plain_position": record["anatomical_position"],
                    "roll_position": (
                        record["anatomical_position"]
                        if pair_type == "mated"
                        else record["non_mated_anatomical_position"]
                    ),
                    "status": registration.status,
                    "reason": registration.summary.get("reason"),
                    "mutual_matches": registration.summary.get("mutual_matches"),
                    "ransac_inliers": registration.summary.get("ransac_inliers"),
                    "inlier_ratio": registration.summary.get("inlier_ratio"),
                    "median_reprojection_error": registration.summary.get("median_reprojection_error"),
                    "p90_reprojection_error": registration.summary.get("p90_reprojection_error"),
                    "plain_hull_coverage": registration.summary.get("plain_hull_coverage"),
                    "roll_hull_coverage": registration.summary.get("roll_hull_coverage"),
                    "overlap_plain_pixels": registration.summary.get("overlap_plain_pixels"),
                    "overlap_roll_pixels": registration.summary.get("overlap_roll_pixels"),
                    "registration_relative_local_path": registration_path.relative_to(PROJECT_ROOT).as_posix(),
                    "registration_sha256": sha256_file(registration_path),
                }
            )
        print(f"ppi={ppi} registrations={record_index}/20", flush=True)
    if ppi == 2000:
        records_1000 = {
            int(item["sample_index"]): item for item in pair_manifest["records"] if int(item["ppi"]) == 1000
        }
        for record_index, record in enumerate(selected_records, start=1):
            first = records_1000[int(record["sample_index"])]
            for impression, source_key in (("plain", "plain_source_id"), ("roll", "mated_roll_source_id")):
                first_entry = cache_manifest["records"][f"1000|{first[source_key]}"]
                second_entry = cache_manifest["records"][f"2000|{record[source_key]}"]
                registration_path = (
                    LOCAL_ROOT
                    / "registrations"
                    / "cross-resolution"
                    / f"S{int(record['sample_index']):02d}-{impression}.npz"
                )
                if first_entry["status"] != "OK" or second_entry["status"] != "OK":
                    registration = _invalid_registration(
                        "PREPROCESSING_FAILURE",
                        tuple(first_entry.get("normalized_shape", (1, 1))),
                        tuple(second_entry.get("normalized_shape", (1, 1))),
                    )
                else:
                    first_image = np.load(PROJECT_ROOT / first_entry["relative_local_path"], allow_pickle=False)
                    second_image = np.load(PROJECT_ROOT / second_entry["relative_local_path"], allow_pickle=False)
                    registration = register_ridge_images(
                        first_image.astype(np.float32) / np.float32(255.0),
                        second_image.astype(np.float32) / np.float32(255.0),
                    )
                registration_to_npz(registration_path, registration)
                registration_rows.append(
                    {
                        "record_id": f"S{int(record['sample_index']):02d}-1000x2000",
                        "sample_index": record["sample_index"],
                        "ppi": "1000x2000",
                        "pair_type": f"cross_resolution_{impression}",
                        "frozen_analysis_status": _pair_analysis_status(first_entry, second_entry, "frozen"),
                        "conformant_analysis_status": _pair_analysis_status(first_entry, second_entry, "conformant"),
                        "subject_id": record["subject_id"],
                        "plain_position": record["anatomical_position"],
                        "roll_position": record["anatomical_position"],
                        "status": registration.status,
                        "reason": registration.summary.get("reason"),
                        "mutual_matches": registration.summary.get("mutual_matches"),
                        "ransac_inliers": registration.summary.get("ransac_inliers"),
                        "inlier_ratio": registration.summary.get("inlier_ratio"),
                        "median_reprojection_error": registration.summary.get("median_reprojection_error"),
                        "p90_reprojection_error": registration.summary.get("p90_reprojection_error"),
                        "plain_hull_coverage": registration.summary.get("plain_hull_coverage"),
                        "roll_hull_coverage": registration.summary.get("roll_hull_coverage"),
                        "overlap_plain_pixels": registration.summary.get("overlap_plain_pixels"),
                        "overlap_roll_pixels": registration.summary.get("overlap_roll_pixels"),
                        "registration_relative_local_path": registration_path.relative_to(PROJECT_ROOT).as_posix(),
                        "registration_sha256": sha256_file(registration_path),
                    }
                )
            print(f"cross-resolution registrations={record_index}/20", flush=True)
    fieldnames = list(registration_rows[0])
    write_csv(summary_path, registration_rows, fieldnames)
    def _effective_status(row: dict[str, Any], analysis: str) -> str:
        if row.get(f"{analysis}_analysis_status") != "OK":
            return "INVALID"
        return str(row["status"])

    status_summary = {
        analysis: {
            pair_type: {
                status: sum(
                    str(row["ppi"]) == str(ppi)
                    and row["pair_type"] == pair_type
                    and _effective_status(row, analysis) == status
                    for row in registration_rows
                )
                for status in ("VALID", "AMBIGUOUS", "INVALID")
            }
            for pair_type in ("mated", "non_mated")
        }
        for analysis in SCALE_FACTOR_BANDS
    }
    registration_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "configuration_id": model_manifest["configuration_id"],
        "ppi": ppi,
        "status": "REGISTRATIONS_FROZEN_BEFORE_DETECTOR_SCORING",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "counts": status_summary,
        "summary_csv_sha256": sha256_file(summary_path),
    }
    write_json(ARTIFACT_ROOT / f"sd300_registration_manifest_{ppi}.json", registration_manifest)
    print(json.dumps(registration_manifest, indent=2))


if __name__ == "__main__":
    main()
