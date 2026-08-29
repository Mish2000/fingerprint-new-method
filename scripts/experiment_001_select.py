#!/usr/bin/env python3
"""Freeze the deterministic sample for SD300 Level-3 experiment 001.

This script deliberately reads filenames, checksum manifests, and file bytes only.
It never decodes image pixels, so the sample is fixed before image quality is seen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fingerprint_new_method.paths import dataset_path

EXPERIMENT_ID = "001-sd300-level3-feasibility"
SAMPLING_SEED = "nist-sd300-level3-exp001-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = dataset_path("NIST")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiment-001" / "selection"

# This exclusion was identified only after the initial sample had been frozen and
# pixels were decoded. The 500/1000 plain files contain the same fingerprint,
# while the officially checksummed 2000 plain file decodes as horizontal scan
# lines over a large black field. This is objective source corruption, not a
# quality-based rejection, and therefore invokes the pre-registered replacement
# rule.
OBJECTIVE_SOURCE_EXCLUSIONS = {
    ("00001724", 5): {
        "reason": "OBJECTIVE_SOURCE_CORRUPTION",
        "source_id": "sd300c/images/2000/png/plain/00001724_plain_2000_05.png",
        "details": (
            "The 500 and 1000 ppi plain files decode as the expected fingerprint, "
            "but the 2000 ppi plain file decodes as horizontal scan-line noise over "
            "a large black field. The file matches the distributed checksum, so the "
            "corruption is present in the local official source payload."
        ),
        "identified_after_initial_freeze": True,
    }
}

PPI_ROOTS = {
    500: Path("sd300a/images/500"),
    1000: Path("sd300b/images/1000"),
    2000: Path("sd300c/images/2000"),
}

FINGERS = {
    1: ("right_thumb", 11),
    2: ("right_index", 2),
    3: ("right_middle", 3),
    4: ("right_ring", 4),
    5: ("right_little", 5),
    6: ("left_thumb", 12),
    7: ("left_index", 7),
    8: ("left_middle", 8),
    9: ("left_ring", 9),
    10: ("left_little", 10),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_manifest_path(dataset_root: Path, ppi: int, impression: str) -> Path:
    return dataset_root / PPI_ROOTS[ppi] / f"checksum_{ppi}_png_{impression}.csv"


def load_checksum_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            filename = row.get("filename") or row.get("name")
            checksum = row.get("sha256")
            if not filename or not checksum:
                raise ValueError(f"Malformed checksum row in {path}: {row}")
            result[filename] = checksum.lower()
    return result


def source_record(
    dataset_root: Path,
    checksum_maps: dict[tuple[int, str], dict[str, str]],
    subject_id: str,
    anatomical_position: int,
    ppi: int,
    impression: str,
) -> dict[str, Any]:
    plain_code = FINGERS[anatomical_position][1]
    frgp = plain_code if impression == "plain" else anatomical_position
    filename = f"{subject_id}_{impression}_{ppi}_{frgp:02d}.png"
    relative_path = PPI_ROOTS[ppi] / "png" / impression / filename
    absolute_path = dataset_root / relative_path
    official_hash = checksum_maps[(ppi, impression)].get(filename)
    return {
        "ppi": ppi,
        "impression": impression,
        "frgp": frgp,
        "filename": filename,
        "source_id": relative_path.as_posix(),
        "absolute_path": str(absolute_path),
        "exists": absolute_path.is_file(),
        "official_sha256": official_hash,
    }


def candidate_sources(
    dataset_root: Path,
    checksum_maps: dict[tuple[int, str], dict[str, str]],
    subject_id: str,
    anatomical_position: int,
) -> list[dict[str, Any]]:
    return [
        source_record(
            dataset_root,
            checksum_maps,
            subject_id,
            anatomical_position,
            ppi,
            impression,
        )
        for impression in ("plain", "roll")
        for ppi in (500, 1000, 2000)
    ]


def rank_digest(subject_id: str, anatomical_position: int) -> str:
    payload = f"{SAMPLING_SEED}|{anatomical_position:02d}|{subject_id}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def discover_subjects(dataset_root: Path) -> list[str]:
    roll_dir = dataset_root / PPI_ROOTS[2000] / "png" / "roll"
    subjects = {
        path.name.split("_", maxsplit=1)[0]
        for path in roll_dir.glob("*_roll_2000_*.png")
    }
    return sorted(subjects)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "subject_id",
        "anatomical_position",
        "finger_name",
        "plain_frgp",
        "roll_frgp",
        "selection_rank_sha256",
        "availability_candidate_count",
        "integrity_status",
        "plain_500_source_id",
        "plain_500_sha256",
        "plain_1000_source_id",
        "plain_1000_sha256",
        "plain_2000_source_id",
        "plain_2000_sha256",
        "roll_500_source_id",
        "roll_500_sha256",
        "roll_1000_source_id",
        "roll_1000_sha256",
        "roll_2000_source_id",
        "roll_2000_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flat: dict[str, Any] = {
                key: record[key]
                for key in fieldnames
                if key in record
            }
            for source in record["sources"]:
                prefix = f"{source['impression']}_{source['ppi']}"
                flat[f"{prefix}_source_id"] = source["source_id"]
                flat[f"{prefix}_sha256"] = source["actual_sha256"]
            writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    if project_root not in output_dir.parents:
        raise ValueError(f"Output must remain under {project_root}; got {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checksum_maps = {
        (ppi, impression): load_checksum_manifest(
            checksum_manifest_path(dataset_root, ppi, impression)
        )
        for ppi in (500, 1000, 2000)
        for impression in ("plain", "roll")
    }
    subjects = discover_subjects(dataset_root)
    selected_subjects: set[str] = set()
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}

    for position in range(1, 11):
        finger_name, plain_frgp = FINGERS[position]
        candidates: list[tuple[str, str, list[dict[str, Any]]]] = []
        for subject_id in subjects:
            sources = candidate_sources(
                dataset_root, checksum_maps, subject_id, position
            )
            if all(source["exists"] and source["official_sha256"] for source in sources):
                candidates.append((rank_digest(subject_id, position), subject_id, sources))
        candidates.sort(key=lambda item: (item[0], item[1]))
        eligible_counts[f"{position:02d}"] = len(candidates)

        chosen_for_position = 0
        for rank_hash, subject_id, sources in candidates:
            if chosen_for_position == 2:
                break
            if subject_id in selected_subjects:
                continue

            objective_exclusion = OBJECTIVE_SOURCE_EXCLUSIONS.get(
                (subject_id, position)
            )
            if objective_exclusion is not None:
                exclusions.append(
                    {
                        "subject_id": subject_id,
                        "anatomical_position": position,
                        "selection_rank_sha256": rank_hash,
                        **objective_exclusion,
                    }
                )
                continue

            integrity_errors: list[str] = []
            for source in sources:
                actual_hash = sha256_file(Path(source["absolute_path"]))
                source["actual_sha256"] = actual_hash
                source.pop("absolute_path")
                if actual_hash != source["official_sha256"]:
                    integrity_errors.append(
                        f"{source['source_id']}: expected {source['official_sha256']}, "
                        f"got {actual_hash}"
                    )
            if integrity_errors:
                exclusions.append(
                    {
                        "subject_id": subject_id,
                        "anatomical_position": position,
                        "selection_rank_sha256": rank_hash,
                        "reason": "SOURCE_HASH_MISMATCH",
                        "details": integrity_errors,
                    }
                )
                continue

            selected_subjects.add(subject_id)
            chosen_for_position += 1
            selected.append(
                {
                    "sample_index": len(selected) + 1,
                    "subject_id": subject_id,
                    "anatomical_position": position,
                    "finger_name": finger_name,
                    "plain_frgp": plain_frgp,
                    "roll_frgp": position,
                    "selection_rank_sha256": rank_hash,
                    "availability_candidate_count": len(candidates),
                    "integrity_status": "PASS",
                    "sources": sources,
                }
            )

        if chosen_for_position != 2:
            raise RuntimeError(
                f"Position {position} yielded {chosen_for_position} valid unique subjects"
            )

    if len(selected) != 20 or len(selected_subjects) != 20:
        raise AssertionError("Selection must contain exactly 20 distinct subjects")

    selection_payload = {
        "experiment_id": EXPERIMENT_ID,
        "selection_revision": 2,
        "revised_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_sample_pixels_decoded_before_initial_freeze": False,
        "replacement_candidate_pixels_decoded_before_selection": False,
        "initial_manifest_snapshot": "selection_manifest_initial.json",
        "sampling_seed": SAMPLING_SEED,
        "selection_rule": (
            "For anatomical positions 01..10 in ascending order, enumerate subjects "
            "with all six expected files and official checksum entries; rank by "
            "SHA256(seed|position|subject); select the first two SHA256-verified "
            "candidates whose subject has not already been selected. Only an objective "
            "missing, unreadable, decoded-corrupt, or checksum-failing source permits "
            "replacement."
        ),
        "plain_to_anatomical_mapping": {
            f"{plain_frgp:02d}": f"{position:02d}"
            for position, (_, plain_frgp) in FINGERS.items()
        },
        "eligible_counts_by_position": eligible_counts,
        "selected": selected,
        "exclusions_and_replacements": exclusions,
    }
    json_path = output_dir / "selection_manifest.json"
    csv_path = output_dir / "selection_manifest.csv"
    json_path.write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, selected)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Selected {len(selected)} fingers from {len(selected_subjects)} subjects")
    print(f"Objective exclusions/replacements: {len(exclusions)}")
    for record in selected:
        print(
            f"{record['sample_index']:02d}  pos={record['anatomical_position']:02d}  "
            f"subject={record['subject_id']}  finger={record['finger_name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
