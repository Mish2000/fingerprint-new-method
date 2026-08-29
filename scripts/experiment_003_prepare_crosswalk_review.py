"""Prepare blinded, local pixel plates for the frozen Experiment 003 review."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fingerprint_new_method.experiment003 import deterministic_rank, write_csv, write_json
from fingerprint_new_method.paths import PROJECT_ROOT, dataset_path

EXPERIMENT_ID = "003-l3sf-annotated-final-crosswalk"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "experiment-003"
EVIDENCE_ROOT = OUTPUT_ROOT / "evidence-pixels" / "crosswalk-review"
FINAL_ROOT = dataset_path("L3_SF_V2", "L3SF_V2", "L3-SF")
ANNOTATED_ROOT = dataset_path("L3_SF_V2", "L3SF_V2", "Pore ground truth", "Fingerprint Images")
REVIEW_SEED = "l3sf-exp003-crosswalk-review-v1"


def read_crosswalk() -> dict[str, dict[str, str]]:
    with (OUTPUT_ROOT / "crosswalk.csv").open(encoding="utf-8", newline="") as handle:
        return {row["annotated_512_id"]: row for row in csv.DictReader(handle)}


def candidate_alternative(row: dict[str, str], proposed: int) -> tuple[int, str]:
    ordered = [
        (int(row["k_top1_identity"]), "k_top1"),
        (int(row["o_top1_identity"]), "o_top1"),
        (int(row["k_top2_identity"]), "k_top2"),
        (int(row["o_top2_identity"]), "o_top2"),
    ]
    for identity, source in ordered:
        if identity != proposed:
            return identity, source
    raise RuntimeError("Could not find a candidate distinct from the diagnostic assignment")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, fill=(20, 20, 20), font=font)


def final_paths(run: str, identity: int) -> list[Path]:
    paths = sorted(
        (FINAL_ROOT / run).glob(f"{identity}_*.png"),
        key=lambda path: tuple(int(part) for part in path.stem.split("_")),
    )
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten final images for {run}/{identity}, got {len(paths)}")
    return [paths[index] for index in (0, 5, 9)]


def build_plate(
    *,
    audit_index: int,
    annotated_path: Path,
    run: str,
    group_a_identity: int,
    group_b_identity: int,
    output_path: Path,
) -> None:
    canvas = Image.new("RGB", (1230, 790), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small_font = ImageFont.load_default(size=17)
    draw_centered(draw, (615, 8), f"FROZEN BLIND AUDIT {audit_index:03d}", font)
    draw_centered(draw, (276, 48), "ANNOTATED REFERENCE", small_font)
    draw_centered(draw, (714, 48), "GROUP A", small_font)
    draw_centered(draw, (1060, 48), "GROUP B", small_font)

    annotated = load_rgb(annotated_path)
    canvas.paste(annotated, (20, 155))
    for column_x, identity in ((554, group_a_identity), (900, group_b_identity)):
        for row_index, path in enumerate(final_paths(run, identity)):
            image = load_rgb(path)
            canvas.paste(image, (column_x, 68 + row_index * 240))
    draw.rectangle((19, 154, 532, 667), outline=(50, 50, 50), width=1)
    draw.line((542, 62, 542, 780), fill=(150, 150, 150), width=2)
    draw.line((888, 62, 888, 780), fill=(150, 150, 150), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> None:
    sample_manifest = json.loads((OUTPUT_ROOT / "crosswalk_review_sample_manifest.json").read_text(encoding="utf-8"))
    crosswalk = read_crosswalk()
    key_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for selected in sample_manifest["selected_images"]:
        audit_index = int(selected["audit_index"])
        annotated_id = selected["canonical_annotated_id"]
        row = crosswalk[annotated_id]
        run, stem = annotated_id.split("/", 1)
        proposed = int(row["global_assignment_identity"])
        alternative, alternative_source = candidate_alternative(row, proposed)
        proposed_is_a = int(deterministic_rank(REVIEW_SEED, "blind-side", annotated_id), 16) % 2 == 0
        group_a = proposed if proposed_is_a else alternative
        group_b = alternative if proposed_is_a else proposed
        plate_path = EVIDENCE_ROOT / f"audit-{audit_index:03d}.png"
        build_plate(
            audit_index=audit_index,
            annotated_path=ANNOTATED_ROOT / run / f"{stem}.jpg",
            run=run,
            group_a_identity=group_a,
            group_b_identity=group_b,
            output_path=plate_path,
        )
        task_rows.append(
            {
                "audit_index": audit_index,
                "plate_relative_path": plate_path.relative_to(PROJECT_ROOT).as_posix(),
                "allowed_choice": "A|B|BOTH_UNCERTAIN|NEITHER",
                "allowed_confidence": "CLEAR|AMBIGUOUS",
            }
        )
        key_rows.append(
            {
                "audit_index": audit_index,
                "annotated_512_id": annotated_id,
                "run": run,
                "pattern": selected["pattern"],
                "proposed_identity": proposed,
                "proposed_source": "preregistered_combined_hungarian_diagnostic",
                "alternative_identity": alternative,
                "alternative_source": alternative_source,
                "group_a_identity": group_a,
                "group_b_identity": group_b,
                "proposed_blind_group": "A" if proposed_is_a else "B",
                "channels_agree": row["channels_agree"] == "true",
                "crosswalk_status_before_review": row["status"],
            }
        )
    write_csv(
        OUTPUT_ROOT / "crosswalk_review_blind_tasks.csv",
        task_rows,
        ["audit_index", "plate_relative_path", "allowed_choice", "allowed_confidence"],
    )
    write_json(
        OUTPUT_ROOT / "crosswalk_review_blind_key.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "review_revision": 1,
            "reviewer_must_not_read_before_ratings_are_frozen": True,
            "protocol_contingency": "Because no consensus identity existed for the frozen sample, the preregistered combined Hungarian diagnostic is shown against the first distinct K/O local candidate. This review cannot upgrade AMBIGUOUS mappings or rescue Gate 1.",
            "impression_positions": [1, 6, 10],
            "rows": key_rows,
        },
    )
    print(f"Prepared {len(task_rows)} blind plates under {EVIDENCE_ROOT}")


if __name__ == "__main__":
    main()
