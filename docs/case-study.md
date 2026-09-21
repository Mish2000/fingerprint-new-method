# Pore Localization and Synthetic-to-Real Feasibility

Experiment 004 developed and evaluated a fingerprint pore localizer trained
from scratch. Median synthetic test F1@4 was **0.9776**, compared with **0.7788**
for a local classical baseline. The real-data transfer assessment remained
**inconclusive**. The recorded conclusion is `SYNTHETIC_LOCALIZATION_ONLY`;
Experiment 004 is complete and further experiments are paused.

## Start by establishing what can be measured

The research question was whether point annotations could support a learned pore
localizer, and whether its frozen outputs could then be assessed on real scans.
That required checking the data before training: visible dots, annotation
coordinates, and repeatable anatomical detail are different kinds of evidence.

[Experiment 001](experiments/001-sd300-level3-feasibility-results.md) examined fine
detail in SD300 scanned ink cards and its repeatability across scans and
impressions. Its observations exposed ambiguity from image quality and alignment.
[Experiment 002](experiments/002-l3sf-pore-annotation-feasibility-results.md)
qualified a synthetic annotated subset for localization, including coordinate
conventions and duplicate labels. These audits shaped a bounded learning task;
they did not establish fingerprint recognition performance.

## Keep the data branches and split units distinct

The development dataset was **740 synthetic 512 × 512 images** from the
`annotated_512` branch of [L3-SF](https://andrewyzy.github.io/L3-SF/), with paired
pore-coordinate annotations. The separate `final_320` branch contains **7,400
images**. [Experiment 003](experiments/003-l3sf-annotated-final-crosswalk-results.md)
returned `NO_RELIABLE_CROSSWALK`: its evidence did not justify mapping identities
or transferring labels between the branches. Experiment 004 therefore used only
the annotated branch. Image dimensions do not supply a physical sensor PPI for
these synthetic images.

The [frozen split](../artifacts/experiment-004/split_manifest.json) grouped images
by `pattern + local_index`, keeping all five dataset runs in the same partition.
This conservative leakage unit makes no claim that those runs represent the same
biometric identity. Hash-based ranking and pattern quotas fixed the allocation.

| Partition | Images | Leakage groups |
| --- | ---: | ---: |
| Train | 440 | 88 |
| Validation | 150 | 30 |
| Test | 150 | 30 |

The annotation audit removed **241 exact duplicate coordinate records** while
preserving the first occurrence. Edge annotations and ambiguous labels remained
in the main metric; edge results were also reported separately. This avoided
selecting a cleaner test population after seeing predictions.

## Build the learning and evaluation pipeline

The [model](../src/fingerprint_new_method/experiment004_model.py) is a four-level
U-Net with GroupNorm, bilinear upsampling, skip connections, and **7,240,225
parameters**, trained without pretrained weights. U-Net originates with
[Ronneberger, Fischer, and Brox](https://arxiv.org/abs/1505.04597); the contribution
here is the task implementation and experimental controls, not architectural
novelty.

Grayscale images undergo percentile normalization and CLAHE. Annotation points
become Gaussian heatmap targets. Training augmentation transforms images and
coordinates together, with additional contrast and brightness variation. AdamW,
soft-target focal loss, gradient accumulation, and early stopping follow the
[frozen model configuration](../artifacts/experiment-004/model_protocol.json).

Seeds **40401, 40402, and 40403** used the same architecture and split. Each
checkpoint was selected by validation loss. A shared probability threshold and
non-maximum suppression (NMS) radius were selected by median validation F1@4 across seeds, then frozen
before test access. The classical baseline uses bright local extrema from an
image-minus-Gaussian-blur response; its parameters were also selected on
validation. It is this experiment's baseline, not an external author system.
The [inference manifest](../artifacts/experiment-004/model_manifest.json) records
both selections. The [pipeline diagram](../README.md#pipeline) shows the path from
image to heatmap, points, and localization metrics.

Engineering work also addressed interrupted execution. The
[training record](../artifacts/experiment-004/training_runs.json) retains **10
attempts: three completed and seven incomplete**, including runtime failures and
deliberate interruptions. OpenCV preprocessing was moved into a local cache after
a native runtime collision. Epoch-boundary resume state restores model,
optimizer, scheduler, and mixed-precision state; batch order and augmentation
derive from recorded seeds. The historical record documents recovery on the
recorded host, not numerical reproducibility on every runtime or GPU.

## Measure localization against annotations

F1@4 uses one-to-one prediction-to-annotation matching within **4 pixels** in the
original synthetic image frame. Matching maximizes the number of pairs, then
minimizes total Euclidean distance. Unmatched predictions are false positives
(FP); unmatched annotations are false negatives (FN). Matched pairs are true
positives (TP).

For each seed, TP, FP, and FN are summed over all **150 test images**, then
`F1 = 2TP / (2TP + FP + FN)`. This is pooled point-level F1, not the mean of
per-image F1 scores or fingerprint identification accuracy. The headline median
is taken across the three seed scores.

| Test result | Precision@4 | Recall@4 | F1@4 |
| --- | ---: | ---: | ---: |
| U-Net, seed 40401 | 0.9916 | 0.9655 | 0.9784 |
| U-Net, seed 40402 | 0.9918 | 0.9639 | 0.9776 |
| U-Net, seed 40403 | 0.9889 | 0.9636 | 0.9761 |
| Local classical baseline | 0.8360 | 0.7290 | 0.7788 |

Values are rounded from [test metrics](../artifacts/experiment-004/test_metrics.json)
and [baseline metrics](../artifacts/experiment-004/baseline_metrics.json).
The median F1 advantage is **0.1988**, or **19.88 percentage points** on the F1
scale, calculated before rounding. Gate A was `STRONG_PASS` within this synthetic
domain. Three seeds assess training variation on one split; the **450 evaluation
rows are 150 images × three seeds**, not additional images or participants.
These results establish neither SOTA nor superiority over external matchers.

## Diagnose the limits of the transfer measurement

SD300 was held outside training and parameter selection. Its primary analysis
used **1000 PPI** scans, followed by a **2000 PPI** sensitivity analysis of related
scans, not an independent population. Ridge-scale normalization and registration
provided the frame for comparing detections across impressions.

| Registration coverage | 1000 PPI, primary | 2000 PPI, sensitivity |
| --- | ---: | ---: |
| Valid mated pairs | 0/20 | 2/20 |
| Valid non-mated pairs | 0/20 | 0/20 |

At 1000 PPI, ridge-period estimates were unreliable for **53/60 images**;
preprocessing blocked every planned registration. At 2000 PPI all images passed
preprocessing, but registration coverage remained limited. The planned effect,
`Delta = repeatability_mated - repeatability_non_mated`, required both
registrations to be valid. The non-mated control required aligning different
fingers while validity rules rejected weak anatomical correspondence: a
structural measurement limitation.

Consequently, Delta and its confidence interval were unavailable (`null`), not
zero. [Gate B](../artifacts/experiment-004/sd300_transfer_summary.json) was
`TRANSFER_INCONCLUSIVE`. The separately documented
[scale-guard amendment](experiments/004-scale-guard-contingency-amendment.md)
defined an exploratory sensitivity variant before SD300 access; it did not
replace the primary analysis or decide the final outcome.

## What the evidence supports

**Observation:** the localizer agreed closely with synthetic labels, while the
frozen transfer measurement lacked valid comparison coverage. **Interpretation:**
the study separates success on the supervised task from an unresolved domain
transfer question. It does not isolate a definitive failure of the detector to
generalize, establish anatomical accuracy on SD300, or demonstrate recognition,
liveness, security, or production readiness.

**Possible follow-up:** independently qualify the registration and negative
control design before another transfer study. This is a proposal requiring a
separate protocol, not work completed here. The
[historical result](experiments/004-pore-localization-and-sd300-transfer-results.md)
and [final summary](../artifacts/experiment-004/summary.json) remain the evidence
for the completed experiment, with their original source identities.

Readers can inspect summaries and run the [software checks](../CONTRIBUTING.md)
without private data or weights. Those checks do not reproduce training or
evaluation. L3-SF is credited to Wyzykowski, Segundo, and Lemes; its terms and
SD300 restrictions are described in [data and licensing](data-and-licensing.md).
Source pixels, weights, and heatmaps remain local. Public access to this repository
does not override its [all-rights-reserved license](../LICENSE).
