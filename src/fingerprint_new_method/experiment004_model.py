# ruff: noqa: I001
"""Frozen U-Net and training/inference helpers for Experiment 004.

PyTorch is an experiment-only dependency. Importing the repository continues to
work without it; functions in this module fail with an actionable message when
the optional runtime is absent.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    import torch.nn.functional as torch_functional
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light CI
    torch = None
    torch_functional = None
    nn = None
    DataLoader = None
    Dataset = object

import numpy as np

from fingerprint_new_method.experiment004 import (
    LOCAL_ROOT,
    PROJECT_ROOT,
    TRAINING_SEEDS,
    blend_tiles,
    gaussian_target,
    read_annotations,
    resolve_source_id,
    sample_affine_parameters,
    validate_resume_state,
)


MODEL_CONFIGURATION: dict[str, Any] = {
    "architecture": "unet-4-level-groupnorm-v1",
    "input_channels": 1,
    "output_channels": 1,
    "encoder_widths": [32, 64, 128, 256],
    "bottleneck_width": 512,
    "convolutions_per_block": 2,
    "normalization": "GroupNorm(8)",
    "activation": "ReLU",
    "downsampling": "MaxPool2d(2)",
    "upsampling": "bilinear align_corners=False + 1x1 Conv",
    "pretrained": False,
    "target_sigma_px": 1.5,
    "target_truncate_sigma": 4.0,
    "loss": {"name": "binary_focal_soft_targets", "alpha": 0.95, "gamma": 2.0},
    "optimizer": {"name": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4},
    "batch_size": 2,
    "gradient_accumulation": 2,
    "maximum_epochs": 80,
    "minimum_epochs_before_early_stop": 20,
    "early_stopping_patience": 12,
    "early_stopping_min_delta": 1e-5,
    "scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-6, "t_max": 80},
    "gradient_norm_clip": 1.0,
    "training_seeds": list(TRAINING_SEEDS),
}


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "Experiment 004 requires PyTorch. Install the frozen runtime with "
            "'.conda-env/python -m pip install torch==2.11.0 "
            "--index-url https://download.pytorch.org/whl/cu128'."
        )


if nn is not None:

    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.layers(values)


    class UpBlock(nn.Module):
        def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
            super().__init__()
            self.reduce = nn.Conv2d(input_channels, output_channels, kernel_size=1)
            self.block = ConvBlock(output_channels + skip_channels, output_channels)

        def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            values = torch_functional.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            values = self.reduce(values)
            return self.block(torch.cat([skip, values], dim=1))


    class PoreUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = MODEL_CONFIGURATION["encoder_widths"]
            self.encoder_1 = ConvBlock(1, widths[0])
            self.encoder_2 = ConvBlock(widths[0], widths[1])
            self.encoder_3 = ConvBlock(widths[1], widths[2])
            self.encoder_4 = ConvBlock(widths[2], widths[3])
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(widths[3], MODEL_CONFIGURATION["bottleneck_width"])
            self.up_4 = UpBlock(512, widths[3], widths[3])
            self.up_3 = UpBlock(widths[3], widths[2], widths[2])
            self.up_2 = UpBlock(widths[2], widths[1], widths[1])
            self.up_1 = UpBlock(widths[1], widths[0], widths[0])
            self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            skip_1 = self.encoder_1(values)
            skip_2 = self.encoder_2(self.pool(skip_1))
            skip_3 = self.encoder_3(self.pool(skip_2))
            skip_4 = self.encoder_4(self.pool(skip_3))
            values = self.bottleneck(self.pool(skip_4))
            values = self.up_4(values, skip_4)
            values = self.up_3(values, skip_3)
            values = self.up_2(values, skip_2)
            values = self.up_1(values, skip_1)
            return self.output(values)


else:

    class PoreUNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            require_torch()


def model_parameter_count() -> int:
    require_torch()
    return int(sum(parameter.numel() for parameter in PoreUNet().parameters()))


def focal_heatmap_loss(logits: Any, targets: Any, *, alpha: float = 0.95, gamma: float = 2.0) -> Any:
    require_torch()
    probabilities = torch.sigmoid(logits)
    cross_entropy = torch_functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_true = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_weight * (1.0 - probability_true).pow(gamma) * cross_entropy).mean()


def _derived_seed(seed: int, epoch: int, canonical_image_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{epoch}|{canonical_image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class L3SFDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]], *, augment: bool, seed: int) -> None:
        require_torch()
        self.records = list(records)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.cached_images: list[np.ndarray] = []
        self.cached_points: list[np.ndarray] = []
        self.cached_targets: list[np.ndarray] = []
        for record in self.records:
            cache_path = preprocessed_cache_path(record)
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing deterministic preprocessing cache {cache_path}; run experiment_004_preprocess_cache.py"
                )
            cached = np.load(cache_path, allow_pickle=False)
            if cached.shape != (512, 512) or cached.dtype != np.uint8:
                raise ValueError(f"Invalid preprocessing cache {cache_path}: {cached.shape}, {cached.dtype}")
            points, _ = _load_unique_points(record)
            self.cached_images.append(cached)
            self.cached_points.append(points)
            if not self.augment:
                self.cached_targets.append(gaussian_target(cached.shape, points, sigma=1.5, truncate=4.0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any, str, Any]:
        record = self.records[index]
        cached = self.cached_images[index]
        processed = cached.astype(np.float32) / np.float32(255.0)
        points = self.cached_points[index]
        image_tensor = torch.from_numpy(np.ascontiguousarray(processed[None, :, :]))
        if self.augment:
            generator = np.random.default_rng(_derived_seed(self.seed, self.epoch, record["canonical_image_id"]))
            parameters = sample_affine_parameters(generator)
            points, augmentation = _augmentation_specification(points, 512, 512, parameters)
            target = gaussian_target(processed.shape, points, sigma=1.5, truncate=4.0)
        else:
            augmentation = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            target = self.cached_targets[index]
        target_tensor = torch.from_numpy(np.ascontiguousarray(target[None, :, :]))
        return image_tensor, target_tensor, str(record["canonical_image_id"]), torch.from_numpy(augmentation)


def _load_unique_points(record: dict[str, Any]) -> tuple[np.ndarray, int]:
    from fingerprint_new_method.experiment004 import deduplicate_points

    points = read_annotations(resolve_source_id(record["annotation_source_id"]))
    unique, duplicates = deduplicate_points(points)
    return np.asarray(unique, dtype=np.float32).reshape(-1, 2), duplicates


def preprocessed_cache_path(record: dict[str, Any]) -> Path:
    run, stem = str(record["canonical_image_id"]).split("/", maxsplit=1)
    return LOCAL_ROOT / "preprocessed" / "l3sf" / run / f"{stem}.npy"


def _affine_matrix_without_opencv(width: int, height: int, parameters: dict[str, float]) -> np.ndarray:
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    angle = np.deg2rad(parameters["angle_degrees"])
    alpha = parameters["scale"] * float(np.cos(angle))
    beta = parameters["scale"] * float(np.sin(angle))
    return np.asarray(
        [
            [
                alpha,
                beta,
                (1.0 - alpha) * center_x - beta * center_y + parameters["translate_x"],
            ],
            [
                -beta,
                alpha,
                beta * center_x + (1.0 - alpha) * center_y + parameters["translate_y"],
            ],
        ],
        dtype=np.float64,
    )


def _augmentation_specification(
    points: np.ndarray,
    width: int,
    height: int,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _affine_matrix_without_opencv(width, height, parameters)
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    inverse_00 = matrix[1, 1] / determinant
    inverse_01 = -matrix[0, 1] / determinant
    inverse_10 = -matrix[1, 0] / determinant
    inverse_11 = matrix[0, 0] / determinant
    inverse_02 = -(inverse_00 * matrix[0, 2] + inverse_01 * matrix[1, 2])
    inverse_12 = -(inverse_10 * matrix[0, 2] + inverse_11 * matrix[1, 2])
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    transformed_x = matrix[0, 0] * point_array[:, 0] + matrix[0, 1] * point_array[:, 1] + matrix[0, 2]
    transformed_y = matrix[1, 0] * point_array[:, 0] + matrix[1, 1] * point_array[:, 1] + matrix[1, 2]
    valid = (
        (transformed_x >= 0.0)
        & (transformed_x <= width - 1)
        & (transformed_y >= 0.0)
        & (transformed_y <= height - 1)
    )
    transformed = np.column_stack([transformed_x[valid], transformed_y[valid]]).astype(np.float32)
    specification = np.asarray(
        [
            inverse_00,
            inverse_01,
            inverse_02,
            inverse_10,
            inverse_11,
            inverse_12,
            parameters["contrast"],
            parameters["brightness"],
        ],
        dtype=np.float32,
    )
    return transformed, specification


def _apply_batch_augmentation(images: Any, specifications: Any, x_grid: Any, y_grid: Any) -> Any:
    height, width = images.shape[-2:]
    input_x = (
        specifications[:, 0, None, None] * x_grid
        + specifications[:, 1, None, None] * y_grid
        + specifications[:, 2, None, None]
    )
    input_y = (
        specifications[:, 3, None, None] * x_grid
        + specifications[:, 4, None, None] * y_grid
        + specifications[:, 5, None, None]
    )
    normalized_x = input_x * (2.0 / (width - 1)) - 1.0
    normalized_y = input_y * (2.0 / (height - 1)) - 1.0
    sampling_grid = torch.stack([normalized_x, normalized_y], dim=-1)
    warped = torch_functional.grid_sample(
        images,
        sampling_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    contrast = specifications[:, 6, None, None, None]
    brightness = specifications[:, 7, None, None, None]
    return ((warped - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)


def configure_determinism(seed: int) -> dict[str, Any]:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    deterministic_error = None
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as error:  # pragma: no cover - backend-specific
        deterministic_error = str(error)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms_requested": True,
        "deterministic_configuration_error": deterministic_error,
    }


def runtime_manifest() -> dict[str, Any]:
    require_torch()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv_distribution": importlib.metadata.version("opencv-python-headless"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }


def _validation_loss(model: Any, loader: Any, device: Any, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_images = 0
    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = focal_heatmap_loss(logits, targets)
            batch_count = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_images += batch_count
    return total_loss / total_images


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint so an interrupted process cannot leave a truncated file."""

    require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume_state(path: Path, seed: int) -> dict[str, Any] | None:
    """Return a compatible epoch-boundary training state, or None to start fresh.

    Resuming is an operational property only: the data order generator is seeded
    per epoch, augmentation randomness is derived from (seed, epoch, image id),
    and the network contains no stochastic layers, so a resumed run continues the
    same deterministic trajectory the uninterrupted run would have produced.
    """

    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return validate_resume_state(state, seed=seed, model_configuration=MODEL_CONFIGURATION)
    except RuntimeError as error:
        raise RuntimeError(f"{error} ({path})") from error


def train_seed(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    *,
    seed: int,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    require_torch()
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"Seed {seed} is not preregistered")
    determinism = configure_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = PoreUNet().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    train_dataset = L3SFDataset(train_records, augment=True, seed=seed)
    validation_dataset = L3SFDataset(validation_records, augment=False, seed=seed)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_amp)
    if checkpoint_path is None:
        checkpoint_path = LOCAL_ROOT / "checkpoints" / f"seed-{seed}" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(512, dtype=torch.float32, device=device),
        torch.arange(512, dtype=torch.float32, device=device),
        indexing="ij",
    )

    best_loss = float("inf")
    early_stop_best = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    resume_events: list[dict[str, Any]] = []
    first_epoch = 1
    elapsed_before = 0.0
    resume_path = checkpoint_path.with_name("last.pt")
    resume_state = _load_resume_state(resume_path, seed)
    if resume_state is not None:
        model.load_state_dict(resume_state["state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        scaler.load_state_dict(resume_state["scaler_state"])
        best_loss = float(resume_state["best_validation_loss"])
        early_stop_best = float(resume_state["early_stop_best"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        history = list(resume_state["history"])
        resume_events = list(resume_state.get("resume_events", []))
        elapsed_before = float(resume_state["elapsed_seconds"])
        first_epoch = int(resume_state["epoch"]) + 1
        resume_events.append(
            {
                "resumed_at_utc": datetime.now(UTC).isoformat(),
                "resumed_after_epoch": int(resume_state["epoch"]),
                "reason": "Interrupted process restarted from the last completed epoch boundary",
            }
        )
        print(f"seed={seed} resuming after epoch {resume_state['epoch']}", flush=True)
    started = time.perf_counter() - elapsed_before
    for epoch in range(first_epoch, 81):
        train_dataset.set_epoch(epoch)
        order_generator = torch.Generator()
        order_generator.manual_seed(seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            generator=order_generator,
            num_workers=0,
            pin_memory=use_amp,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        image_count = 0
        accumulation_count = 0
        for batch_index, (images, targets, _, specifications) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            specifications = specifications.to(device, non_blocking=True)
            images = _apply_batch_augmentation(images, specifications, x_grid, y_grid)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                unscaled_loss = focal_heatmap_loss(logits, targets)
                loss = unscaled_loss / 2.0
            scaler.scale(loss).backward()
            accumulation_count += 1
            if accumulation_count == 2 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
            batch_count = int(images.shape[0])
            epoch_loss += float(unscaled_loss.detach().cpu()) * batch_count
            image_count += batch_count
        validation_loss = _validation_loss(model, validation_loader, device, use_amp)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_loss = epoch_loss / image_count
        checkpoint_improved = validation_loss < best_loss - 1e-8 or (
            abs(validation_loss - best_loss) <= 1e-8 and best_epoch == 0
        )
        if checkpoint_improved:
            best_loss = validation_loss
            best_epoch = epoch
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model_configuration": MODEL_CONFIGURATION,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        if validation_loss < early_stop_best - 1e-5:
            early_stop_best = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "checkpoint_selected": checkpoint_improved,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        scheduler.step()
        _atomic_torch_save(
            {
                "schema_version": 1,
                "seed": seed,
                "epoch": epoch,
                "model_configuration": MODEL_CONFIGURATION,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "early_stop_best": early_stop_best,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": time.perf_counter() - started,
                "history": history,
                "resume_events": resume_events,
            },
            resume_path,
        )
        print(
            f"seed={seed} epoch={epoch:03d} train={training_loss:.8f} "
            f"validation={validation_loss:.8f} best={best_loss:.8f}@{best_epoch}",
            flush=True,
        )
        if epoch >= 20 and epochs_without_improvement >= 12:
            break

    if not checkpoint_path.is_file():
        raise RuntimeError(f"Training did not produce checkpoint {checkpoint_path}")
    resume_path.unlink(missing_ok=True)
    return {
        "seed": seed,
        "status": "COMPLETED",
        "device": str(device),
        "mixed_precision": use_amp,
        "memory_format": "channels_last",
        "determinism": determinism,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_path": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "resume_events": resume_events,
        "history": history,
    }


def load_model(checkpoint_path: Path, *, device: str | None = None) -> tuple[Any, Any, dict[str, Any]]:
    require_torch()
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    if checkpoint.get("model_configuration") != MODEL_CONFIGURATION:
        raise ValueError(f"Checkpoint model configuration mismatch: {checkpoint_path}")
    model = PoreUNet().to(selected_device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, selected_device, checkpoint


def infer_heatmap(
    model: Any,
    image: np.ndarray,
    device: Any,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """Infer one merged sigmoid heatmap from a preprocessed image."""

    require_torch()
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional preprocessed image")
    original_height, original_width = values.shape
    padded_height = max(tile_size, original_height)
    padded_width = max(tile_size, original_width)
    bottom = padded_height - original_height
    right = padded_width - original_width
    padded = np.pad(values, ((0, bottom), (0, right)), mode="reflect") if bottom or right else values
    from fingerprint_new_method.experiment004 import tile_origins

    y_origins = tile_origins(padded_height, tile_size=tile_size, overlap=overlap)
    x_origins = tile_origins(padded_width, tile_size=tile_size, overlap=overlap)
    locations = [(y_value, x_value) for y_value in y_origins for x_value in x_origins]
    predictions: list[tuple[int, int, np.ndarray]] = []
    use_amp = getattr(device, "type", str(device)) == "cuda"
    with torch.no_grad():
        for start in range(0, len(locations), batch_size):
            batch_locations = locations[start : start + batch_size]
            batch = np.stack(
                [padded[y_value : y_value + tile_size, x_value : x_value + tile_size] for y_value, x_value in batch_locations]
            )[:, None, :, :]
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(
                device,
                memory_format=torch.channels_last,
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(tensor)
                probabilities = torch.sigmoid(logits)
            arrays = probabilities[:, 0].float().cpu().numpy()
            predictions.extend(
                (y_value, x_value, array) for (y_value, x_value), array in zip(batch_locations, arrays, strict=True)
            )
    merged = blend_tiles((padded_height, padded_width), predictions, tile_size=tile_size, ramp=overlap // 2)
    return merged[:original_height, :original_width]
