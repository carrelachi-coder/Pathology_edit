"""Pretrain a same-WSI appearance encoder on real patch pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from controlnet_train.data.common import load_image_tensor, load_mask_array
from controlnet_train.training.same_wsi_appearance import (
    SameWSIAppearanceConfig,
    SameWSIAppearanceEncoder,
    SameWSIPairClassifier,
    save_same_wsi_checkpoint,
)


class SameWSIPairDataset(Dataset):
    """Balanced same/different WSI pairs from Cross metadata records."""

    def __init__(
        self,
        metadata_path: str | Path,
        *,
        pairs_per_epoch: int,
        image_size: int = 256,
        seed: int = 42,
        hard_negative_prob: float = 0.8,
        hard_negative_pool_size: int = 32,
        hard_negative_candidate_count: int = 512,
    ) -> None:
        print(f"[same-wsi] loading metadata: {metadata_path}", file=sys.stderr, flush=True)
        self.samples = _samples_from_cross_metadata(metadata_path)
        if len(self.samples) < 2:
            raise ValueError(f"Need at least two samples for same-WSI pretraining, got {len(self.samples)}.")
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.image_size = int(image_size)
        self.rng = random.Random(seed)
        self.hard_negative_prob = min(1.0, max(0.0, float(hard_negative_prob)))
        self.hard_negative_pool_size = max(1, int(hard_negative_pool_size))
        self.hard_negative_candidate_count = max(
            self.hard_negative_pool_size,
            int(hard_negative_candidate_count),
        )
        self.by_case: dict[str, list[dict]] = {}
        for sample in self.samples:
            self.by_case.setdefault(sample["case_key"], []).append(sample)
        self.positive_cases = [case for case, rows in self.by_case.items() if len(rows) >= 2]
        if not self.positive_cases:
            raise ValueError("No case/WSI group has at least two patches for positive same-WSI pairs.")
        print(
            (
                f"[same-wsi] samples={len(self.samples)} cases={len(self.by_case)} "
                f"positive_cases={len(self.positive_cases)} hard_negative_prob={self.hard_negative_prob} "
                f"pool={self.hard_negative_pool_size} candidates={self.hard_negative_candidate_count}"
            ),
            file=sys.stderr,
            flush=True,
        )
        self.hard_negatives = _build_hard_negative_index(
            self.samples,
            pool_size=self.hard_negative_pool_size,
            candidate_count=self.hard_negative_candidate_count,
            rng=self.rng,
        )
        print("[same-wsi] hard-negative index ready", file=sys.stderr, flush=True)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int) -> dict:
        same = index % 2 == 0
        if same:
            case_key = self.rng.choice(self.positive_cases)
            sample_a, sample_b = self.rng.sample(self.by_case[case_key], 2)
        else:
            sample_a = self.rng.choice(self.samples)
            sample_b = self._draw_negative(sample_a)
        return {
            "image_a": _resize(load_image_tensor(sample_a["image_path"]), self.image_size),
            "image_b": _resize(load_image_tensor(sample_b["image_path"]), self.image_size),
            "label": torch.tensor(1.0 if same else 0.0, dtype=torch.float32),
        }

    def _draw_negative(self, sample_a: dict) -> dict:
        hard_pool = self.hard_negatives.get(sample_a["sample_key"], [])
        if hard_pool and self.rng.random() < self.hard_negative_prob:
            return self.rng.choice(hard_pool)
        sample_b = self.rng.choice(self.samples)
        attempts = 0
        while sample_b["case_key"] == sample_a["case_key"] and attempts < 100:
            sample_b = self.rng.choice(self.samples)
            attempts += 1
        if sample_b["case_key"] == sample_a["case_key"]:
            raise ValueError("Could not draw a different-WSI negative pair.")
        return sample_b


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", required=True, help="Cross metadata JSON containing same-case pairs.")
    parser.add_argument("--val-metadata", default=None, help="Optional validation Cross metadata JSON.")
    parser.add_argument("--output-dir", default="same_wsi_appearance")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=20000)
    parser.add_argument("--val-pairs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument(
        "--hard-negative-prob",
        type=float,
        default=0.8,
        help="Probability that a negative pair uses a tissue-composition hard negative.",
    )
    parser.add_argument(
        "--hard-negative-pool-size",
        type=int,
        default=32,
        help="Top-K different-WSI tissue-similar candidates kept per patch for hard negatives.",
    )
    parser.add_argument(
        "--hard-negative-candidate-count",
        type=int,
        default=512,
        help=(
            "Approximate hard-negative search width per patch. Larger is harder but slower; "
            "set near dataset size for exact all-pairs search."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(input_args)


def main(input_args=None) -> None:
    args = parse_args(input_args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SameWSIAppearanceConfig(input_size=args.input_size, embedding_dim=args.embedding_dim)
    model = SameWSIPairClassifier(SameWSIAppearanceEncoder(config)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset = SameWSIPairDataset(
        args.train_metadata,
        pairs_per_epoch=args.pairs_per_epoch,
        image_size=args.input_size,
        seed=args.seed,
        hard_negative_prob=args.hard_negative_prob,
        hard_negative_pool_size=args.hard_negative_pool_size,
        hard_negative_candidate_count=args.hard_negative_candidate_count,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = None
    if args.val_metadata:
        val_dataset = SameWSIPairDataset(
            args.val_metadata,
            pairs_per_epoch=args.val_pairs,
            image_size=args.input_size,
            seed=args.seed + 1000,
            hard_negative_prob=args.hard_negative_prob,
            hard_negative_pool_size=args.hard_negative_pool_size,
            hard_negative_candidate_count=args.hard_negative_candidate_count,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, device=args.device)
        val_metrics = _run_epoch(model, val_loader, optimizer=None, device=args.device) if val_loader else {}
        payload = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        save_same_wsi_checkpoint(
            output_dir / "last.pt",
            model=model,
            extra={"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics},
        )
        score = float(val_metrics.get("accuracy", train_metrics["accuracy"]))
        if score >= best_val:
            best_val = score
            save_same_wsi_checkpoint(
                output_dir / "best.pt",
                model=model,
                extra={"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics},
            )


def _run_epoch(model, loader, *, optimizer, device: str) -> dict[str, float]:
    if loader is None:
        return {}
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            image_a = batch["image_a"].to(device, non_blocking=True)
            image_b = batch["image_b"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(image_a, image_b)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().item()) * labels.numel()
            preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
            total_correct += int((preds == labels).sum().item())
            total += labels.numel()
    return {
        "loss": total_loss / max(1, total),
        "accuracy": total_correct / max(1, total),
        "pairs": float(total),
    }


def _samples_from_cross_metadata(metadata_path: str | Path) -> list[dict]:
    payload = json.loads(Path(metadata_path).read_text(encoding="utf8"))
    records = payload["pairs"] if isinstance(payload, dict) else payload
    by_key: dict[tuple[str, str], dict] = {}
    for record in records:
        dataset = str(record["dataset"])
        case_id = str(record["case_id"])
        for prefix in ("target", "reference"):
            sample_id_key = "sample_id" if prefix == "target" else "reference_sample_id"
            image_key = f"{prefix}_image"
            tissue_key = f"{prefix}_tissue_mask"
            key = (dataset, str(record[sample_id_key]))
            tissue_hist = _tissue_histogram(record[tissue_key])
            by_key[key] = {
                "dataset": dataset,
                "sample_id": str(record[sample_id_key]),
                "sample_key": f"{dataset}::{record[sample_id_key]}",
                "case_key": f"{dataset}::{case_id}",
                "image_path": str(record[image_key]),
                "tissue_mask_path": str(record[tissue_key]),
                "tissue_hist": tissue_hist,
            }
        if len(by_key) > 0 and len(by_key) % 5000 == 0:
            print(f"[same-wsi] summarized {len(by_key)} unique patches", file=sys.stderr, flush=True)
    return list(by_key.values())


def _build_hard_negative_index(
    samples: list[dict],
    *,
    pool_size: int,
    candidate_count: int,
    rng: random.Random,
) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    by_case: dict[str, list[dict]] = {}
    for sample in samples:
        by_case.setdefault(sample["case_key"], []).append(sample)
    case_keys = list(by_case)
    for sample_index, sample in enumerate(samples, start=1):
        sampled_candidates = _draw_different_case_candidates(
            sample,
            by_case=by_case,
            case_keys=case_keys,
            count=candidate_count,
            rng=rng,
        )
        candidates = [
            (
                _tissue_similarity(sample["tissue_hist"], candidate["tissue_hist"]),
                candidate,
            )
            for candidate in sampled_candidates
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        index[sample["sample_key"]] = [candidate for _, candidate in candidates[:pool_size]]
        if sample_index % 5000 == 0:
            print(
                f"[same-wsi] built hard-negative pools for {sample_index}/{len(samples)} patches",
                file=sys.stderr,
                flush=True,
            )
    return index


def _draw_different_case_candidates(
    sample: dict,
    *,
    by_case: dict[str, list[dict]],
    case_keys: list[str],
    count: int,
    rng: random.Random,
) -> list[dict]:
    candidates: list[dict] = []
    if len(case_keys) <= 1:
        return candidates
    attempts = 0
    max_attempts = max(count * 10, 100)
    seen: set[str] = set()
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        case_key = rng.choice(case_keys)
        if case_key == sample["case_key"]:
            continue
        candidate = rng.choice(by_case[case_key])
        sample_key = str(candidate["sample_key"])
        if sample_key in seen:
            continue
        seen.add(sample_key)
        candidates.append(candidate)
    return candidates


def _tissue_histogram(mask_path: str | Path) -> np.ndarray:
    mask = load_mask_array(mask_path).astype(np.int64, copy=False)
    labels, counts = np.unique(mask, return_counts=True)
    hist = np.zeros(max(int(labels.max()) + 1, 1), dtype=np.float32)
    for label, count in zip(labels, counts):
        label = int(label)
        if label <= 0:
            continue
        hist[label] = float(count)
    total = float(hist.sum())
    if total > 0.0:
        hist /= total
    return hist


def _tissue_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    width = max(hist_a.shape[0], hist_b.shape[0])
    a = np.zeros(width, dtype=np.float32)
    b = np.zeros(width, dtype=np.float32)
    a[: hist_a.shape[0]] = hist_a
    b[: hist_b.shape[0]] = hist_b
    intersection = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    overlap = intersection / union if union > 0.0 else 0.0
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return 0.5 * overlap + 0.5 * cosine


def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


if __name__ == "__main__":
    main()
