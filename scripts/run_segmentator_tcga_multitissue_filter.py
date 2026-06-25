from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable
from urllib import parse, request

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentator.model import BaselineSegmenter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

DEFAULT_LABELS = (
    "background",
    "tumor",
    "stroma",
    "necrosis",
    "immune_infiltrate",
    "normal_epithelium",
    "blood_vessel",
    "other_tissue",
)

TCGA_PROJECTS: dict[str, dict[str, str]] = {
    "ACC": {"organ": "adrenal gland", "disease": "adrenocortical carcinoma"},
    "BLCA": {"organ": "bladder", "disease": "bladder urothelial carcinoma"},
    "BRCA": {"organ": "breast", "disease": "breast carcinoma"},
    "CESC": {"organ": "cervix", "disease": "cervical squamous cell carcinoma and endocervical adenocarcinoma"},
    "CHOL": {"organ": "bile duct", "disease": "cholangiocarcinoma"},
    "COAD": {"organ": "colon", "disease": "colon adenocarcinoma"},
    "DLBC": {"organ": "lymphoid tissue", "disease": "diffuse large B-cell lymphoma"},
    "ESCA": {"organ": "esophagus", "disease": "esophageal carcinoma"},
    "GBM": {"organ": "brain", "disease": "glioblastoma"},
    "HNSC": {"organ": "head and neck", "disease": "head and neck squamous cell carcinoma"},
    "KICH": {"organ": "kidney", "disease": "kidney chromophobe"},
    "KIRC": {"organ": "kidney", "disease": "kidney renal clear cell carcinoma"},
    "KIRP": {"organ": "kidney", "disease": "kidney renal papillary cell carcinoma"},
    "LAML": {"organ": "bone marrow and blood", "disease": "acute myeloid leukemia"},
    "LGG": {"organ": "brain", "disease": "lower grade glioma"},
    "LIHC": {"organ": "liver", "disease": "liver hepatocellular carcinoma"},
    "LUAD": {"organ": "lung", "disease": "lung adenocarcinoma"},
    "LUSC": {"organ": "lung", "disease": "lung squamous cell carcinoma"},
    "MESO": {"organ": "pleura", "disease": "mesothelioma"},
    "OV": {"organ": "ovary", "disease": "ovarian serous cystadenocarcinoma"},
    "PAAD": {"organ": "pancreas", "disease": "pancreatic adenocarcinoma"},
    "PCPG": {"organ": "adrenal gland and paraganglia", "disease": "pheochromocytoma and paraganglioma"},
    "PRAD": {"organ": "prostate", "disease": "prostate adenocarcinoma"},
    "READ": {"organ": "rectum", "disease": "rectum adenocarcinoma"},
    "SARC": {"organ": "soft tissue", "disease": "sarcoma"},
    "SKCM": {"organ": "skin", "disease": "skin cutaneous melanoma"},
    "STAD": {"organ": "stomach", "disease": "stomach adenocarcinoma"},
    "TGCT": {"organ": "testis", "disease": "testicular germ cell tumor"},
    "THCA": {"organ": "thyroid", "disease": "thyroid carcinoma"},
    "THYM": {"organ": "thymus", "disease": "thymoma"},
    "UCEC": {"organ": "uterus", "disease": "uterine corpus endometrial carcinoma"},
    "UCS": {"organ": "uterus", "disease": "uterine carcinosarcoma"},
    "UVM": {"organ": "eye", "disease": "uveal melanoma"},
}

TCGA_BARCODE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9]{2,4})*)", re.IGNORECASE)


@dataclass(frozen=True)
class TcgaBarcode:
    slide_barcode: str
    case_barcode: str


@dataclass(frozen=True)
class OrganInfo:
    organ: str
    project_id: str
    disease: str
    source: str
    prompt: str


def parse_tcga_barcode(text: str) -> TcgaBarcode:
    match = TCGA_BARCODE_RE.search(text.upper())
    if not match:
        return TcgaBarcode(slide_barcode="", case_barcode="")
    slide = match.group(1).upper()
    parts = slide.split("-")
    case = "-".join(parts[:3]) if len(parts) >= 3 else ""
    return TcgaBarcode(slide_barcode=slide, case_barcode=case)


def normalize_project_id(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().upper()
    if text.startswith("TCGA-"):
        text = text[5:]
    return text if text in TCGA_PROJECTS else ""


def project_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        tokens = [token.upper() for token in re.split(r"[^A-Za-z0-9]+", part) if token]
        for idx, token in enumerate(tokens):
            project = normalize_project_id(token)
            if project:
                return project
            if token == "TCGA" and idx + 1 < len(tokens):
                project = normalize_project_id(tokens[idx + 1])
                if project:
                    return project
    return ""


def build_segmentator_prompt(organ: str, project_id: str = "", disease: str = "") -> str:
    if not disease and project_id in TCGA_PROJECTS:
        disease = TCGA_PROJECTS[project_id]["disease"]
    site_text = disease or (f"{organ} cancer" if organ and organ != "unknown" else "cancer")
    return (
        f"H&E stained {site_text} histopathology with tumor, stroma, necrosis, "
        "immune infiltrate, normal epithelium, blood vessel, and other tissue."
    )


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB image array, got shape {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def resize_rgb(image: Image.Image, image_size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((image_size, image_size), resample=resampling)


def organ_info_from_project(project_id: str, source: str) -> OrganInfo | None:
    project_id = normalize_project_id(project_id)
    if not project_id:
        return None
    metadata = TCGA_PROJECTS[project_id]
    return OrganInfo(
        organ=metadata["organ"],
        project_id=project_id,
        disease=metadata["disease"],
        source=source,
        prompt=build_segmentator_prompt(metadata["organ"], project_id, metadata["disease"]),
    )


def _first_present(row: dict[str, str], names: Iterable[str]) -> str:
    lower_to_key = {key.lower(): key for key in row}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None:
            value = row.get(key, "")
            if value:
                return value.strip()
    return ""


def _metadata_organ_info(row: dict[str, str]) -> OrganInfo | None:
    project = normalize_project_id(
        _first_present(row, ("project_id", "project", "tcga_project", "study", "cancer_type", "cohort"))
    )
    organ = _first_present(
        row,
        (
            "organ",
            "primary_site",
            "site",
            "tissue",
            "tissue_or_organ_of_origin",
            "primary_diagnosis_site",
        ),
    )
    disease = _first_present(row, ("disease", "disease_type", "diagnosis", "primary_diagnosis"))
    prompt = _first_present(row, ("segmentator_prompt", "prompt"))
    if project and not organ:
        organ = TCGA_PROJECTS[project]["organ"]
    if project and not disease:
        disease = TCGA_PROJECTS[project]["disease"]
    if not organ and not project:
        return None
    organ = organ.lower() if organ else "unknown"
    return OrganInfo(
        organ=organ,
        project_id=project,
        disease=disease,
        source="metadata_csv",
        prompt=prompt or build_segmentator_prompt(organ, project, disease),
    )


def load_metadata_csv(path: Path | None) -> dict[str, OrganInfo]:
    if path is None:
        return {}
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    mapping: dict[str, OrganInfo] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            info = _metadata_organ_info(row)
            if info is None:
                continue
            values = [
                _first_present(
                    row,
                    (
                        "file_name",
                        "filename",
                        "image",
                        "image_path",
                        "path",
                        "slide_barcode",
                        "slide_submitter_id",
                        "case_barcode",
                        "case_submitter_id",
                        "submitter_id",
                        "bcr_patient_barcode",
                        "barcode",
                    ),
                )
            ]
            for value in list(values):
                parsed = parse_tcga_barcode(value)
                values.extend([parsed.slide_barcode, parsed.case_barcode])
                values.extend([Path(value).name, Path(value).stem])
            for value in values:
                key = _metadata_key(value)
                if key:
                    mapping[key] = info
    return mapping


def _metadata_key(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parsed = parse_tcga_barcode(text)
    if parsed.slide_barcode:
        return parsed.slide_barcode
    return Path(text).name.upper()


def resolve_organ_info(
    path: Path,
    metadata: dict[str, OrganInfo],
    gdc_cases: dict[str, OrganInfo] | None = None,
) -> OrganInfo:
    barcode = parse_tcga_barcode(path.name)
    lookup_keys = (
        path.name.upper(),
        path.stem.upper(),
        barcode.slide_barcode,
        barcode.case_barcode,
    )
    for key in lookup_keys:
        info = metadata.get(_metadata_key(key))
        if info is not None:
            return info

    project = project_from_path(path)
    info = organ_info_from_project(project, source="path_project")
    if info is not None:
        return info

    if gdc_cases and barcode.case_barcode:
        info = gdc_cases.get(barcode.case_barcode)
        if info is not None:
            return info

    return OrganInfo(
        organ="unknown",
        project_id="",
        disease="",
        source="unresolved",
        prompt=build_segmentator_prompt("unknown"),
    )


def load_gdc_cache(path: Path | None) -> dict[str, OrganInfo]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, OrganInfo] = {}
    for case_barcode, item in payload.items():
        if isinstance(item, dict):
            out[case_barcode.upper()] = OrganInfo(
                organ=str(item.get("organ", "unknown")),
                project_id=str(item.get("project_id", "")),
                disease=str(item.get("disease", "")),
                source=str(item.get("source", "gdc_cache")),
                prompt=str(item.get("prompt", "")) or build_segmentator_prompt(
                    str(item.get("organ", "unknown")),
                    str(item.get("project_id", "")),
                    str(item.get("disease", "")),
                ),
            )
    return out


def save_gdc_cache(path: Path | None, cache: dict[str, OrganInfo]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {
            "organ": value.organ,
            "project_id": value.project_id,
            "disease": value.disease,
            "source": value.source,
            "prompt": value.prompt,
        }
        for key, value in sorted(cache.items())
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_gdc_case_info(case_barcodes: list[str], batch_size: int = 100, timeout: int = 30) -> dict[str, OrganInfo]:
    resolved: dict[str, OrganInfo] = {}
    endpoint = "https://api.gdc.cancer.gov/cases"
    for start in tqdm(range(0, len(case_barcodes), batch_size), desc="gdc metadata", dynamic_ncols=True):
        batch = case_barcodes[start : start + batch_size]
        filters = {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": batch,
            },
        }
        body = parse.urlencode(
            {
                "filters": json.dumps(filters),
                "fields": "submitter_id,project.project_id,project.primary_site,disease_type",
                "format": "JSON",
                "size": str(len(batch)),
            }
        ).encode("utf-8")
        req = request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        for hit in payload.get("data", {}).get("hits", []):
            case = str(hit.get("submitter_id", "")).upper()
            project = normalize_project_id((hit.get("project") or {}).get("project_id", ""))
            organ = str((hit.get("project") or {}).get("primary_site", "")).lower()
            disease = str(hit.get("disease_type", ""))
            if project:
                project_info = TCGA_PROJECTS[project]
                organ = organ or project_info["organ"]
                disease = disease or project_info["disease"]
            if case:
                resolved[case] = OrganInfo(
                    organ=organ or "unknown",
                    project_id=project,
                    disease=disease,
                    source="gdc_api",
                    prompt=build_segmentator_prompt(organ or "unknown", project, disease),
                )
    return resolved


class PatchImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int | None) -> None:
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                original_width, original_height = rgb.size
                if self.image_size:
                    rgb = resize_rgb(rgb, self.image_size)
                tensor = pil_to_normalized_tensor(rgb)
        except Exception as exc:
            return {
                "ok": False,
                "path": str(path),
                "filename": path.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "ok": True,
            "image": tensor,
            "path": str(path),
            "filename": path.name,
            "original_width": original_width,
            "original_height": original_height,
            "inference_width": int(tensor.shape[-1]),
            "inference_height": int(tensor.shape[-2]),
        }


def collate_patch_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    ok_items = [item for item in items if item.get("ok")]
    error_items = [item for item in items if not item.get("ok")]
    images = torch.stack([item["image"] for item in ok_items], dim=0) if ok_items else None
    return {
        "images": images,
        "items": ok_items,
        "errors": error_items,
    }


def find_image_paths(input_dir: Path, recursive: bool, limit: int | None) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_segmentator_model(args: argparse.Namespace, device: torch.device) -> BaselineSegmenter:
    model = BaselineSegmenter(
        num_classes=args.num_classes,
        freeze_encoder=True,
        local_repo=args.uni2h_repo,
        decoder=args.decoder,
        mask2former_queries=args.mask2former_queries,
        mask2former_ignore_index=args.mask2former_ignore_index,
    )
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported checkpoint format: {args.checkpoint}")
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def summarize_prediction_counts(
    counts: np.ndarray,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    include_background: bool = False,
    min_class_fraction: float = 0.01,
    min_class_pixels: int = 256,
    min_foreground_fraction: float = 0.05,
) -> dict[str, Any]:
    total_pixels = int(counts.sum())
    class_ids = range(len(labels)) if include_background else range(1, len(labels))
    denominator = int(sum(int(counts[idx]) for idx in class_ids))
    foreground_pixels = int(sum(int(counts[idx]) for idx in range(1, len(labels))))
    foreground_fraction = float(foreground_pixels / total_pixels) if total_pixels else 0.0
    fractions = {
        labels[idx]: (float(int(counts[idx]) / denominator) if denominator else 0.0)
        for idx in class_ids
    }
    ranked = sorted(
        (
            {
                "class_id": idx,
                "label": labels[idx],
                "pixels": int(counts[idx]),
                "fraction": fractions[labels[idx]],
            }
            for idx in class_ids
        ),
        key=lambda item: item["pixels"],
        reverse=True,
    )
    qualifying = [
        item
        for item in ranked
        if item["pixels"] >= min_class_pixels and item["fraction"] >= min_class_fraction
    ]
    selected = len(qualifying) >= 2 and foreground_fraction >= min_foreground_fraction
    return {
        "selected": bool(selected),
        "total_pixels": total_pixels,
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": foreground_fraction,
        "qualifying_tissue_count": len(qualifying),
        "top_classes": ranked[:4],
        "class_pixel_counts": {labels[idx]: int(counts[idx]) for idx in range(len(labels))},
        "class_fractions_of_considered_pixels": fractions,
    }


def prediction_row(
    item: dict[str, Any],
    mask: torch.Tensor,
    labels: tuple[str, ...],
    organ_info: OrganInfo,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = Path(item["path"])
    counts = torch.bincount(mask.reshape(-1).cpu(), minlength=len(labels)).numpy()
    summary = summarize_prediction_counts(
        counts=counts,
        labels=labels,
        include_background=args.include_background,
        min_class_fraction=args.min_class_fraction,
        min_class_pixels=args.min_class_pixels,
        min_foreground_fraction=args.min_foreground_fraction,
    )
    barcode = parse_tcga_barcode(path.name)
    top = summary["top_classes"]
    top1 = top[0] if len(top) > 0 else {"label": "", "fraction": 0.0, "pixels": 0}
    top2 = top[1] if len(top) > 1 else {"label": "", "fraction": 0.0, "pixels": 0}
    return {
        "selected": int(summary["selected"]),
        "filename": path.name,
        "image_path": str(path),
        "organ": organ_info.organ,
        "project_id": organ_info.project_id,
        "organ_source": organ_info.source,
        "tcga_slide_barcode": barcode.slide_barcode,
        "tcga_case_barcode": barcode.case_barcode,
        "segmentator_prompt": organ_info.prompt,
        "foreground_fraction": f"{summary['foreground_fraction']:.6f}",
        "qualifying_tissue_count": summary["qualifying_tissue_count"],
        "top1_label": top1["label"],
        "top1_fraction": f"{top1['fraction']:.6f}",
        "top1_pixels": top1["pixels"],
        "top2_label": top2["label"],
        "top2_fraction": f"{top2['fraction']:.6f}",
        "top2_pixels": top2["pixels"],
        "total_pixels": summary["total_pixels"],
        "foreground_pixels": summary["foreground_pixels"],
        "class_pixel_counts_json": json.dumps(summary["class_pixel_counts"], separators=(",", ":")),
        "class_fractions_json": json.dumps(
            {
                key: round(value, 6)
                for key, value in summary["class_fractions_of_considered_pixels"].items()
            },
            separators=(",", ":"),
        ),
        "original_width": item.get("original_width", ""),
        "original_height": item.get("original_height", ""),
        "inference_width": item.get("inference_width", ""),
        "inference_height": item.get("inference_height", ""),
    }


def save_mask(mask: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    array = mask.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(array, mode="L").save(out_path)


def should_save_mask(mode: str, selected: bool) -> bool:
    return mode == "all" or (mode == "selected" and selected)


def csv_fieldnames() -> list[str]:
    return [
        "selected",
        "filename",
        "image_path",
        "organ",
        "project_id",
        "organ_source",
        "tcga_slide_barcode",
        "tcga_case_barcode",
        "segmentator_prompt",
        "foreground_fraction",
        "qualifying_tissue_count",
        "top1_label",
        "top1_fraction",
        "top1_pixels",
        "top2_label",
        "top2_fraction",
        "top2_pixels",
        "total_pixels",
        "foreground_pixels",
        "class_pixel_counts_json",
        "class_fractions_json",
        "original_width",
        "original_height",
        "inference_width",
        "inference_height",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run segmentator over TCGA patches and write a CSV of patches with at least two "
            "non-trivial predicted tissue classes."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--save-mask-mode", choices=["none", "selected", "all"], default="selected")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Optional TCGA metadata CSV/TSV with case/project/organ columns.")
    parser.add_argument("--allow-gdc-lookup", action="store_true", help="Query GDC by TCGA case barcode when metadata/path cannot resolve organ.")
    parser.add_argument("--gdc-cache", type=Path, default=None, help="JSON cache for --allow-gdc-lookup results.")
    parser.add_argument("--gdc-batch-size", type=int, default=100)
    parser.add_argument("--gdc-timeout", type=int, default=30)
    parser.add_argument("--uni2h-repo", default="UNI-2h")
    parser.add_argument("--decoder", choices=["upernet", "mask2former"], default="mask2former")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--mask2former-queries", type=int, default=100)
    parser.add_argument("--mask2former-ignore-index", type=int, default=255)
    parser.add_argument("--image-size", type=int, default=512, help="Resize square side for inference. Use 0 to keep native size.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-class-fraction", type=float, default=0.01, help="Minimum fraction of foreground tissue pixels for each counted tissue class.")
    parser.add_argument("--min-class-pixels", type=int, default=256)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.05)
    parser.add_argument("--include-background", action="store_true", help="Count background as a class for the >=2-class rule. Default counts only tissue labels 1..7.")
    parser.add_argument("--write-all", action="store_true", help="Write all patches to CSV with selected=0/1 instead of only selected patches/errors.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference.")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_classes != len(DEFAULT_LABELS):
        raise ValueError(f"this script expects {len(DEFAULT_LABELS)} classes, got {args.num_classes}")
    if args.save_mask_mode != "none" and args.mask_dir is None:
        args.mask_dir = args.output_csv.with_suffix("").parent / f"{args.output_csv.stem}_masks"

    image_paths = find_image_paths(args.input_dir, recursive=args.recursive, limit=args.limit)
    if not image_paths:
        raise FileNotFoundError(f"no image files found under {args.input_dir}")

    metadata = load_metadata_csv(args.metadata_csv)
    gdc_cache = load_gdc_cache(args.gdc_cache)
    if args.allow_gdc_lookup:
        unresolved_cases = []
        for path in image_paths:
            barcode = parse_tcga_barcode(path.name)
            if not barcode.case_barcode or barcode.case_barcode in gdc_cache:
                continue
            info_without_gdc = resolve_organ_info(path, metadata, gdc_cases={})
            if info_without_gdc.source == "unresolved":
                unresolved_cases.append(barcode.case_barcode)
        unresolved_cases = sorted(set(unresolved_cases))
        if unresolved_cases:
            gdc_cache.update(
                fetch_gdc_case_info(
                    unresolved_cases,
                    batch_size=args.gdc_batch_size,
                    timeout=args.gdc_timeout,
                )
            )
            save_gdc_cache(args.gdc_cache, gdc_cache)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_segmentator_model(args, device)
    dataset = PatchImageDataset(image_paths, image_size=args.image_size or None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_patch_batch,
        pin_memory=device.type == "cuda",
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_count = 0
    processed_count = 0
    error_count = 0
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames() + ["error"])
        writer.writeheader()
        with torch.no_grad():
            for batch in tqdm(loader, desc="segmentator", total=len(loader), dynamic_ncols=True):
                for error in batch["errors"]:
                    error_count += 1
                    if args.write_all:
                        writer.writerow(
                            {
                                "selected": 0,
                                "filename": error["filename"],
                                "image_path": error["path"],
                                "organ": "unknown",
                                "project_id": "",
                                "organ_source": "unreadable",
                                "error": error["error"],
                            }
                        )
                images = batch["images"]
                if images is None:
                    continue
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    outputs = model(images)
                preds = outputs["pred"].cpu()
                for item, mask in zip(batch["items"], preds):
                    processed_count += 1
                    organ_info = resolve_organ_info(Path(item["path"]), metadata, gdc_cache)
                    row = prediction_row(item, mask, DEFAULT_LABELS, organ_info, args)
                    selected = bool(int(row["selected"]))
                    if selected:
                        selected_count += 1
                    if args.write_all or selected:
                        row["error"] = ""
                        writer.writerow(row)
                    if args.mask_dir is not None and should_save_mask(args.save_mask_mode, selected):
                        save_mask(mask, args.mask_dir / f"{Path(item['path']).stem}.png")

    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_csv": str(args.output_csv),
                "mask_dir": str(args.mask_dir) if args.mask_dir else None,
                "images_found": len(image_paths),
                "processed": processed_count,
                "selected": selected_count,
                "errors": error_count,
                "min_class_fraction": args.min_class_fraction,
                "min_class_pixels": args.min_class_pixels,
                "min_foreground_fraction": args.min_foreground_fraction,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
