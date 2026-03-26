#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    REPO_ROOT
    / "nnUNet-master"
    / "DATASET"
    / "nnUNet_raw"
    / "Dataset666_FiberSegmentation"
)
PROJECT_ROOT = REPO_ROOT

DEFAULT_PRETRAIN_PRED_DIR = DATASET_ROOT / "labelsTr"
DEFAULT_PRETRAIN_REFINE_DIR = DATASET_ROOT / "labelsTr"

DEFAULT_FINETUNE_PRED_DIR = DATASET_ROOT / "finetune_prediction"
DEFAULT_FINETUNE_REFINE_DIR = DATASET_ROOT / "finetune_prediction_manual_refine"

DEFAULT_TEST_PRED_DIR = DATASET_ROOT / "test_prediction"

DEFAULT_SPLIT_JSON = (
    REPO_ROOT
    / "nnUNet-master"
    / "DATASET"
    / "nnUNet_preprocessed"
    / "Dataset666_FiberSegmentation"
    / "splits_final.json"
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_case_stem(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return Path(filename).stem


def collect_nii_files(folder: Path):
    nii_files = sorted([p for p in folder.iterdir() if p.name.endswith(".nii.gz")])
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {folder}")
    return nii_files


def load_nii_as_2d_array(nii_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(nii_path))
    arr = sitk.GetArrayFromImage(img)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"File {nii_path} is not 2D after squeeze. Final shape: {arr.shape}"
        )

    return arr.astype(np.uint8)


def save_rgb_png(arr_rgb: np.ndarray, out_path: Path):
    Image.fromarray(arr_rgb).save(str(out_path))


def save_uint8_png(arr: np.ndarray, out_path: Path):
    Image.fromarray(arr.astype(np.uint8)).save(str(out_path))


def save_uint16_png(arr: np.ndarray, out_path: Path):
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    img = Image.fromarray(arr, mode="I;16")
    img.save(str(out_path))


def semantic_to_rgb(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[arr == 1] = (255, 0, 0)
    rgb[arr == 2] = (0, 255, 0)
    return rgb


def make_red_only_rgb(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[arr == 1] = (255, 0, 0)
    return rgb


def make_weft_semantic_mask(arr: np.ndarray) -> np.ndarray:
    return np.where(arr == 2, 2, 0).astype(np.uint8)


def make_instance_mask_from_label1(arr: np.ndarray) -> np.ndarray:
    binary = (arr == 1).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num_features = ndimage.label(binary, structure=structure)

    if num_features > 65535:
        raise ValueError(
            f"Too many connected instances ({num_features}), exceeds uint16 range."
        )

    return labeled.astype(np.uint16)


def get_cellpose_dataset_dir(project_root: Path, stage: str) -> Path:
    if stage == "pretrain":
        out_dir = project_root / "cellpose-main" / "dataset" / "pretrain_only_warp"
    elif stage == "finetune":
        out_dir = project_root / "cellpose-main" / "dataset" / "finetune_only_warp"
    elif stage == "test":
        out_dir = project_root / "cellpose-main" / "dataset" / "test_only_warp"
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    ensure_dir(out_dir)
    return out_dir


def get_reconstruction_dirs(project_root: Path):
    nnunet_prediction_dir = project_root / "reconstruction" / "nnunet_prediction"
    weft_mask_dir = project_root / "reconstruction" / "weft_semantic_seg_mask"

    ensure_dir(nnunet_prediction_dir)
    ensure_dir(weft_mask_dir)

    return nnunet_prediction_dir, weft_mask_dir


def load_nnunet_split(split_json_path: Path, split_index: int = 0):
    if not split_json_path.exists():
        raise FileNotFoundError(f"Split json not found: {split_json_path}")

    with open(split_json_path, "r", encoding="utf-8") as handle:
        splits = json.load(handle)

    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError(f"Invalid split file format: {split_json_path}")

    if split_index >= len(splits):
        raise IndexError(
            f"Requested split_index={split_index}, but only {len(splits)} splits found."
        )

    split = splits[split_index]
    train_ids = set(split.get("train", []))
    val_ids = set(split.get("val", []))

    if not train_ids and not val_ids:
        raise ValueError(
            f"Split file {split_json_path} contains empty train/val lists at split_index={split_index}"
        )

    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"Found duplicated case IDs in both train and val: {sorted(overlap)}"
        )

    return train_ids, val_ids


def split_pretrain_dataset_by_json(output_dir: Path, split_json_path: Path, split_index: int = 0):
    train_ids, val_ids = load_nnunet_split(split_json_path, split_index=split_index)

    train_dir = output_dir / "train_image"
    val_dir = output_dir / "val_image"

    reset_dir(train_dir)
    reset_dir(val_dir)

    image_files = sorted(
        [p for p in output_dir.glob("*.png") if not p.name.endswith("_masks.png")]
    )
    mask_files = sorted(output_dir.glob("*_masks.png"))

    image_map = {p.stem: p for p in image_files}
    mask_map = {p.stem[:-6]: p for p in mask_files}

    all_cases_in_output = set(image_map.keys()) | set(mask_map.keys())

    missing_mask = sorted(set(image_map.keys()) - set(mask_map.keys()))
    missing_img = sorted(set(mask_map.keys()) - set(image_map.keys()))

    if missing_mask:
        raise RuntimeError(
            "Missing masks before splitting:\n"
            + "\n".join(
                [f"  - {case_id}.png -> expected {case_id}_masks.png" for case_id in missing_mask]
            )
        )
    if missing_img:
        raise RuntimeError(
            "Missing images before splitting:\n"
            + "\n".join(
                [f"  - {case_id}_masks.png -> expected {case_id}.png" for case_id in missing_img]
            )
        )

    assigned_cases = train_ids | val_ids
    unassigned_cases = sorted(all_cases_in_output - assigned_cases)
    missing_cases_from_output = sorted(assigned_cases - all_cases_in_output)

    if missing_cases_from_output:
        raise RuntimeError(
            "The following case IDs are defined in splits_final.json but not found in pretrain output_dir:\n"
            + "\n".join([f"  - {case_id}" for case_id in missing_cases_from_output])
        )

    if unassigned_cases:
        raise RuntimeError(
            "The following generated case IDs are not covered by train/val in splits_final.json:\n"
            + "\n".join([f"  - {case_id}" for case_id in unassigned_cases])
        )

    for case_id in sorted(train_ids):
        shutil.move(str(image_map[case_id]), str(train_dir / image_map[case_id].name))
        shutil.move(str(mask_map[case_id]), str(train_dir / mask_map[case_id].name))

    for case_id in sorted(val_ids):
        shutil.move(str(image_map[case_id]), str(val_dir / image_map[case_id].name))
        shutil.move(str(mask_map[case_id]), str(val_dir / mask_map[case_id].name))

    return train_dir, val_dir, sorted(train_ids), sorted(val_ids)


def generate_training_images_from_prediction(pred_input_dir: Path, output_dir: Path):
    nii_files = collect_nii_files(pred_input_dir)

    for nii_path in nii_files:
        stem = get_case_stem(nii_path.name)
        arr = load_nii_as_2d_array(nii_path)
        red_only = make_red_only_rgb(arr)
        save_rgb_png(red_only, output_dir / f"{stem}.png")


def generate_masks_from_refine(refine_input_dir: Path, output_dir: Path):
    nii_files = collect_nii_files(refine_input_dir)

    for nii_path in nii_files:
        stem = get_case_stem(nii_path.name)
        arr = load_nii_as_2d_array(nii_path)
        instance_mask = make_instance_mask_from_label1(arr)
        save_uint16_png(instance_mask, output_dir / f"{stem}_masks.png")


def generate_reconstruction_outputs(pred_input_dir: Path, refine_input_dir: Path, project_root: Path):
    nnunet_prediction_dir, weft_mask_dir = get_reconstruction_dirs(project_root)

    pred_files = collect_nii_files(pred_input_dir)
    for nii_path in pred_files:
        stem = get_case_stem(nii_path.name)
        arr = load_nii_as_2d_array(nii_path)
        rgb = semantic_to_rgb(arr)
        save_rgb_png(rgb, nnunet_prediction_dir / f"{stem}.png")

    refine_files = collect_nii_files(refine_input_dir)
    for nii_path in refine_files:
        stem = get_case_stem(nii_path.name)
        arr = load_nii_as_2d_array(nii_path)
        weft_mask = make_weft_semantic_mask(arr)
        save_uint8_png(weft_mask, weft_mask_dir / f"{stem}.png")


def validate_image_mask_pairs(output_dir: Path):
    all_pngs = sorted(output_dir.glob("*.png"))
    image_files = [p for p in all_pngs if not p.name.endswith("_masks.png")]
    mask_files = [p for p in all_pngs if p.name.endswith("_masks.png")]

    image_map = {p.stem: p for p in image_files}
    mask_map = {p.stem[:-6]: p for p in mask_files}

    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))

    errors = []

    if missing_masks:
        errors.append(
            "Missing masks for the following images:\n"
            + "\n".join(
                [f"  - {name}.png -> expected {name}_masks.png" for name in missing_masks]
            )
        )

    if missing_images:
        errors.append(
            "Missing images for the following masks:\n"
            + "\n".join(
                [f"  - {name}_masks.png -> expected {name}.png" for name in missing_images]
            )
        )

    common_names = sorted(set(image_map.keys()) & set(mask_map.keys()))

    for name in common_names:
        img_path = image_map[name]
        mask_path = mask_map[name]

        with Image.open(img_path) as img:
            img_size = img.size

        with Image.open(mask_path) as mask:
            mask_size = mask.size

        if img_size != mask_size:
            errors.append(
                f"Size mismatch:\n"
                f"  - image: {img_path.name}, size={img_size}\n"
                f"  - mask : {mask_path.name}, size={mask_size}"
            )

    if errors:
        raise RuntimeError("\n\n".join(errors))

    print(f"[OK] Image/mask validation passed in {output_dir}. Pairs: {len(common_names)}")


def run_pretrain(
    project_root: Path,
    pred_input_dir: Path,
    refine_input_dir: Path,
    split_json_path: Path,
    split_index: int = 0,
):
    print("[Stage] PRETRAIN")

    output_dir = get_cellpose_dataset_dir(project_root, "pretrain")
    ensure_dir(output_dir)

    for png_path in output_dir.glob("*.png"):
        png_path.unlink()

    generate_training_images_from_prediction(pred_input_dir, output_dir)
    generate_masks_from_refine(refine_input_dir, output_dir)
    validate_image_mask_pairs(output_dir)

    train_dir, val_dir, train_ids, val_ids = split_pretrain_dataset_by_json(
        output_dir=output_dir,
        split_json_path=split_json_path,
        split_index=split_index,
    )

    validate_image_mask_pairs(train_dir)
    validate_image_mask_pairs(val_dir)

    print("[Done] Pretraining dataset preparation completed.")
    print(f"  - root: {output_dir}")
    print(f"  - train_image: {train_dir}  | cases: {len(train_ids)}")
    print(f"  - val_image  : {val_dir}  | cases: {len(val_ids)}")


def run_finetune(project_root: Path, pred_input_dir: Path, refine_input_dir: Path):
    print("[Stage] FINETUNE")

    output_dir = get_cellpose_dataset_dir(project_root, "finetune")

    generate_training_images_from_prediction(pred_input_dir, output_dir)
    generate_masks_from_refine(refine_input_dir, output_dir)
    validate_image_mask_pairs(output_dir)

    print("[Done] Fine-tuning dataset preparation completed.")
    print(f"  - {output_dir}")


def run_test(project_root: Path, pred_input_dir: Path):
    print("[Stage] TEST")

    output_dir = get_cellpose_dataset_dir(project_root, "test")

    generate_training_images_from_prediction(pred_input_dir, output_dir)
    generate_reconstruction_outputs(
        pred_input_dir=pred_input_dir,
        refine_input_dir=pred_input_dir,
        project_root=project_root,
    )

    print("[Done] Test dataset preparation completed.")
    print(f"  - {output_dir}")
    print(f"  - {project_root / 'reconstruction' / 'nnunet_prediction'}")
    print(f"  - {project_root / 'reconstruction' / 'weft_semantic_seg_mask'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare Cellpose-compatible datasets for pretraining, fine-tuning, and testing."
    )

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pretrain", "finetune", "test"],
        help="Pipeline stage to run.",
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root containing cellpose-main/ and reconstruction/.",
    )

    parser.add_argument(
        "--pred_input_dir",
        type=str,
        default=None,
        help="Prediction input directory. If omitted, stage-specific default will be used.",
    )

    parser.add_argument(
        "--refine_input_dir",
        type=str,
        default=None,
        help="Refinement label directory. If omitted, stage-specific default will be used.",
    )

    parser.add_argument(
        "--split_json",
        type=str,
        default=str(DEFAULT_SPLIT_JSON),
        help="nnUNet splits_final.json path, only used in pretrain stage.",
    )

    parser.add_argument(
        "--split_index",
        type=int,
        default=0,
        help="Which split entry to use in splits_final.json. Default: 0",
    )

    return parser


def resolve_defaults(stage: str, pred_input_dir: str | None, refine_input_dir: str | None):
    if stage == "pretrain":
        pred_path = Path(pred_input_dir) if pred_input_dir else DEFAULT_PRETRAIN_PRED_DIR
        refine_path = Path(refine_input_dir) if refine_input_dir else DEFAULT_PRETRAIN_REFINE_DIR
    elif stage == "finetune":
        pred_path = Path(pred_input_dir) if pred_input_dir else DEFAULT_FINETUNE_PRED_DIR
        refine_path = Path(refine_input_dir) if refine_input_dir else DEFAULT_FINETUNE_REFINE_DIR
    elif stage == "test":
        pred_path = Path(pred_input_dir) if pred_input_dir else DEFAULT_TEST_PRED_DIR
        refine_path = None
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    return pred_path, refine_path


def main():
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    pred_input_dir, refine_input_dir = resolve_defaults(
        args.stage, args.pred_input_dir, args.refine_input_dir
    )

    pred_input_dir = pred_input_dir.resolve()
    if refine_input_dir is not None:
        refine_input_dir = refine_input_dir.resolve()

    if not pred_input_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_input_dir}")

    if args.stage in {"pretrain", "finetune"} and refine_input_dir is None:
        raise ValueError(f"Refinement directory is required for stage {args.stage}")

    if refine_input_dir is not None and not refine_input_dir.exists():
        raise FileNotFoundError(f"Refinement directory does not exist: {refine_input_dir}")

    if args.stage == "pretrain":
        run_pretrain(
            project_root=project_root,
            pred_input_dir=pred_input_dir,
            refine_input_dir=refine_input_dir,
            split_json_path=Path(args.split_json).resolve(),
            split_index=args.split_index,
        )
    elif args.stage == "finetune":
        run_finetune(
            project_root=project_root,
            pred_input_dir=pred_input_dir,
            refine_input_dir=refine_input_dir,
        )
    elif args.stage == "test":
        run_test(
            project_root=project_root,
            pred_input_dir=pred_input_dir,
        )


if __name__ == "__main__":
    main()
