#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_ID = 666
DEFAULT_DATASET_NAME = "Dataset666_FiberSegmentation"
DEFAULT_CONFIGURATION = "2d"
DEFAULT_FOLD = "0"
DEFAULT_SPLIT_INDEX = 0
DEFAULT_TEST_DIAM_MEAN = 25
DEFAULT_NNUNET_TRAINER = "nnUNetTrainer"
DEFAULT_CELLPOSE_PRETRAIN_EPOCHS = 500
DEFAULT_CELLPOSE_FINETUNE_EPOCHS = 100
DEFAULT_CELLPOSE_LR = 0.001
DEFAULT_CONDA_ENV_ROOT = Path("/opt/conda/envs")
DEFAULT_SEMANTIC_ENV_NAME = "FabricFlow_semantic"
DEFAULT_INSTANCE_ENV_NAME = "FabricFlow_instance"


@dataclass(frozen=True)
class SegmentationPaths:
    repo_root: Path
    nnunet_root: Path
    cellpose_root: Path
    prepare_script: Path
    nnunet_dataset_root: Path
    nnunet_raw_root: Path
    nnunet_preprocessed_root: Path
    nnunet_results_root: Path
    dataset_root: Path
    images_tr: Path
    labels_tr: Path
    dataset_json: Path
    split_json: Path
    finetune_image: Path
    finetune_prediction: Path
    finetune_refine: Path
    test_image: Path
    test_prediction: Path
    pretrain_cellpose_root: Path
    pretrain_train_dir: Path
    pretrain_val_dir: Path
    pretrain_model_dir: Path
    finetune_cellpose_root: Path
    finetune_model_dir: Path
    test_cellpose_root: Path
    reconstruction_root: Path
    reconstruction_nnunet_prediction: Path
    reconstruction_weft_mask: Path
    reconstruction_warp_instance_mask: Path


@dataclass(frozen=True)
class RuntimeConfig:
    conda_env_root: Path
    semantic_env_name: str
    instance_env_name: str
    semantic_python: Path
    instance_python: Path


def build_paths(repo_root: Path, dataset_name: str) -> SegmentationPaths:
    nnunet_root = repo_root / "nnUNet-master"
    cellpose_root = repo_root / "cellpose-main"
    nnunet_dataset_root = nnunet_root / "DATASET"
    nnunet_raw_root = nnunet_dataset_root / "nnUNet_raw"
    nnunet_preprocessed_root = nnunet_dataset_root / "nnUNet_preprocessed"
    nnunet_results_root = nnunet_dataset_root / "nnUNet_trained_models"
    dataset_root = nnunet_raw_root / dataset_name
    pretrain_cellpose_root = cellpose_root / "dataset" / "pretrain_only_warp"
    pretrain_train_dir = pretrain_cellpose_root / "train_image"
    pretrain_val_dir = pretrain_cellpose_root / "val_image"
    finetune_cellpose_root = cellpose_root / "dataset" / "finetune_only_warp"
    test_cellpose_root = cellpose_root / "dataset" / "test_only_warp"
    reconstruction_root = repo_root / "reconstruction"
    return SegmentationPaths(
        repo_root=repo_root,
        nnunet_root=nnunet_root,
        cellpose_root=cellpose_root,
        prepare_script=repo_root / "segmentation" / "prepare_instance_seg_dataset.py",
        nnunet_dataset_root=nnunet_dataset_root,
        nnunet_raw_root=nnunet_raw_root,
        nnunet_preprocessed_root=nnunet_preprocessed_root,
        nnunet_results_root=nnunet_results_root,
        dataset_root=dataset_root,
        images_tr=dataset_root / "imagesTr",
        labels_tr=dataset_root / "labelsTr",
        dataset_json=dataset_root / "dataset.json",
        split_json=nnunet_preprocessed_root / dataset_name / "splits_final.json",
        finetune_image=dataset_root / "finetune_image",
        finetune_prediction=dataset_root / "finetune_prediction",
        finetune_refine=dataset_root / "finetune_prediction_manual_refine",
        test_image=dataset_root / "test_image",
        test_prediction=dataset_root / "test_prediction",
        pretrain_cellpose_root=pretrain_cellpose_root,
        pretrain_train_dir=pretrain_train_dir,
        pretrain_val_dir=pretrain_val_dir,
        pretrain_model_dir=pretrain_train_dir / "models",
        finetune_cellpose_root=finetune_cellpose_root,
        finetune_model_dir=finetune_cellpose_root / "models",
        test_cellpose_root=test_cellpose_root,
        reconstruction_root=reconstruction_root,
        reconstruction_nnunet_prediction=reconstruction_root / "nnunet_prediction",
        reconstruction_weft_mask=reconstruction_root / "weft_semantic_seg_mask",
        reconstruction_warp_instance_mask=reconstruction_root / "warp_instance_seg_mask",
    )


def log(message: str) -> None:
    print(message, flush=True)


def require_path(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_case_id(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected image file ending with {suffix}: {path}")
    return path.name[: -len(suffix)]


def label_case_id(path: Path) -> str:
    suffix = ".nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected label file ending with {suffix}: {path}")
    return path.name[: -len(suffix)]


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def resolve_python(
    explicit_python: Path | None,
    conda_env_root: Path,
    env_name: str,
) -> Path:
    if explicit_python is not None:
        return explicit_python.expanduser().resolve()
    return (conda_env_root / env_name / "bin" / "python").resolve()


def resolve_runtime(args) -> RuntimeConfig:
    conda_env_root = args.conda_env_root.expanduser().resolve()
    semantic_python = resolve_python(args.semantic_python, conda_env_root, args.semantic_env)
    instance_python = resolve_python(args.instance_python, conda_env_root, args.instance_env)
    return RuntimeConfig(
        conda_env_root=conda_env_root,
        semantic_env_name=args.semantic_env,
        instance_env_name=args.instance_env,
        semantic_python=semantic_python,
        instance_python=instance_python,
    )


def build_env(paths: SegmentationPaths, python_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(paths.nnunet_raw_root)
    env["nnUNet_preprocessed"] = str(paths.nnunet_preprocessed_root)
    env["nnUNet_results"] = str(paths.nnunet_results_root)
    paths.nnunet_preprocessed_root.mkdir(parents=True, exist_ok=True)
    paths.nnunet_results_root.mkdir(parents=True, exist_ok=True)

    pythonpath_parts = [str(paths.nnunet_root), str(paths.cellpose_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    path_parts = [str(python_bin.parent)]
    if env.get("PATH"):
        path_parts.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def run_command(
    command: list[str],
    env: dict[str, str],
    dry_run: bool,
    cwd: Path,
    runtime_label: str,
) -> None:
    log(f"[env={runtime_label}] $ {command_to_text(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def latest_cellpose_model(model_dir: Path) -> Path:
    require_path(model_dir, "Cellpose model directory")
    candidates = sorted(
        [path for path in model_dir.glob("cellpose_*") if path.is_file()],
        key=lambda item: (item.stat().st_mtime, item.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No Cellpose model found in {model_dir}")
    return candidates[-1]


def resolve_model(model_path: str | None, default_dir: Path, dry_run: bool) -> Path:
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
        if not dry_run:
            require_path(resolved, "Cellpose model")
        elif not resolved.exists():
            log(f"[dry-run] model path does not exist yet, using placeholder: {resolved}")
        return resolved
    try:
        return latest_cellpose_model(default_dir)
    except FileNotFoundError:
        if not dry_run:
            raise
        placeholder = default_dir / "<latest_cellpose_model>"
        log(f"[dry-run] no model found in {default_dir}, using placeholder: {placeholder}")
        return placeholder


def validate_dataset(paths: SegmentationPaths, dataset_name: str, split_index: int) -> None:
    require_path(paths.dataset_json, "dataset.json")
    require_path(paths.images_tr, "imagesTr directory")
    require_path(paths.labels_tr, "labelsTr directory")
    require_path(paths.split_json, "splits_final.json")

    dataset_json = read_json(paths.dataset_json)
    if dataset_json.get("name") != dataset_name:
        raise ValueError(
            f"dataset.json name mismatch: expected {dataset_name}, got {dataset_json.get('name')}"
        )
    if dataset_json.get("file_ending") != ".nii.gz":
        raise ValueError(
            f"dataset.json file_ending must be .nii.gz, got {dataset_json.get('file_ending')}"
        )

    image_files = sorted(paths.images_tr.glob("*_0000.nii.gz"))
    label_files = sorted(paths.labels_tr.glob("*.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No training images found in {paths.images_tr}")
    if not label_files:
        raise FileNotFoundError(f"No training labels found in {paths.labels_tr}")

    image_map = {image_case_id(path): path for path in image_files}
    label_map = {label_case_id(path): path for path in label_files}

    missing_labels = sorted(set(image_map) - set(label_map))
    extra_labels = sorted(set(label_map) - set(image_map))
    if missing_labels:
        raise ValueError(
            "Missing labels for training images:\n"
            + "\n".join(f"  - {case}_0000.nii.gz -> expected {case}.nii.gz" for case in missing_labels)
        )
    if extra_labels:
        raise ValueError(
            "Labels without matching training images:\n"
            + "\n".join(f"  - {case}.nii.gz" for case in extra_labels)
        )

    num_training = dataset_json.get("numTraining")
    if num_training != len(image_map):
        raise ValueError(
            "dataset.json numTraining does not match the actual image/label pairs. "
            f"dataset.json={num_training}, actual={len(image_map)}."
        )

    splits = read_json(paths.split_json)
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"Invalid split definition in {paths.split_json}")

    if split_index >= len(splits):
        raise IndexError(
            f"Requested split_index={split_index}, but only {len(splits)} splits exist"
        )

    split = splits[split_index]
    train_ids = set(split.get("train", []))
    val_ids = set(split.get("val", []))
    if not train_ids or not val_ids:
        raise ValueError("splits_final.json must contain non-empty train and val lists")

    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(
            "Train/val overlap detected in splits_final.json:\n"
            + "\n".join(f"  - {case}" for case in overlap)
        )

    all_cases = set(image_map)
    assigned_cases = train_ids | val_ids
    unknown_cases = sorted(assigned_cases - all_cases)
    unassigned_cases = sorted(all_cases - assigned_cases)
    if unknown_cases:
        raise ValueError(
            "splits_final.json references cases that do not exist in imagesTr/labelsTr:\n"
            + "\n".join(f"  - {case}" for case in unknown_cases)
        )
    if unassigned_cases:
        raise ValueError(
            "The following training cases are missing from splits_final.json:\n"
            + "\n".join(f"  - {case}" for case in unassigned_cases)
        )

    log("[OK] Dataset layout check passed")
    log(f"  - dataset: {dataset_name}")
    log(f"  - train/val pairs: {len(image_map)}")
    log(f"  - train split size: {len(train_ids)}")
    log(f"  - val split size: {len(val_ids)}")


def move_cellpose_masks(paths: SegmentationPaths, dry_run: bool) -> None:
    require_path(paths.test_cellpose_root, "Cellpose test directory")
    paths.reconstruction_warp_instance_mask.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(paths.test_cellpose_root.glob("*_cp_masks.png"))
    if not mask_files:
        if dry_run:
            log(
                f"[dry-run] no *_cp_masks.png files yet in {paths.test_cellpose_root}; "
                "skip final move step"
            )
            return
        raise FileNotFoundError(f"No *_cp_masks.png files found in {paths.test_cellpose_root}")

    for source in mask_files:
        target = paths.reconstruction_warp_instance_mask / source.name
        log(f"move {source} -> {target}")
        if dry_run:
            continue
        shutil.move(str(source), str(target))


def python_script_command(python_bin: Path, script: Path, *args: str) -> list[str]:
    return [str(python_bin), str(script), *args]


def python_module_command(python_bin: Path, module: str, *args: str) -> list[str]:
    return [str(python_bin), "-m", module, *args]


def python_entrypoint_command(
    python_bin: Path,
    module_name: str,
    function_name: str,
    *args: str,
) -> list[str]:
    snippet = f"from {module_name} import {function_name}; {function_name}()"
    return [str(python_bin), "-c", snippet, *args]


def run_pretrain(args, paths: SegmentationPaths, runtime: RuntimeConfig) -> None:
    validate_dataset(paths, args.dataset_name, args.split_index)

    semantic_env = build_env(paths, runtime.semantic_python)
    instance_env = build_env(paths, runtime.instance_python)

    commands = [
        (
            runtime.semantic_env_name,
            python_entrypoint_command(
                runtime.semantic_python,
                "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
                "plan_and_preprocess_entry",
                "-d",
                str(args.dataset_id),
                "--verify_dataset_integrity",
            ),
            semantic_env,
        ),
        (
            runtime.semantic_env_name,
            python_entrypoint_command(
                runtime.semantic_python,
                "nnunetv2.run.run_training",
                "run_training_entry",
                str(args.dataset_id),
                args.configuration,
                args.fold,
                "-tr",
                args.nnunet_trainer,
            ),
            semantic_env,
        ),
        (
            runtime.semantic_env_name,
            python_script_command(
                runtime.semantic_python,
                paths.prepare_script,
                "--stage",
                "pretrain",
                "--project_root",
                str(paths.repo_root),
                "--pred_input_dir",
                str(paths.labels_tr),
                "--refine_input_dir",
                str(paths.labels_tr),
                "--split_json",
                str(paths.split_json),
                "--split_index",
                str(args.split_index),
            ),
            semantic_env,
        ),
        (
            runtime.instance_env_name,
            python_module_command(
                runtime.instance_python,
                "cellpose",
                "--dir",
                str(paths.pretrain_train_dir),
                "--test_dir",
                str(paths.pretrain_val_dir),
                "--pretrained_model",
                "None",
                "--n_epochs",
                str(args.cellpose_pretrain_epochs),
                "--learning_rate",
                str(args.cellpose_learning_rate),
                "--verbose",
                "--save_txt",
                "--train",
                *([] if args.cpu else ["--use_gpu"]),
            ),
            instance_env,
        ),
    ]

    for runtime_label, command, env in commands:
        run_command(command, env, args.dry_run, paths.repo_root, runtime_label)


def run_finetune_predict(args, paths: SegmentationPaths, runtime: RuntimeConfig) -> None:
    require_path(paths.finetune_image, "finetune_image directory")
    semantic_env = build_env(paths, runtime.semantic_python)
    command = python_entrypoint_command(
        runtime.semantic_python,
        "nnunetv2.inference.predict_from_raw_data",
        "predict_entry_point",
        "-i",
        str(paths.finetune_image),
        "-o",
        str(paths.finetune_prediction),
        "-d",
        str(args.dataset_id),
        "-tr",
        args.nnunet_trainer,
        "-c",
        args.configuration,
        "-f",
        args.fold,
    )
    run_command(command, semantic_env, args.dry_run, paths.repo_root, runtime.semantic_env_name)
    log("Manual step required next:")
    log(f"  - refine adhesion cases in {paths.finetune_refine}")
    log("  - then run: bash run_seg.sh finetune-train")


def run_finetune_prepare(args, paths: SegmentationPaths, runtime: RuntimeConfig) -> None:
    require_path(paths.finetune_prediction, "finetune_prediction directory")
    require_path(paths.finetune_refine, "finetune_prediction_manual_refine directory")
    semantic_env = build_env(paths, runtime.semantic_python)
    command = python_script_command(
        runtime.semantic_python,
        paths.prepare_script,
        "--stage",
        "finetune",
        "--project_root",
        str(paths.repo_root),
        "--pred_input_dir",
        str(paths.finetune_prediction),
        "--refine_input_dir",
        str(paths.finetune_refine),
    )
    run_command(command, semantic_env, args.dry_run, paths.repo_root, runtime.semantic_env_name)


def run_finetune_train(args, paths: SegmentationPaths, runtime: RuntimeConfig) -> None:
    require_path(paths.finetune_cellpose_root, "Cellpose finetune dataset directory")
    pretrained_model = resolve_model(args.pretrain_model, paths.pretrain_model_dir, args.dry_run)
    instance_env = build_env(paths, runtime.instance_python)
    command = python_module_command(
        runtime.instance_python,
        "cellpose",
        "--dir",
        str(paths.finetune_cellpose_root),
        "--pretrained_model",
        str(pretrained_model),
        "--n_epochs",
        str(args.cellpose_finetune_epochs),
        "--learning_rate",
        str(args.cellpose_learning_rate),
        "--verbose",
        "--save_png",
        "--train",
        *([] if args.cpu else ["--use_gpu"]),
    )
    run_command(command, instance_env, args.dry_run, paths.repo_root, runtime.instance_env_name)


def run_test(args, paths: SegmentationPaths, runtime: RuntimeConfig) -> None:
    require_path(paths.test_image, "test_image directory")
    finetuned_model = resolve_model(args.finetune_model, paths.finetune_model_dir, args.dry_run)

    semantic_env = build_env(paths, runtime.semantic_python)
    instance_env = build_env(paths, runtime.instance_python)

    commands = [
        (
            runtime.semantic_env_name,
            python_entrypoint_command(
                runtime.semantic_python,
                "nnunetv2.inference.predict_from_raw_data",
                "predict_entry_point",
                "-i",
                str(paths.test_image),
                "-o",
                str(paths.test_prediction),
                "-d",
                str(args.dataset_id),
                "-tr",
                args.nnunet_trainer,
                "-c",
                args.configuration,
                "-f",
                args.fold,
            ),
            semantic_env,
        ),
        (
            runtime.semantic_env_name,
            python_script_command(
                runtime.semantic_python,
                paths.prepare_script,
                "--stage",
                "test",
                "--project_root",
                str(paths.repo_root),
                "--pred_input_dir",
                str(paths.test_prediction),
            ),
            semantic_env,
        ),
        (
            runtime.instance_env_name,
            python_module_command(
                runtime.instance_python,
                "cellpose",
                "--dir",
                str(paths.test_cellpose_root),
                "--pretrained_model",
                str(finetuned_model),
                "--diam_mean",
                str(args.test_diam_mean),
                "--save_png",
                "--verbose",
                *([] if args.cpu else ["--use_gpu"]),
            ),
            instance_env,
        ),
    ]

    for runtime_label, command, env in commands:
        run_command(command, env, args.dry_run, paths.repo_root, runtime_label)

    move_cellpose_masks(paths, args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified FabricFlow segmentation pipeline for nnUNetv2 + Cellpose."
    )
    parser.add_argument(
        "command",
        choices=[
            "check",
            "pretrain",
            "finetune-predict",
            "finetune-prepare",
            "finetune-train",
            "test",
        ],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent directory of segmentation/.",
    )
    parser.add_argument(
        "--conda-env-root",
        type=Path,
        default=DEFAULT_CONDA_ENV_ROOT,
        help="Fallback environment root used when explicit Python executables are not provided.",
    )
    parser.add_argument(
        "--semantic-env",
        type=str,
        default=DEFAULT_SEMANTIC_ENV_NAME,
        help="Semantic segmentation environment name.",
    )
    parser.add_argument(
        "--instance-env",
        type=str,
        default=DEFAULT_INSTANCE_ENV_NAME,
        help="Instance segmentation environment name.",
    )
    parser.add_argument(
        "--semantic-python",
        type=Path,
        default=None,
        help="Explicit Python executable for nnUNet and dataset preparation.",
    )
    parser.add_argument(
        "--instance-python",
        type=Path,
        default=None,
        help="Explicit Python executable for Cellpose stages.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help="nnUNet dataset ID. Default: 666",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Dataset folder name under nnUNet_raw. Default: Dataset666_FiberSegmentation",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default=DEFAULT_CONFIGURATION,
        help="nnUNet configuration. Default: 2d",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=DEFAULT_FOLD,
        help="nnUNet fold. Default: 0",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=DEFAULT_SPLIT_INDEX,
        help="Split index in splits_final.json. Default: 0",
    )
    parser.add_argument(
        "--nnunet-trainer",
        type=str,
        default=DEFAULT_NNUNET_TRAINER,
        help="nnUNet trainer name. Default: nnUNetTrainer",
    )
    parser.add_argument(
        "--cellpose-pretrain-epochs",
        type=int,
        default=DEFAULT_CELLPOSE_PRETRAIN_EPOCHS,
        help="Cellpose pretrain epochs. Default: 500",
    )
    parser.add_argument(
        "--cellpose-finetune-epochs",
        type=int,
        default=DEFAULT_CELLPOSE_FINETUNE_EPOCHS,
        help="Cellpose finetune epochs. Default: 100",
    )
    parser.add_argument(
        "--cellpose-learning-rate",
        type=float,
        default=DEFAULT_CELLPOSE_LR,
        help="Cellpose learning rate. Default: 0.001",
    )
    parser.add_argument(
        "--pretrain-model",
        type=str,
        default=None,
        help="Explicit pretrained Cellpose model for fine-tuning.",
    )
    parser.add_argument(
        "--finetune-model",
        type=str,
        default=None,
        help="Explicit fine-tuned Cellpose model for test inference.",
    )
    parser.add_argument(
        "--test-diam-mean",
        type=int,
        default=DEFAULT_TEST_DIAM_MEAN,
        help="Cellpose diam_mean for test inference. Default: 25",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable --use_gpu for Cellpose commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    paths = build_paths(args.repo_root.resolve(), args.dataset_name)
    runtime = resolve_runtime(args)

    require_path(paths.repo_root, "repository root")
    require_path(paths.prepare_script, "prepare_instance_seg_dataset.py")

    if not args.dry_run:
        require_path(runtime.semantic_python, "semantic python executable")
        require_path(runtime.instance_python, "instance python executable")

    log(f"[runtime] semantic env: {runtime.semantic_env_name} -> {runtime.semantic_python}")
    log(f"[runtime] instance env: {runtime.instance_env_name} -> {runtime.instance_python}")

    if args.command == "check":
        validate_dataset(paths, args.dataset_name, args.split_index)
        return 0
    if args.command == "pretrain":
        run_pretrain(args, paths, runtime)
        return 0
    if args.command == "finetune-predict":
        run_finetune_predict(args, paths, runtime)
        return 0
    if args.command == "finetune-prepare":
        run_finetune_prepare(args, paths, runtime)
        return 0
    if args.command == "finetune-train":
        run_finetune_train(args, paths, runtime)
        return 0
    if args.command == "test":
        run_test(args, paths, runtime)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
