from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage import measure


ROTATION_ANGLE = 95
GROUP_COUNT = 9
POINTS_PER_GROUP = 4
TOTAL_IDS = GROUP_COUNT * POINTS_PER_GROUP

BASE_DIR = Path(__file__).resolve().parent
WARP_INPUT_DIR = BASE_DIR / "warp_instance_seg_mask"
WEFT_INPUT_DIR = BASE_DIR / "weft_semantic_seg_mask"
NNUNET_INPUT_DIR = BASE_DIR / "nnunet_prediction"

WARP_OUTPUT_DIR = BASE_DIR / "warp_instance_reconstruction_values"
SEGMENTATION_OUTPUT_DIR = BASE_DIR / "segmentation_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FabricFlow reconstruction pipeline")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Reconstruction base directory. Defaults to the current script directory.",
    )
    parser.add_argument(
        "--rotation-angle",
        type=float,
        default=ROTATION_ANGLE,
        help="Rotation angle in degrees. Default is 95.",
    )
    return parser.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_png_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() == ".png")


def normalize_warp_masks(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(input_dir.iterdir()):
        if not src.is_file():
            continue
        if src.name.endswith("_cp_masks.png"):
            dst_name = src.name.replace("_cp_masks.png", ".png")
        elif src.suffix.lower() == ".png":
            dst_name = src.name
        else:
            continue
        shutil.copy2(src, output_dir / dst_name)


def erode_connected_components(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((3, 3), np.uint8)

    for image_path in iter_png_files(input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        eroded_image = np.zeros_like(image)
        labels = measure.label(image, connectivity=2)
        props = measure.regionprops(labels)

        for prop in props:
            if prop.label == 0:
                continue
            mask = (labels == prop.label).astype(np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            eroded_image[eroded_mask > 0] = prop.label

        cv2.imwrite(str(output_dir / image_path.name), eroded_image)


def binarize_instance_masks(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_png_files(input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        gray_8bit = np.zeros_like(image, dtype=np.uint8)
        if image.max() > 0:
            gray_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / float(image.max())))
            gray_8bit[gray_8bit > 0] = 255

        cv2.imwrite(str(output_dir / image_path.name), gray_8bit)


def binarize_weft_masks(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_png_files(input_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[gray > 0] = 255
        cv2.imwrite(str(output_dir / image_path.name), gray)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT,
    )


def rotate_directory(input_dir: Path, output_dir: Path, angle: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_png_files(input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        rotated_image = rotate_image(image, angle)
        cv2.imwrite(str(output_dir / image_path.name), rotated_image)


def get_centroids(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    return image, centroids[1:], stats[1:], labels


def kmeans_vertical_lines(
    centroids: np.ndarray,
    k: int = GROUP_COUNT,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = centroids[:, 0].reshape(-1, 1)
    centers = np.linspace(x_values.min(), x_values.max(), k).reshape(-1, 1)
    labels = np.zeros(x_values.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        distances = np.abs(x_values - centers.T)
        new_labels = np.argmin(distances, axis=1)
        new_centers = np.array(
            [
                x_values[new_labels == idx].mean() if np.any(new_labels == idx) else centers[idx, 0]
                for idx in range(k)
            ]
        ).reshape(-1, 1)
        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            labels = new_labels
            break
        centers = new_centers
        labels = new_labels

    return centers.flatten(), labels


def filter_cluster(cluster_indices: list[int], areas: np.ndarray, target_count: int = POINTS_PER_GROUP) -> list[int]:
    if len(cluster_indices) <= target_count:
        return cluster_indices
    return sorted(cluster_indices, key=lambda idx: areas[idx], reverse=True)[:target_count]


def process_slice(image_path: Path) -> dict:
    image, centroids, stats, component_labels = get_centroids(image_path)
    if len(centroids) == 0:
        raise ValueError(f"No connected components found in {image_path}")

    centers, labels = kmeans_vertical_lines(centroids, k=GROUP_COUNT)
    clusters = {}
    for idx in range(GROUP_COUNT):
        member_indices = np.where(labels == idx)[0]
        clusters[idx] = member_indices.tolist()

    areas = stats[:, 4]
    for idx in range(GROUP_COUNT):
        if len(clusters[idx]) > POINTS_PER_GROUP:
            clusters[idx] = filter_cluster(clusters[idx], areas, target_count=POINTS_PER_GROUP)

    match = re.search(r"Fiber1B(\d{4})\.png", image_path.name)
    slice_num = int(match.group(1)) if match else -1
    return {
        "slice_num": slice_num,
        "filename": image_path.name,
        "img": image,
        "centroids": centroids,
        "stats": stats,
        "comp_labels": component_labels,
        "centers": centers,
        "clusters": clusters,
        "labels": labels,
    }


def process_all_slices(src_folder: Path) -> dict[int, dict]:
    slices = {}
    for image_path in iter_png_files(src_folder):
        slice_data = process_slice(image_path)
        slices[slice_data["slice_num"]] = slice_data
    return dict(sorted(slices.items(), key=lambda item: item[0]))


def get_cluster_points(slice_data: dict, cluster_idx: int) -> np.ndarray:
    points = []
    for idx in slice_data["clusters"].get(cluster_idx, []):
        if idx < len(slice_data["centroids"]):
            points.append(slice_data["centroids"][idx])

    if "added_centroids_dict" in slice_data and cluster_idx in slice_data["added_centroids_dict"]:
        points.extend(slice_data["added_centroids_dict"][cluster_idx])

    return np.array(points) if points else np.empty((0, 2))


def correct_under_segmentation(slices: dict[int, dict]) -> dict[int, dict]:
    slice_nums = sorted(slices.keys())
    for slice_num in slice_nums:
        slice_a = slices[slice_num]
        slice_a.setdefault("added_centroids_dict", {})

        for cluster_idx in range(GROUP_COUNT):
            cluster = slice_a["clusters"].get(cluster_idx, [])
            if len(cluster) >= POINTS_PER_GROUP:
                continue

            missing = POINTS_PER_GROUP - len(cluster)
            candidate_slice = None
            best_diff = None

            for other_slice_num in slice_nums:
                if other_slice_num == slice_num:
                    continue
                slice_b = slices[other_slice_num]
                points_b = get_cluster_points(slice_b, cluster_idx)
                if points_b.shape[0] != POINTS_PER_GROUP:
                    continue
                diff = abs(other_slice_num - slice_num)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    candidate_slice = slice_b

            if candidate_slice is None:
                continue

            points_b = get_cluster_points(candidate_slice, cluster_idx)
            points_a = get_cluster_points(slice_a, cluster_idx)
            assigned = set()

            if points_a.shape[0] > 0:
                for point in points_a:
                    distances = np.linalg.norm(points_b - point, axis=1)
                    assigned.add(int(np.argmin(distances)))

            missing_indices = [idx for idx in range(POINTS_PER_GROUP) if idx not in assigned]
            added_points = []
            for idx in missing_indices[:missing]:
                new_point = points_b[idx]
                added_points.append(new_point)
                new_index = len(slice_a["centroids"]) + len(added_points) - 1
                slice_a["clusters"][cluster_idx].append(new_index)

            slice_a["added_centroids_dict"].setdefault(cluster_idx, [])
            slice_a["added_centroids_dict"][cluster_idx].extend(added_points)

    return slices


def assign_global_ids(slices: dict[int, dict]) -> dict[int, dict]:
    for slice_num, slice_data in slices.items():
        slice_data["labeled_points"] = []
        for group_idx in range(GROUP_COUNT):
            points = get_cluster_points(slice_data, group_idx)
            if points.shape[0] != POINTS_PER_GROUP:
                continue

            sorted_points = sorted(points, key=lambda point: point[1], reverse=(group_idx % 2 == 1))
            base_id = group_idx * POINTS_PER_GROUP + 1
            for offset, point in enumerate(sorted_points):
                global_id = base_id + offset
                slice_data["labeled_points"].append((global_id, point[0], point[1], slice_num))

    return slices


def assign_component_labels(slices: dict[int, dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for slice_data in slices.values():
        num_components = len(slice_data["centroids"])
        new_labels = np.zeros(num_components, dtype=np.uint8)
        labeled_points = slice_data["labeled_points"]
        centroids = slice_data["centroids"]
        component_labels = slice_data["comp_labels"]

        centroid_to_label = {}
        for global_id, x, y, _ in labeled_points:
            distances = np.linalg.norm(centroids - np.array([x, y]), axis=1)
            nearest_idx = int(np.argmin(distances))
            if distances[nearest_idx] < 1.0:
                centroid_to_label[nearest_idx] = global_id

        for idx in range(num_components):
            if idx in centroid_to_label:
                new_labels[idx] = centroid_to_label[idx]

        labeled_image = np.zeros_like(slice_data["img"], dtype=np.uint8)
        for idx in range(num_components):
            if new_labels[idx] > 0:
                labeled_image[component_labels == (idx + 1)] = new_labels[idx]

        cv2.imwrite(str(output_dir / slice_data["filename"]), labeled_image)


def expand_labels(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((3, 3), np.uint8)

    for image_path in iter_png_files(input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        expanded_image = image.copy()
        unique_labels = np.unique(image)
        unique_labels = unique_labels[unique_labels > 0]
        occupied_mask = image > 0

        for label in unique_labels:
            component = (image == label).astype(np.uint8)
            dilated_component = cv2.dilate(component, kernel, iterations=1)
            expansion_mask = (dilated_component > 0) & (expanded_image == 0) & (~occupied_mask)
            expanded_image[expansion_mask] = label
            occupied_mask = occupied_mask | (dilated_component > 0)

        cv2.imwrite(str(output_dir / image_path.name), expanded_image)


def annotate_component_values(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_png_files(input_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        labeled_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        unique_values = np.unique(image)
        unique_values = unique_values[unique_values != 0]

        for value in unique_values:
            mask = (image == value).astype(np.uint8)
            num_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for idx in range(1, num_labels):
                centroid_x, centroid_y = int(centroids[idx][0]), int(centroids[idx][1])
                cv2.putText(
                    labeled_image,
                    str(value),
                    (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(output_dir / image_path.name), labeled_image)


def combine_segmentations(warp_dir: Path, weft_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for warp_path in iter_png_files(warp_dir):
        weft_path = weft_dir / warp_path.name
        if not weft_path.exists():
            continue

        warp_image = cv2.imread(str(warp_path), cv2.IMREAD_GRAYSCALE)
        weft_image = cv2.imread(str(weft_path), cv2.IMREAD_GRAYSCALE)
        if warp_image is None or weft_image is None:
            raise ValueError(f"Failed to read paired images: {warp_path}, {weft_path}")
        if warp_image.shape != weft_image.shape:
            raise ValueError(f"Image size mismatch: {warp_path.name}")

        combined = np.where(warp_image != 0, warp_image, weft_image).astype(np.uint8)
        cv2.imwrite(str(output_dir / warp_path.name), combined)


def run_pipeline(base_dir: Path, rotation_angle: float) -> None:
    base_dir = base_dir.resolve()
    warp_input_dir = base_dir / WARP_INPUT_DIR.name
    weft_input_dir = base_dir / WEFT_INPUT_DIR.name
    nnunet_input_dir = base_dir / NNUNET_INPUT_DIR.name
    warp_output_dir = base_dir / WARP_OUTPUT_DIR.name
    segmentation_output_dir = base_dir / SEGMENTATION_OUTPUT_DIR.name

    ensure_exists(warp_input_dir)
    ensure_exists(weft_input_dir)
    ensure_exists(nnunet_input_dir)

    recreate_dir(warp_output_dir)
    recreate_dir(segmentation_output_dir)

    with tempfile.TemporaryDirectory(dir=base_dir, prefix=".tmp_reconstruction_") as temp_root_str:
        temp_root = Path(temp_root_str)
        normalized_warp_dir = temp_root / "warp_instance_seg_mask"
        eroded_dir = temp_root / "warp_instance_seg_mask_eroded"
        eroded_gray_dir = temp_root / "warp_instance_seg_mask_eroded_gray"
        rotated_warp_dir = temp_root / f"warp_instance_seg_mask_eroded_gray_rotated_{int(rotation_angle)}"
        weft_gray_dir = temp_root / "weft_semantic_seg_mask_gray"
        rotated_weft_dir = temp_root / f"weft_semantic_seg_mask_gray_rotated_{int(rotation_angle)}"
        labeled_dir = temp_root / f"warp_instance_seg_mask_eroded_gray_rotated_{int(rotation_angle)}_labeled"
        expanded_dir = temp_root / (
            f"warp_instance_seg_mask_eroded_gray_rotated_{int(rotation_angle)}_labeled_expanded"
        )

        print(f"Using rotation angle: {rotation_angle}")
        print(f"Base directory: {base_dir}")
        print("Note: nnunet_prediction exists and is validated, but the original notebook pipeline does not use it.")

        normalize_warp_masks(warp_input_dir, normalized_warp_dir)
        erode_connected_components(normalized_warp_dir, eroded_dir)
        binarize_instance_masks(eroded_dir, eroded_gray_dir)
        binarize_weft_masks(weft_input_dir, weft_gray_dir)
        rotate_directory(eroded_gray_dir, rotated_warp_dir, rotation_angle)
        rotate_directory(weft_gray_dir, rotated_weft_dir, rotation_angle)

        slices = process_all_slices(rotated_warp_dir)
        slices = correct_under_segmentation(slices)
        slices = assign_global_ids(slices)
        assign_component_labels(slices, labeled_dir)

        expand_labels(labeled_dir, expanded_dir)
        annotate_component_values(expanded_dir, warp_output_dir)
        combine_segmentations(expanded_dir, rotated_weft_dir, segmentation_output_dir)

    print(f"Saved warp reconstruction values to: {warp_output_dir}")
    print(f"Saved combined segmentation results to: {segmentation_output_dir}")


def main() -> None:
    args = parse_args()
    run_pipeline(base_dir=args.base_dir, rotation_angle=args.rotation_angle)


if __name__ == "__main__":
    main()
