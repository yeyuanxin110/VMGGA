"""
VMGGA Demo — Multimodal Image Matching.

Fill in the required parameters inside main(), then run:
    python demo_vmgga.py

Result saved to demo_result/<modality>_matches.png.
"""

import sys
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "src").is_dir() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.model import get_vmgga_model_config
from src.utils.image import expand_image_size, imread_mine, shear_warp
from src.vmgga import VMGGA


warnings.filterwarnings("ignore")


# ─────────────────────────── config helpers ───────────────────────────

def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path_from_project(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_demo_config(modality):
    config_path = PROJECT_ROOT / "configs" / "demo" / f"{modality}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot find demo config: {config_path}")
    return load_yaml_config(config_path), config_path


# ─────────────────────────── model loading ────────────────────────────

def load_vmgga(weight_path, device, match_threshold=2e-5, fine_window_size=None):
    config = get_vmgga_model_config(match_threshold, fine_window_size)
    model = VMGGA(config=config).to(device)

    if not Path(weight_path).is_file():
        modality_hint = Path(weight_path).stem  # e.g. vmgga_optical_sar
        raise FileNotFoundError(
            f"Model weight not found: {weight_path}\n"
            f"Please download the checkpoint for '{modality_hint}' from the Model Zoo\n"
            f"(see README) and place it in weight/."
        )

    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ──────────────────────── geometry utilities ──────────────────────────

def border_tuple(value):
    value = int(value)
    return value, value, value


def load_homography_label(label, invert=False):
    """Load one 3x3 homography mapping image0 coordinates to image1."""
    if isinstance(label, (str, Path)):
        label_path = Path(label)
        if not label_path.is_file():
            raise FileNotFoundError(f"Cannot read homography label: {label_path}")

        suffix = label_path.suffix.lower()
        if suffix == ".npy":
            homography = np.load(label_path)
        elif suffix == ".npz":
            archive = np.load(label_path)
            if not archive.files:
                raise ValueError(f"No array found in homography label: {label_path}")
            homography = archive[archive.files[0]]
        else:
            delimiter = "," if suffix == ".csv" else None
            homography = np.loadtxt(label_path, delimiter=delimiter)
    else:
        homography = np.asarray(label)

    homography = np.asarray(homography, dtype=np.float64).squeeze()
    if homography.size != 9:
        raise ValueError(
            "Homography label must contain exactly 9 values for one 3x3 matrix."
        )
    homography = homography.reshape(3, 3)
    if not np.isfinite(homography).all():
        raise ValueError("Homography label contains NaN or infinity.")
    if abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("Homography label is singular.")

    if invert:
        homography = np.linalg.inv(homography)
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    return homography


# ───────────────────── matching & visualization ──────────────────────

def ransac_filter(points0, points1, reprojection_threshold=2.0):
    if len(points0) < 4:
        return points0[:0], points1[:0], None

    try:
        import pydegensac

        homography, inlier_mask = pydegensac.findHomography(
            points0,
            points1,
            px_th=float(reprojection_threshold),
            conf=0.9999,
            max_iters=100000,
        )
    except Exception:
        homography, inlier_mask = cv.findHomography(
            points0,
            points1,
            getattr(cv, "USAC_MAGSAC", cv.RANSAC),
            float(reprojection_threshold),
            confidence=0.9999,
            maxIters=100000,
        )

    if homography is None or inlier_mask is None:
        return points0[:0], points1[:0], None

    inlier_mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    return points0[inlier_mask], points1[inlier_mask], homography


def geometric_correct_mask(points0, points1, homography, threshold, scale_factor=1.0):
    if homography is None or len(points0) == 0:
        return np.ones(len(points0), dtype=bool)

    points0_h = np.concatenate(
        [points0.astype(np.float64), np.ones((len(points0), 1))],
        axis=1,
    )
    projected = (homography @ points0_h.T).T
    valid = np.abs(projected[:, 2]) > 1e-8

    predicted = np.full_like(points0, np.inf, dtype=np.float64)
    predicted[valid] = projected[valid, :2] / projected[valid, 2:3]
    error = np.linalg.norm(predicted - points1, axis=1) / float(scale_factor)
    return valid & (error < threshold)


def draw_matches_canvas(
    image0_rgb,
    image1_rgb,
    points0,
    points1,
    homography_gt=None,
    correct_threshold=5.0,
    scale_factor=1.0,
    background_value=255,
    max_matches=0,
    correct_line_thickness=1,
    incorrect_line_thickness=1,
    point_radius=2,
):
    """Draw correct matches (green) and incorrect matches (blue) on a side-by-side canvas."""
    height = max(image0_rgb.shape[0], image1_rgb.shape[0])
    width0 = image0_rgb.shape[1]
    width1 = image1_rgb.shape[1]
    canvas = np.full(
        (height, width0 + width1, 3),
        int(background_value),
        dtype=np.uint8,
    )
    canvas[: image0_rgb.shape[0], :width0] = image0_rgb
    canvas[: image1_rgb.shape[0], width0:] = image1_rgb

    if max_matches > 0 and len(points0) > max_matches:
        indices = np.linspace(0, len(points0) - 1, max_matches, dtype=np.int64)
        points0 = points0[indices]
        points1 = points1[indices]

    correct = geometric_correct_mask(
        points0, points1, homography_gt, correct_threshold, scale_factor=scale_factor,
    )

    points0_int = np.round(points0).astype(np.int32)
    points1_int = np.round(points1).astype(np.int32)
    offset = np.array([width0, 0], dtype=np.int32)

    # green = correct, blue = incorrect
    if correct_line_thickness > 0:
        for pt0, pt1 in zip(points0_int[correct], points1_int[correct]):
            cv.line(canvas, tuple(pt0), tuple(pt1 + offset), (0, 255, 0), correct_line_thickness)

    if incorrect_line_thickness > 0:
        for pt0, pt1 in zip(points0_int[~correct], points1_int[~correct]):
            cv.line(canvas, tuple(pt0), tuple(pt1 + offset), (255, 0, 0), incorrect_line_thickness)

    # yellow keypoints on top
    if point_radius > 0:
        for pt0 in points0_int:
            cv.circle(canvas, tuple(pt0), point_radius, (255, 255, 0), -1)
        for pt1 in points1_int:
            cv.circle(canvas, tuple(pt1 + offset), point_radius, (255, 255, 0), -1)

    return canvas, correct


@torch.no_grad()
def match_pair(
    model,
    image0_path,
    image1_path,
    device,
    mode=1,
    homography_label=None,
    invert_homography_label=False,
    rotate=0.0,
    scale=1.0,
    homography_x=0.0,
    homography_y=0.0,
    inference_border=0,
    display_border=255,
    ransac_threshold=2.0,
):
    """Run VMGGA on one image pair and return match results."""
    image0_bgr = cv.imread(str(image0_path), cv.IMREAD_COLOR)
    image1_bgr = cv.imread(str(image1_path), cv.IMREAD_COLOR)
    if image0_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image0_path}")
    if image1_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image1_path}")

    if mode == 1:
        # pre-aligned pair + synthetic rotation, scale, and perspective perturbation
        # 预对齐图像对 + 模拟旋转、尺度和透视扰动
        image1_inference, homography_gt = imread_mine(
            str(image1_path), rotate=rotate, scale=scale,
            border_value=border_tuple(inference_border),
        )
        image1_display, display_homography = imread_mine(
            str(image1_path), rotate=rotate, scale=scale,
            border_value=border_tuple(display_border),
        )
        if homography_x or homography_y:
            image1_inference, perspective = shear_warp(
                image1_inference, 0, 0, homography_x, homography_y,
                border_value=border_tuple(inference_border),
            )
            image1_display, display_perspective = shear_warp(
                image1_display, 0, 0, homography_x, homography_y,
                border_value=border_tuple(display_border),
            )
            homography_gt = perspective @ homography_gt
            display_homography = display_perspective @ display_homography

        if not np.allclose(homography_gt, display_homography):
            raise RuntimeError("Inference and display transformations do not match.")
        scale_factor = scale
        verification_enabled = True

    elif mode == 2:
        # unaligned pair + user-provided homography label
        # 未对齐图像对 + 用户提供的单应矩阵标签
        if homography_label is None:
            raise ValueError("Mode 2 requires a 3x3 homography label.")
        image1_inference = image1_bgr.copy()
        image1_display = image1_bgr.copy()
        homography_gt = load_homography_label(homography_label, invert=invert_homography_label)
        scale_factor = 1.0
        verification_enabled = True

    elif mode == 3:
        # two images only; no accuracy evaluation
        # 仅输入两张图，不做精度验证
        image1_inference = image1_bgr.copy()
        image1_display = image1_bgr.copy()
        homography_gt = None
        scale_factor = 1.0
        verification_enabled = False

    else:
        raise ValueError("mode must be 1, 2, or 3.")

    gray0 = cv.cvtColor(image0_bgr, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(image1_inference, cv.COLOR_BGR2GRAY)
    original_height0, original_width0 = gray0.shape
    original_height1, original_width1 = gray1.shape

    gray0, _ = expand_image_size(gray0, 16)
    gray1, _ = expand_image_size(gray1, 16)

    tensor0 = torch.from_numpy(gray0).float()[None, None].to(device) / 255.0
    tensor1 = torch.from_numpy(gray1).float()[None, None].to(device) / 255.0
    batch = {"image0": tensor0, "image1": tensor1}
    model(batch)

    points0 = batch["mkpts0_f"].detach().cpu().numpy()
    points1 = batch["mkpts1_f"].detach().cpu().numpy()
    valid = (
        (points0[:, 0] >= 0) & (points0[:, 0] < original_width0)
        & (points0[:, 1] >= 0) & (points0[:, 1] < original_height0)
        & (points1[:, 0] >= 0) & (points1[:, 0] < original_width1)
        & (points1[:, 1] >= 0) & (points1[:, 1] < original_height1)
    )
    points0 = points0[valid]
    points1 = points1[valid]

    inliers0, inliers1, homography_pred = ransac_filter(
        points0, points1, reprojection_threshold=ransac_threshold,
    )
    image0_rgb = cv.cvtColor(image0_bgr, cv.COLOR_BGR2RGB)
    image1_rgb = cv.cvtColor(image1_display, cv.COLOR_BGR2RGB)

    return {
        "image0_rgb": image0_rgb,
        "image1_rgb": image1_rgb,
        "points0": points0,
        "points1": points1,
        "inliers0": inliers0,
        "inliers1": inliers1,
        "homography_gt": homography_gt,
        "homography_pred": homography_pred,
        "scale_factor": scale_factor,
        "verification_enabled": verification_enabled,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          MAIN  /  主函数                            ║
# ║                                                                    ║
# ║  Fill in the parameters below, then run:  python demo_vmgga.py     ║
# ║  填写下列参数后运行:  python demo_vmgga.py                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():

    # =====================================================================
    #  必填 / REQUIRED
    # =====================================================================

    # 模态 (Modality)
    #    optical_infrared | optical_sar | optical_map | optical_depth
    modality = "optical_sar"

    # 基准图像路径 (Reference image path)
    #    Absolute or relative to the project root.
    #    绝对路径或相对于项目根目录的路径。
    image0 = f"examples/{modality}/image0.png"

    # 实时图像路径 (Sensed image path)
    #    Absolute or relative to the project root.
    #    绝对路径或相对于项目根目录的路径。
    image1 = f"examples/{modality}/image1.png"

    # 匹配模式 (Matching mode)
    #    1 = 预对齐图像对 + 自动模拟旋转/尺度/透视扰动 (推荐)
    #        pre-aligned pair + auto synthetic rotation / scale / perspective
    #    2 = 未对齐图像对 + 用户提供单应矩阵标签文件
    #        unaligned pair + your own homography label file
    #    3 = 仅输入两张图像，不做精度评估
    #        two images only, no accuracy evaluation
    mode = 1

    if mode == 1:
        # ── Mode 1 参数（必填）/ Mode 1 parameters (required) ──

        # 旋转角度 (Rotation angle in degrees)
        rotate = 15

        # 尺度因子 (Scale factor)
        scale = 1.2

        # 透视扰动 x (Perspective contraction in x direction)
        homography_x = 5e-4

        # 透视扰动 y (Perspective contraction in y direction)
        homography_y = 5e-4

    elif mode == 2:
        # ── Mode 2 参数（必填）/ Mode 2 parameters (required) ──

        # 单应矩阵标签文件路径 (Path to homography label file)
        #    Supports .npy / .npz / .txt / .csv.
        #    标签方向为 image0 → image1。
        homography_label = f"examples/{modality}/homography.txt"

        # 反转标签方向 (Invert label direction)
        #    True  = label maps image1 → image0
        #    False = label maps image0 → image1 (默认 / default)
        invert_label = False

    # =====================================================================
    #  以下为运行逻辑，通常无需修改。
    #  Runtime logic below — usually no need to edit.
    # =====================================================================

    # load YAML config for the chosen modality
    cfg, config_path = load_demo_config(modality)

    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    demo_cfg = cfg.get("demo", {})

    # resolve paths
    image0 = resolve_path_from_project(image0)
    image1 = resolve_path_from_project(image1)
    weights = resolve_path_from_project(cfg.get("weight_path"))
    output = resolve_path_from_project(
        demo_cfg.get("output") or f"demo_result/{modality}_matches.png"
    )

    if not weights:
        raise ValueError(
            f"No weight_path found in config: {config_path}\n"
            f"Make sure the YAML has a 'weight_path' entry."
        )
    if not image0 or not image1:
        raise ValueError(
            "Both image0 and image1 paths are required.\n"
            "Please set them inside main() at the top of this file."
        )

    # build mode-specific options
    mode = int(mode)
    if mode == 1:
        mode_options = {
            "rotate": float(rotate),
            "scale": float(scale),
            "homography_x": float(homography_x),
            "homography_y": float(homography_y),
        }
    elif mode == 2:
        if homography_label is None:
            raise ValueError(
                "Mode 2 requires homography_label.\n"
                "Please set it inside main() at the top of this file."
            )
        mode_options = {
            "homography_label": resolve_path_from_project(homography_label),
            "invert_homography_label": bool(invert_label),
        }
    elif mode == 3:
        mode_options = {}
    else:
        raise ValueError(f"mode must be 1, 2, or 3. Got: {mode}")

    # read backend parameters from YAML (用户无需关心 / users don't need to touch)
    match_threshold = model_cfg.get("match_threshold", 2e-5)
    fine_window_size = model_cfg.get("fine_window_size")
    ransac_threshold = float(eval_cfg.get("ransac_threshold", 2.0))
    correct_threshold = float(demo_cfg.get("correct_threshold", 5.0))
    inference_border = int(eval_cfg.get("inference_border", 0))
    display_border = int(demo_cfg.get("display_border", 255))
    max_matches = int(demo_cfg.get("max_matches", 0))
    correct_line_thickness = int(demo_cfg.get("correct_line_thickness", 1))
    incorrect_line_thickness = int(demo_cfg.get("incorrect_line_thickness", 1))
    point_radius = int(demo_cfg.get("point_radius", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = load_vmgga(weights, device, match_threshold=match_threshold, fine_window_size=fine_window_size)

    # run matching
    result = match_pair(
        model, image0, image1, device=device, mode=mode,
        inference_border=inference_border, display_border=display_border,
        ransac_threshold=ransac_threshold, **mode_options,
    )

    # draw and save
    canvas_rgb, correct = draw_matches_canvas(
        result["image0_rgb"], result["image1_rgb"],
        result["inliers0"], result["inliers1"],
        homography_gt=result["homography_gt"],
        correct_threshold=correct_threshold,
        scale_factor=result["scale_factor"],
        background_value=display_border,
        max_matches=max_matches,
        correct_line_thickness=correct_line_thickness,
        incorrect_line_thickness=incorrect_line_thickness,
        point_radius=point_radius,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_bgr = cv.cvtColor(canvas_rgb, cv.COLOR_RGB2BGR)
    if not cv.imwrite(str(output), output_bgr):
        raise OSError(f"Failed to save visualization: {output}")

    # summary
    mode_names = {1: "synthetic transform", 2: "homography label", 3: "images only"}
    print(f"Modality : {modality}")
    print(f"Mode     : {mode} ({mode_names.get(mode, 'unknown')})")
    print(f"Weight   : {weights}")
    print(f"Image 0  : {image0}")
    print(f"Image 1  : {image1}")
    print(f"Matches  : {len(result['points0'])} raw  →  {len(result['inliers0'])} RANSAC inliers")
    if result["verification_enabled"]:
        print(f"Correct  : {int(correct.sum())}/{len(correct)}  (within {correct_threshold:g} px)")
    print(f"Saved    : {output.resolve()}")


if __name__ == "__main__":
    main()
