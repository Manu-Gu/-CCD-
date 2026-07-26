"""分度板刻度线检测与标定模块 —— 基于垂直投影法"""

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from . import config


def detect_scale_lines_by_projection(gray_image):
    """
    垂直投影法检测刻度线位置 —— 比霍夫直线更鲁棒

    原理：将图像沿竖直方向做灰度投影（列求和），
    刻度线处像素较暗 → 投影曲线形成凹陷/低谷，
    检测低谷位置即为刻度线 x 坐标。

    返回: x 坐标列表（从小到大排序）
    """
    h, w = gray_image.shape

    # 0. 自动裁剪到刻度线所在行区域
    roi = _find_scale_row_region(gray_image)
    if roi is not None:
        y0, y1 = roi
        crop = gray_image[y0:y1, :]
    else:
        y0, y1 = 0, h
        crop = gray_image

    # 1. 增强竖直方向的线条：形态学顶帽 + 竖直核
    ch, cw = crop.shape
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, ch // 30)))
    tophat = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel_v)

    # 2. 二值化（刻度线通常为深色即低灰度值）
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 竖直方向投影：每列的像素亮度和（刻度线处 = 白色 = 高值）
    projection = np.sum(binary, axis=0).astype(np.float64)

    # 4. 对投影做平滑
    window = max(3, w // 100)
    if window % 2 == 0:
        window += 1
    window = min(window, len(projection) - 2)
    if window < 3:
        return []
    smoothed = savgol_filter(projection, window, 2)

    # 5. 找峰值（刻度线位置）
    min_distance = max(5, w // 80)
    height_threshold = np.max(smoothed) * 0.15
    peaks, properties = find_peaks(
        smoothed,
        distance=min_distance,
        height=height_threshold,
        prominence=np.max(smoothed) * 0.05
    )

    if len(peaks) < 3:
        return []

    # 6. 用间距一致性过滤异常点
    peak_positions = peaks.astype(np.float64).tolist()
    peak_positions = _filter_by_spacing_consistency(peak_positions)

    return peak_positions


def _find_scale_row_region(gray_image):
    """在图像中找到刻度线所在的竖直行区域 (y0, y1)"""
    h, w = gray_image.shape
    # 竖向梯度：刻度线处水平方向变化大
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_x = np.abs(grad_x)

    # 逐行求和，刻度线密集的行梯度大
    row_grad = np.sum(grad_x, axis=1)

    # 平滑并找峰值区域
    smooth_len = max(3, h // 20)
    if smooth_len % 2 == 0:
        smooth_len += 1
    if smooth_len >= 3 and len(row_grad) > smooth_len:
        row_smooth = savgol_filter(row_grad, min(smooth_len, len(row_grad) - 2), 2)
    else:
        row_smooth = row_grad

    threshold = np.mean(row_smooth) + 0.3 * np.std(row_smooth)
    mask = row_smooth > threshold

    if not np.any(mask):
        return None

    indices = np.where(mask)[0]
    y0 = max(0, indices[0] - h // 20)
    y1 = min(h, indices[-1] + h // 20)
    return (y0, y1)


def get_projection_signal(gray_image, axis=0):
    """
    获取投影信号
    axis=0: 垂直投影（用于竖刻度线，返回 x 方向投影）
    axis=1: 水平投影（用于横刻度线，返回 y 方向投影）
    """
    h, w = gray_image.shape

    if axis == 1:
        # 水平投影：横刻度线 → 沿 y 轴投影
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 30), 1))
        tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel_h)
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        projection = np.sum(binary, axis=1).astype(np.float64)
    else:
        # 垂直投影：竖刻度线
        roi = _find_scale_row_region(gray_image)
        if roi is not None:
            y0, y1 = roi
            crop = gray_image[y0:y1, :]
        else:
            crop = gray_image
        ch, cw = crop.shape
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, ch // 30)))
        tophat = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel_v)
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        projection = np.sum(binary, axis=0).astype(np.float64)

    length = len(projection)
    window = max(3, length // 100)
    if window % 2 == 0:
        window += 1
    window = min(window, length - 2)
    if window < 3:
        return projection

    smoothed = savgol_filter(projection, window, 2)
    return smoothed


def detect_scale_lines_guided(gray_image, zero_pos, adjacent_positions):
    """
    半自动模式：方向自适应，支持竖线和横线刻度。

    zero_pos:       (x, y) 零点位置
    adjacent_positions: [(x1,y1), (x2,y2), ...] 相邻刻度线位置（方向不限）
    
    返回: (lines, is_vertical, orientation)
      lines:        刻度线坐标列表（沿主轴方向，从小到大排序）
      is_vertical:  True=竖刻度线, False=横刻度线
    """
    if len(adjacent_positions) < 2:
        raise ValueError("至少需要标注2条相邻刻度线才能计算间距")

    # 1. 判断方向（竖刻度 vs 横刻度）
    #    比较相邻点在 x 和 y 方向上的离散度
    pts = np.array(adjacent_positions)
    z = np.array(zero_pos)
    all_pts = np.vstack([z, pts])

    x_range = np.ptp(all_pts[:, 0])
    y_range = np.ptp(all_pts[:, 1])

    if x_range >= y_range:
        # 竖刻度线：x 方向变化为主
        is_vertical = True
        zero_val = float(zero_pos[0])
        adjacent_vals = sorted([p[0] for p in adjacent_positions])
        projection = get_projection_signal(gray_image, axis=0)
    else:
        # 横刻度线：y 方向变化为主
        is_vertical = False
        zero_val = float(zero_pos[1])
        adjacent_vals = sorted([p[1] for p in adjacent_positions])
        projection = get_projection_signal(gray_image, axis=1)

    # 2. 计算间距
    gaps = np.diff(adjacent_vals)
    spacing = np.median(gaps)
    if spacing <= 0:
        raise ValueError("无法计算刻度线间距，请检查标注")

    # 3. 判断刻度值增长方向（零线在哪侧）
    #    相邻点在零线的哪一侧？
    direction = 1 if adjacent_vals[0] > zero_val else -1

    # 4. 以 zero 为锚点向两侧扩展
    half_window = max(3, int(spacing * 0.35))

    def snap_to_peak(proj, nominal, window):
        w = len(proj)
        lo = max(0, int(nominal - window))
        hi = min(w, int(nominal + window + 1))
        if hi <= lo:
            return nominal
        segment = proj[lo:hi]
        if len(segment) < 3:
            return nominal
        local_peaks, _ = find_peaks(segment, prominence=np.max(segment) * 0.1)
        if len(local_peaks) == 0:
            return nominal
        center_rel = nominal - lo
        best = local_peaks[np.argmin(np.abs(local_peaks - center_rel))]
        return float(lo + best)

    lines = []

    # 正向扩展
    for i in range(0, 200):
        nominal = zero_val + i * spacing * direction
        if nominal < 0 or nominal > len(projection) - 1:
            break
        snapped = snap_to_peak(projection, nominal, half_window)
        if snapped < 0 or snapped > len(projection) - 1:
            break
        lines.append(snapped)

    # 反向扩展
    for i in range(1, 200):
        nominal = zero_val - i * spacing * direction
        if nominal < 0 or nominal > len(projection) - 1:
            break
        snapped = snap_to_peak(projection, nominal, half_window)
        if snapped < 0 or snapped > len(projection) - 1:
            break
        lines.append(snapped)

    lines.sort()

    # 5. 剔除间距异常点
    lines = _filter_by_spacing_consistency(lines)

    if len(lines) < 3:
        raise ValueError(f"半自动检测结果不足（仅{len(lines)}条），请检查标注")

    return lines, is_vertical


def _filter_by_spacing_consistency(positions):
    """基于刻度线均匀分布假设，剔除间距异常的离群点"""
    if len(positions) < 3:
        return positions

    gaps = np.diff(positions)
    median_gap = np.median(gaps)
    if median_gap <= 0:
        return positions

    # 逐点检查相邻间距，偏差超过 30% 视为异常
    valid = [positions[0]]
    for i in range(1, len(positions)):
        gap = positions[i] - valid[-1]
        # 检查当前间距是否在合理范围（可能是多倍间距的大刻度线）
        ratio = gap / median_gap
        if 0.5 < ratio < 2.5 or (ratio > 2.5 and abs(round(ratio) - ratio) < 0.25):
            valid.append(positions[i])
        # 否则跳过该点

    return valid


def find_scale_lines(edge_image):
    """保留旧接口兼容性，实际推荐使用 detect_scale_lines_by_projection"""
    lines = cv2.HoughLinesP(
        edge_image, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=config.MIN_TICK_LENGTH, maxLineGap=5
    )
    return _normalize_lines(lines)


def _normalize_lines(lines):
    if lines is None or len(lines) == 0:
        return np.array([]).reshape(0, 4)
    if lines.ndim == 3:
        return lines[:, 0, :]
    return lines


def is_vertical(x1, y1, x2, y2, angle_threshold=10):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dy == 0:
        return False
    return np.degrees(np.arctan(dx / dy)) < angle_threshold


def filter_vertical_lines(lines):
    lines = _normalize_lines(lines)
    if len(lines) == 0:
        return np.array([]).reshape(0, 4)
    mask = np.array([is_vertical(*line) for line in lines])
    return lines[mask]


def cluster_lines_by_x(lines, x_tolerance=5):
    """保留，但新流程不走此路径"""
    if len(lines) == 0:
        return []
    x_centers = [(line[0] + line[2]) / 2 for line in lines]
    sorted_idx = np.argsort(x_centers)
    clusters = []
    current = [x_centers[sorted_idx[0]]]
    for i in range(1, len(sorted_idx)):
        if x_centers[sorted_idx[i]] - current[-1] < x_tolerance:
            current.append(x_centers[sorted_idx[i]])
        else:
            clusters.append(np.mean(current))
            current = [x_centers[sorted_idx[i]]]
    clusters.append(np.mean(current))
    return clusters


def calculate_scale_ratio(x_positions):
    if len(x_positions) < 2:
        return None
    gaps = np.diff(x_positions)
    median_gap = np.median(gaps)
    if median_gap <= 0:
        return None
    return median_gap / config.SCALE_UNIT


def detect_scale_region(image):
    """检测分度板在图像中的大致区域"""
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)
