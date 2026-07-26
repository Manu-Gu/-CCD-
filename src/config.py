"""配置参数"""

# ---- CCD相机参数 ----
CAMERA_INDEX = 0          # 摄像头索引
CAMERA_WIDTH = 1920       # 采集分辨率宽
CAMERA_HEIGHT = 1080      # 采集分辨率高

# ---- 分度板参数 ----
SCALE_UNIT = 0.5          # 每小格代表的实际值（mm）
MAJOR_TICK_INTERVAL = 10  # 长刻度线间隔（格数）
MIN_TICK_LENGTH = 20      # 最短刻度线像素长度阈值

# ---- 光斑检测参数 ----
SPOT_THRESHOLD = 200      # 光斑亮度阈值（0-255）
SPOT_MIN_AREA = 10        # 光斑最小面积（像素）
SPOT_MAX_AREA = 5000      # 光斑最大面积（像素）

# ---- 图像预处理参数 ----
GAUSSIAN_BLUR_SIZE = 5    # 高斯模糊核大小（奇数）
CLAHE_CLIP_LIMIT = 2.0    # CLAHE对比度限制
CLAHE_TILE_SIZE = (8, 8)  # CLAHE网格大小
CANNY_LOW = 50            # Canny边缘检测低阈值
CANNY_HIGH = 150          # Canny边缘检测高阈值

# ---- ROI区域 ----
ROI_ENABLED = False       # 是否启用ROI
ROI_X, ROI_Y = 0, 0
ROI_W, ROI_H = 800, 600
