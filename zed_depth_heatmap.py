import pyzed.sl as sl
import numpy as np
import cv2

# === Initialize ZED Camera ===
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # ✅ Compatible with GTX 1050 Ti
init_params.coordinate_units = sl.UNIT.MILLIMETER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ Failed to open ZED camera.")
    exit()

# Create containers
image = sl.Mat()
depth = sl.Mat()

print("📡 ZED started. Press ESC to quit.\n")

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        depth_np = depth.get_data()
        depth_np[np.isinf(depth_np)] = 0
        depth_np[np.isnan(depth_np)] = 0
        depth_np[depth_np > 5000] = 0

        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        cv2.imshow("🔥 ZED Depth Heatmap (ULTRA)", heatmap)

        if cv2.waitKey(1) & 0xFF == 27:
            break

zed.close()
cv2.destroyAllWindows()
print("✅ Camera closed.")
