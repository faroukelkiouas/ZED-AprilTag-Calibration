import pyzed.sl as sl
import numpy as np
import cv2
from pupil_apriltags import Detector

# === AprilTag board setup ===
tag_size = 0.04       # 4cm per tag
tag_spacing = 0.01    # 1cm gap between tags
grid_cols, grid_rows = 5, 7  # tag grid layout

def generate_object_points(tag_ids):
    object_points = []
    for tag_id in tag_ids:
        row = tag_id // grid_cols
        col = tag_id % grid_cols
        x = col * (tag_size + tag_spacing)
        y = row * (tag_size + tag_spacing)
        z = 0
        corners = [
            [x, y, z],
            [x + tag_size, y, z],
            [x + tag_size, y + tag_size, z],
            [x, y + tag_size, z]
        ]
        object_points.append(corners)
    return np.array(object_points, dtype=np.float32)

# === Initialize AprilTag Detector ===
detector = Detector(families="tag36h11")

# === Data containers ===
all_objpoints = []
all_corners_left = []
all_corners_right = []
img_size = None

# === Initialize ZED Camera ===
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("❌ ZED camera failed to open.")
    exit()

image_left = sl.Mat()
image_right = sl.Mat()

print("📷 Press 'c' to capture calibration frame | 'q' to quit")

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)

        left = image_left.get_data()
        right = image_right.get_data()

        stacked = np.hstack((cv2.resize(left, (640, 360)), cv2.resize(right, (640, 360))))
        cv2.imshow("Stereo View", stacked)

        key = cv2.waitKey(1)
        if key == ord('c'):
            gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            detections_l = detector.detect(gray_l)
            detections_r = detector.detect(gray_r)

            if detections_l and detections_r:
                ids_l = np.array([d.tag_id for d in detections_l])
                ids_r = np.array([d.tag_id for d in detections_r])

                shared_ids = np.intersect1d(ids_l, ids_r)

                if len(shared_ids) == 0:
                    print("[-] No shared AprilTags between both cameras.")
                    continue

                corners_l = []
                corners_r = []
                for tag_id in shared_ids:
                    for det in detections_l:
                        if det.tag_id == tag_id:
                            corners_l.append(det.corners)
                    for det in detections_r:
                        if det.tag_id == tag_id:
                            corners_r.append(det.corners)

                object_points = generate_object_points(shared_ids)

                all_objpoints.append(object_points)
                all_corners_left.append(np.array(corners_l, dtype=np.float32))
                all_corners_right.append(np.array(corners_r, dtype=np.float32))

                img_size = gray_l.shape[::-1]
                print(f"[+] Captured frame #{len(all_objpoints)} with {len(shared_ids)} tags.")
            else:
                print("[-] AprilTags not detected in both views.")

        elif key == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()

# === Calibration ===
print("\n[*] Calibrating stereo system...")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
flags = cv2.CALIB_FIX_INTRINSIC

# Mono calibration (optional for validation)
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(all_objpoints, all_corners_left, img_size, None, None)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(all_objpoints, all_corners_right, img_size, None, None)

# Stereo calibration
retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objectPoints=all_objpoints,
    imagePoints1=all_corners_left,
    imagePoints2=all_corners_right,
    cameraMatrix1=mtx_l,
    distCoeffs1=dist_l,
    cameraMatrix2=mtx_r,
    distCoeffs2=dist_r,
    imageSize=img_size,
    criteria=criteria,
    flags=flags
)

print("\n✅ Stereo Calibration Complete")
print("Rotation Matrix (R):\n", R)
print("Translation Vector (T):\n", T)
