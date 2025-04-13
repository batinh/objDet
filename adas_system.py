import cv2
import numpy as np
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO('yolov8n.pt')

# Chế độ lựa chọn
LANE_ONLY = 1
SIGN_ONLY = 2
OBJECT_ONLY = 3

# Danh sách đối tượng
TRAFFIC_SIGNS = ['traffic light', 'stop sign']
OTHER_OBJECTS = ['person', 'car', 'motorbike', 'bus', 'truck', 'bicycle']

# Phát hiện vạch kẻ đường
def detect_lanes(frame):
    height, width = frame.shape[:2]

    # Chuyển ảnh sang grayscale và làm mờ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Áp dụng Canny Edge Detection
    edges = cv2.Canny(blur, 100, 200)

    # Vùng quan tâm (ROI) hình thang
    mask = np.zeros_like(edges)
    roi_polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, roi_polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform để tìm các đường thẳng
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100,
                            minLineLength=60, maxLineGap=50)

    # Vẽ các đường hợp lệ
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Chỉ giữ lại đường gần dọc (nghiêng ít)
            if abs(x2 - x1) < abs(y2 - y1) * 0.5:
                # Chỉ giữ đường ở nửa dưới ảnh (gần xe hơn)
                if y1 > height * 0.5 and y2 > height * 0.5:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    return frame
# Vẽ khung cho object
def draw_objects(frame, results, classes_to_detect):
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in classes_to_detect:
            continue
        conf = box.conf[0]
        if conf < 0.4:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0)
        if label in TRAFFIC_SIGNS:
            color = (0, 255, 255)  # Vàng cho biển báo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Hàm callback cho trackbar
def on_trackbar(val):
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# Chọn chế độ
print("Chọn chế độ:")
print("1 - Chỉ phát hiện vạch kẻ đường")
print("2 - Chỉ phát hiện biển báo")
print("3 - Chỉ phát hiện các đối tượng còn lại")
mode = int(input("Nhập lựa chọn (1/2/3): "))

# Đọc video
cap = cv2.VideoCapture("input_video.mp4")  # Thay bằng 0 nếu dùng webcam
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow("ADAS System")
cv2.createTrackbar("Position", "ADAS System", 0, total_frames - 1, on_trackbar)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Position", "ADAS System", current_frame)

    # Chế độ phát hiện
    if mode == LANE_ONLY:
        frame = detect_lanes(frame)
    else:
        results = model(frame)[0]
        if mode == SIGN_ONLY:
            frame = draw_objects(frame, results, TRAFFIC_SIGNS)
        elif mode == OBJECT_ONLY:
            frame = draw_objects(frame, results, OTHER_OBJECTS)

    cv2.imshow("ADAS System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()