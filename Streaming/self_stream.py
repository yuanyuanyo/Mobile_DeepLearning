import Quartz
import numpy as np
import cv2
from ultralytics import YOLO
import os

def get_window_list():
    # 获取当前所有窗口的列表
    window_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    return window_list


def print_window_list(window_list):
    for window in window_list:
        window_number = window.get('kCGWindowNumber')
        owner_name = window.get('kCGWindowOwnerName', 'Unknown')
        window_name = window.get('kCGWindowName', 'Unnamed Window')
        bounds = window.get('kCGWindowBounds')
        pid = window.get('kCGWindowOwnerPID')
        layer = window.get('kCGWindowLayer')

        print(f"Window Number: {window_number}")
        print(f"Owner Name: {owner_name}")
        print(f"Window Name: {window_name}")
        print(f"Bounds: {bounds}")
        print(f"PID: {pid}")
        print(f"Layer: {layer}")
        print("-" * 40)


def get_window_image(window_id):
    # 捕获指定窗口的内容
    image = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming
    )

    if image is None:
        return None

    # 转换为 numpy 数组
    width = Quartz.CGImageGetWidth(image)
    height = Quartz.CGImageGetHeight(image)
    bpp = Quartz.CGImageGetBitsPerPixel(image) // 8
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image)

    data_provider = Quartz.CGImageGetDataProvider(image)
    data = Quartz.CGDataProviderCopyData(data_provider)
    buffer = np.frombuffer(data, dtype=np.uint8)

    # 确保数据大小与图像尺寸匹配
    expected_size = height * bytes_per_row
    if buffer.size != expected_size:
        print(f"Buffer size {buffer.size} does not match expected size {expected_size}")
        return None

    # 重塑缓冲区为图像数组
    img_array = np.zeros((height, width, bpp), dtype=np.uint8)
    for row in range(height):
        start = row * bytes_per_row
        end = start + (width * bpp)
        img_array[row, :width] = buffer[start:end].reshape(-1, bpp)

    return img_array


def detect_objects(frame, model, class_names, output_folder="output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    results = model(frame)

    detected = False
    detection_info = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            label = f"{class_names[class_id]}: {confidence:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            detected = True
            detection_info.append(f"{class_names[class_id]}: {confidence:.2f}, Coordinates: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

    if detected:
        output_image_path = os.path.join(output_folder, "detected_frame.png")
        cv2.imwrite(output_image_path, frame)

        output_txt_path = os.path.join(output_folder, "res.txt")
        with open(output_txt_path, 'a') as f:
            f.write(f"Detected objects in frame:\n")
            for info in detection_info:
                f.write(f"{info}\n")
            f.write("\n")

    return frame

def main():
    target_window_id = 22711
    target_owner_name = "Casting"

    window_list = get_window_list()

    # 检查窗口是否存在
    window_exists = any(
        window.get('kCGWindowNumber') == target_window_id and window.get('kCGWindowOwnerName') == target_owner_name
        for window in window_list
    )

    if not window_exists:
        print(f"未找到 Window Number: {target_window_id} 和 Owner Name: {target_owner_name} 的窗口")
        return

    # 加载您训练好的YOLOv8s模型
    model = YOLO("./yolomodel/yolov8s/yolov8n/train_v8n/weights/best.pt")

    # 定义类别名称
    class_names = ['Aligator Crack', 'Longitudinal Crack', 'Transverse Crack', 'Pothole', 'Repair']

    while True:
        frame = get_window_image(target_window_id)
        if frame is None:
            print("无法捕获窗口内容")
            break

        # 进行目标检测
        frame = detect_objects(frame, model, class_names)

        # 显示检测结果
        cv2.imshow(f"Window Capture - {target_owner_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()