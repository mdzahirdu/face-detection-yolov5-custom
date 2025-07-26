import os
import cv2
import torch
import time
import pandas as pd

# === Paths ===
root_folder = r'your\path\to\image_folder'
output_folder = r'your\path\to\output_folder'
os.makedirs(output_folder, exist_ok=True)

# === Specific subjects to process (set to None to process all) ===
#specific_subjects = ["Subject_269", "Subject_270", "Subject_271", "Subject_272", "Subject_273"]
specific_subjects = None #it will process for all subject 

# === Device setup ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# === Load YOLOv5 model ===

model = torch.hub.load('your/path/to/yolov5', 'custom', path='your/path/to/model.pt', source='local')

model.to(device)
model.conf = 0.5
model.iou = 0.4

# === Results storage ===
results = []
subject_list = os.listdir(root_folder)

for subject_id in subject_list:
    if specific_subjects is not None and subject_id not in specific_subjects:
        continue

    subject_path = os.path.join(root_folder, subject_id)
    if not os.path.isdir(subject_path):
        continue

    print(f"\n🔹 Processing subject: {subject_id}")

    for device_type in ["Camera", "Mobile"]:
        device_path = os.path.join(subject_path, device_type)
        if not os.path.isdir(device_path):
            continue

        for filename in os.listdir(device_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(device_path, filename)
            print(f"  ➡️ Processing: {filename}")
            try:
                parts = filename.split("_")
                subject_number = ''.join(filter(str.isdigit, parts[0]))  # numeric ID
                inout = parts[1].lower()
                indoor_outdoor = "Indoor" if inout == "in" else "Outdoor"
                device_code = parts[2].lower()
                device_label = "Camera" if device_code == "c" else "Mobile"
                distance = parts[3]

                category = "Unknown"
                for part in parts[4:]:
                    if "propcat" in part.lower():
                        category = "Props"
                        break
                    elif "spoof" in part.lower():
                        category = "Spoof"
                        break
                    elif "live" in part.lower():
                        category = "Live"

            except Exception as e:
                print(f"⚠️ Filename parse error for {filename}: {e}")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Could not read image: {filename}")
                continue

            input_size = f"{img.shape[1]}x{img.shape[0]}"
            box_coords = "No face detected"
            cropped_size = "N/A"
            detection_status = "No Detection"

            try:
                start_time = time.time()
                result = model(img)
                predictions = result.xyxy[0]
                elapsed_time = round(time.time() - start_time, 2)

                if predictions is not None and len(predictions) > 0:
                    best_idx = predictions[:, 4].argmax()
                    x1, y1, x2, y2, conf, cls = predictions[best_idx].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cropped_face = img[y1:y2, x1:x2]

                    if cropped_face.size > 0:
                        save_path = os.path.join(output_folder, f"crop_{subject_number}_{filename}")
                        cv2.imwrite(save_path, cropped_face)
                        cropped_size = f"{x2 - x1}x{y2 - y1}"
                        box_coords = f"{x1},{y1},{x2},{y2}"
                        detection_status = "Detected"
                    else:
                        print(f"⚠️ Invalid crop area for {filename}")

                else:
                    print(f"⚠️ No face detected in {filename}")
                    elapsed_time = round(time.time() - start_time, 2)

            except Exception as e:
                print(f"⚠️ Detection error for {filename}: {e}")
                elapsed_time = -1

            results.append({
                "subject number": subject_number,
                "image name": filename,
                "indoor/outdoor": indoor_outdoor,
                "device": device_label,
                "distance": distance,
                "category": category,
                "detection": detection_status,
                "face box coordinate": box_coords,
                "detection time (s)": elapsed_time,
                "input image size": input_size,
                "cropped image size": cropped_size
            })

# === Save to Excel ===
df = pd.DataFrame(results)
excel_path = os.path.join(output_folder, "face_detection_details_yolov5.xlsx")
df.to_excel(excel_path, index=False)
print(f"\n✅ Excel saved at: {excel_path}")
print(f"✅ Done! Cropped images saved in: {output_folder}")
