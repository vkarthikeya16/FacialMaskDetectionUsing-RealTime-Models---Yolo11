import cv2
import time
import os
import csv
import threading
from datetime import datetime

from ultralytics import YOLO

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Windows-only sound (safe import)
try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False


# ================== CONFIG ==================
MODEL_PATH = "best3.pt"          # your trained model file
CONF_THRESH = 0.35               # try 0.35–0.50 to reduce duplicates
NMS_IOU = 0.60                   # NMS IoU for YOLO inference
MAX_DET = 50                     # limit detections
ALERT_COOLDOWN_SEC = 0.5
LOG_FILE = "maskguard_violations_log.csv"

WINDOW_TITLE = "MaskGuard | Real-Time Mask Detection"

# Tracker settings (stable Face IDs + smoother motion)
TRACK_IOU_MATCH = 0.35           # IoU needed to match the same face across frames
TRACK_MAX_AGE = 20               # frames to keep an ID without seeing it
SMOOTH_ALPHA = 0.70              # 0..1 (higher = smoother/less jitter)


# ================== LOAD MODEL ==================
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ✅ 3-class labels (MUST match the training class order)
# 0: Mask, 1: No Mask, 2: Incorrect Mask
CLASS_NAMES = {0: "Mask", 1: "No Mask", 2: "Incorrect Mask"}
try:
    model.model.names = CLASS_NAMES
except Exception:
    model.names = CLASS_NAMES

print("[INFO] Model loaded.")
print("[INFO] Model class names:", getattr(model, "names", getattr(model.model, "names", None)))


# ================== LOGGING ==================
def init_log_file(path):
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "total_faces",
                "mask_count",
                "no_mask_count",
                "incorrect_mask_count",
                "violation_count",
                "compliance_percent"
            ])
        print(f"[INFO] Created log file: {path}")
    else:
        print(f"[INFO] Using existing log file: {path}")

def log_violation(total_faces, mask_count, no_mask_count, incorrect_count, compliance):
    # log when there's any violation (No Mask or Incorrect Mask)
    violation_count = no_mask_count + incorrect_count
    if violation_count <= 0:
        return
    ts = datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            total_faces,
            mask_count,
            no_mask_count,
            incorrect_count,
            violation_count,
            f"{compliance:.1f}"
        ])
    print(
        f"[LOG] {ts} | Faces={total_faces}, Mask={mask_count}, "
        f"NoMask={no_mask_count}, Incorrect={incorrect_count}, "
        f"Violations={violation_count}, Compl={compliance:.1f}%"
    )

init_log_file(LOG_FILE)


# ================== SOUND ALERT ==================
last_alert_time = 0.0
alert_lock = threading.Lock()

def play_alert_beep():
    def _beep():
        if HAS_WINSOUND:
            try:
                winsound.Beep(1000, 200)
            except RuntimeError:
                try:
                    winsound.MessageBeep()
                except Exception:
                    pass
    threading.Thread(target=_beep, daemon=True).start()

def trigger_alert_if_needed(violation_count):
    global last_alert_time
    if violation_count <= 0:
        return
    now = time.time()
    with alert_lock:
        if now - last_alert_time >= ALERT_COOLDOWN_SEC:
            last_alert_time = now
            print("[ALERT] Mask violation detected! Playing sound.")
            play_alert_beep()


# ================== TRACKER + SMOOTHING ==================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)

def smooth_box(prev_box, new_box, alpha=0.7):
    # EMA: smoothed = alpha*prev + (1-alpha)*new (alpha high => smoother)
    if prev_box is None:
        return new_box
    px1, py1, px2, py2 = prev_box
    nx1, ny1, nx2, ny2 = new_box
    sx1 = int(alpha * px1 + (1 - alpha) * nx1)
    sy1 = int(alpha * py1 + (1 - alpha) * ny1)
    sx2 = int(alpha * px2 + (1 - alpha) * nx2)
    sy2 = int(alpha * py2 + (1 - alpha) * ny2)
    return (sx1, sy1, sx2, sy2)

class SmoothFaceTracker:
    """
    - Greedy IoU matching to keep stable IDs.
    - EMA smoothing per track to make motion smoother.
    """
    def __init__(self, iou_match=0.35, max_age=20, alpha=0.7):
        self.iou_match = iou_match
        self.max_age = max_age
        self.alpha = alpha
        self.next_id = 1
        # id -> {"box":..., "smooth_box":..., "age":0}
        self.tracks = {}

    def update(self, det_boxes_xyxy):
        # Age tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        assigned_ids = [-1] * len(det_boxes_xyxy)
        used_track_ids = set()

        # Match detections to existing tracks
        for di, dbox in enumerate(det_boxes_xyxy):
            best_tid = None
            best_iou = 0.0
            for tid, tinfo in self.tracks.items():
                if tid in used_track_ids:
                    continue
                tiou = iou_xyxy(dbox, tinfo["box"])
                if tiou > best_iou:
                    best_iou = tiou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_match:
                assigned_ids[di] = best_tid
                used_track_ids.add(best_tid)

                # Update raw + smooth
                self.tracks[best_tid]["box"] = dbox
                self.tracks[best_tid]["smooth_box"] = smooth_box(
                    self.tracks[best_tid].get("smooth_box"),
                    dbox,
                    alpha=self.alpha
                )
                self.tracks[best_tid]["age"] = 0

        # New tracks for unmatched detections
        for di, dbox in enumerate(det_boxes_xyxy):
            if assigned_ids[di] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box": dbox,
                    "smooth_box": dbox,
                    "age": 0
                }
                assigned_ids[di] = tid

        return assigned_ids

    def get_smooth_box(self, tid, fallback_box):
        t = self.tracks.get(tid)
        if t is None:
            return fallback_box
        return t.get("smooth_box") or fallback_box

tracker = SmoothFaceTracker(iou_match=TRACK_IOU_MATCH, max_age=TRACK_MAX_AGE, alpha=SMOOTH_ALPHA)


# ================== VIDEO + TKINTER GUI ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Make sure a camera is connected & not used by other apps.")

root = tk.Tk()
root.title(WINDOW_TITLE)

video_label = tk.Label(root)
video_label.pack()

stats_var = tk.StringVar()
stats_label = tk.Label(root, textvariable=stats_var, font=("Arial", 11))
stats_label.pack(pady=5)

def on_close():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

prev_time = time.time()
fps = 0.0


def process_frame():
    global prev_time, fps

    ret, frame = cap.read()
    if not ret:
        stats_var.set("Failed to grab frame from webcam.")
        root.after(50, process_frame)
        return

    h, w = frame.shape[:2]

    # YOLO inference
    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=NMS_IOU,
        max_det=MAX_DET,
        verbose=False
    )

    r = results[0]
    boxes = r.boxes

    annotated = frame.copy()

    det_xyxy = []
    det_cls = []
    det_conf = []

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
            det_xyxy.append(clamp_box((x1, y1, x2, y2), w, h))
            det_cls.append(int(c))
            det_conf.append(float(cf))

    # Stable Face IDs
    face_ids = tracker.update(det_xyxy) if det_xyxy else []

    total_faces = 0
    mask_count = 0
    nomask_count = 0
    incorrect_count = 0

    # Draw + count
    for idx, (box, c, cf, fid) in enumerate(zip(det_xyxy, det_cls, det_conf, face_ids), start=1):
        total_faces += 1

        # Use smoothed box for drawing
        sbox = tracker.get_smooth_box(fid, box)
        x1, y1, x2, y2 = clamp_box(sbox, w, h)

        # class name from our 3-class mapping
        label_name = CLASS_NAMES.get(int(c), str(int(c)))

        # ✅ Counts + colors for 3 classes
        if c == 0:
            mask_count += 1
            color = (0, 255, 0)          # green
        elif c == 1:
            nomask_count += 1
            color = (0, 0, 255)          # red
        else:  # c == 2
            incorrect_count += 1
            color = (0, 165, 255)        # orange

        text1 = f"Face {fid}"
        text2 = f"{label_name} {cf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, text1, (x1, max(y1 - 26, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(annotated, text2, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # ✅ Compliance: strictly "Mask" only
    compliance = (mask_count / total_faces * 100.0) if total_faces > 0 else 100.0

    # FPS
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = (1.0 / dt) if dt > 0 else fps

    violation_count = nomask_count + incorrect_count

    overlay_text = (
        f"FPS: {fps:.1f} | Faces: {total_faces} | "
        f"Mask: {mask_count} | No Mask: {nomask_count} | Incorrect: {incorrect_count} | "
        f"Violations: {violation_count} | Compliance: {compliance:.1f}%"
    )

    cv2.putText(
        annotated,
        overlay_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    stats_var.set(overlay_text)

    # Alert + log (trigger on No Mask OR Incorrect Mask)
    if violation_count > 0:
        trigger_alert_if_needed(violation_count)
        log_violation(total_faces, mask_count, nomask_count, incorrect_count, compliance)

    # Tkinter display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(annotated_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, process_frame)


process_frame()

try:
    root.mainloop()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released, application closed.")
