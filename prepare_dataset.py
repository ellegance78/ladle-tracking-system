"""
Pota Takip Projesi - Label Studio Export'undan YOLO Dataset Hazırlama
========================================================================
Label Studio'nun YOLO export'unu (images/ + labels/, aynı dosya adıyla
eşleşen çift) kamera bazında dengeli train/val split'i yaparak
YOLO klasör yapısına dönüştürür. Export klasörü zaten görsel + etiket
çiftlerini içerdiği için ayrı bir kaynak görsel klasörüyle eşleştirme
gerekmez.
"""

import os
import re
import shutil
import random
import urllib.parse
from collections import defaultdict

random.seed(42)

EXPORT_DIR = "<proje-dizini>"
IMAGES_SRC = os.path.join(EXPORT_DIR, "images")
LABELS_SRC = os.path.join(EXPORT_DIR, "labels")
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

VAL_RATIO = 0.2
# Kamera adını, dosya adındaki ilk 14 haneli zaman damgasından önceki kısımdan çıkarır.
# Hem eski format (KAMERA_20260625133418.jpg) hem de frame-extract format
# (KAMERA_20260720162700_20260720164755_t000500.jpg) için çalışır.
CAMERA_RE = re.compile(r"NVR_(.*?)_\d{14}", re.IGNORECASE)


def read_classes():
    with open(os.path.join(EXPORT_DIR, "classes.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def extract_camera(filename):
    m = CAMERA_RE.search(urllib.parse.unquote(filename))
    if m:
        return m.group(1).strip()
    return "BILINMEYEN"


def main():
    classes = read_classes()
    print(f"Sınıflar: {classes}")

    images = {}
    for f in os.listdir(IMAGES_SRC):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            base = os.path.splitext(f)[0]
            images[base] = f

    labels = {}
    for f in os.listdir(LABELS_SRC):
        if f.endswith(".txt"):
            base = os.path.splitext(f)[0]
            labels[base] = f

    matched = sorted(set(images.keys()) & set(labels.keys()))
    print(f"Eşleşen görsel+etiket çifti: {len(matched)}")
    print(f"Sadece görselde olan (etiketsiz): {len(set(images) - set(labels))}")
    print(f"Sadece etikette olan (görselsiz): {len(set(labels) - set(images))}")

    camera_groups = defaultdict(list)
    for base in matched:
        cam = extract_camera(images[base])
        camera_groups[cam].append(base)

    print(f"\nToplam kamera sayısı: {len(camera_groups)}")
    for cam, bases in sorted(camera_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {cam}: {len(bases)}")

    train_bases, val_bases = [], []
    for cam, bases in camera_groups.items():
        bases = bases[:]
        random.shuffle(bases)
        n = len(bases)
        n_val = max(1, round(n * VAL_RATIO)) if n >= 3 else 0
        val_bases.extend(bases[:n_val])
        train_bases.extend(bases[n_val:])

    print(f"\nTrain: {len(train_bases)}  Val: {len(val_bases)}")

    for split, bases in [("train", train_bases), ("val", val_bases)]:
        img_dir = os.path.join(TARGET_DIR, split, "images")
        lbl_dir = os.path.join(TARGET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for base in bases:
            shutil.copy2(os.path.join(IMAGES_SRC, images[base]), os.path.join(img_dir, images[base]))
            shutil.copy2(os.path.join(LABELS_SRC, labels[base]), os.path.join(lbl_dir, base + ".txt"))

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    yaml_content = f"""# Pota Takip Projesi - YOLO11 Konfigürasyonu
path: {os.path.abspath(TARGET_DIR)}
train: train/images
val: val/images

nc: {len(classes)}
names:
{names_yaml}
"""
    with open(os.path.join(TARGET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\ndata.yaml oluşturuldu: {os.path.join(TARGET_DIR, 'data.yaml')}")
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
