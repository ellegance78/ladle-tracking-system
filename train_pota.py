"""
Pota Takip - YOLO11l Yerel Eğitim (Apple Silicon / MPS)
=========================================================
Doğruluk öncelikli ayarlar: YOLO11l (large), imgsz=960.
Kesintiye uğrarsa: python train_pota.py --resume ile kaldığı yerden devam eder.
"""

import argparse
import os

import torch
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(PROJECT_DIR, "dataset", "data.yaml")
RUNS_DIR = "<proje-dizini>"
RUN_NAME = "pota_arabasi+pota_takip-2"

# Önceki eğitimin ağırlığı (aynı 2 sınıf: pota, pota_arabasi) — sıfırdan COCO
# ağırlıklarıyla başlamak yerine buradan warm-start yapılır. 835 görselin 790'ını
# zaten görmüş olduğu için sadece 45 yeni kareye uyum sağlaması yeterli, çok daha
# az epoch'ta yakınsar.
WARM_START_WEIGHTS = "<proje-dizini>"
BASE_WEIGHTS = WARM_START_WEIGHTS if os.path.exists(WARM_START_WEIGHTS) else "yolo11l.pt"

CONFIG = {
    "data": DATA_YAML,
    "epochs": 150,
    "imgsz": 960,          # kaynak görüntüler 2560x1440, detay için 640 yerine 960
    "batch": 4,            # 24GB birleşik bellekte 960px + YOLO11l için güvenli üst sınır
                           # (batch 8'de 21.4GB'a çıkıp belleği tıkadı; ultralytics küçük
                           # batch'i nbs=64'e gradyan biriktirmeyle tamamladığı için
                           # doğruluk kaybı beklenmez)
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "patience": 30,        # 30 epoch iyileşme olmazsa erken dur
    "cache": "ram",        # 835 görsel RAM'e sığar, epoch'lar arası disk okumasını kaldırır
    "project": RUNS_DIR,
    "name": RUN_NAME,
    "exist_ok": True,
    "plots": True,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Yarıda kalan eğitime devam et")
    args = parser.parse_args()

    if args.resume:
        last = os.path.join(RUNS_DIR, RUN_NAME, "weights", "last.pt")
        model = YOLO(last)
        model.train(resume=True)
    else:
        print(f"Başlangıç ağırlığı: {BASE_WEIGHTS}")
        model = YOLO(BASE_WEIGHTS)
        model.train(**CONFIG)

    metrics = model.val()
    print("\n=== SONUÇ ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"En iyi ağırlık: {os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
