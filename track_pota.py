"""
Pota Takip - Tek Kamera Prototipi
==================================
Eğitilmiş YOLO modelini ByteTrack ile video üzerinde çalıştırır,
her potanın kameraya giriş/çıkış zamanını ve görünme süresini
veritabanına kaydeder (varsayılan SQLite; POTA_DB_URL ortam
değişkeni ile PostgreSQL — bkz. db.py).

Kullanım:
    python3 track_pota.py --video "kayit.mp4" --camera "CAM-04"
    python3 track_pota.py --video "kayit.mp4"   # kamera adını dosya adından çıkarmayı dener
    python3 track_pota.py --video "kayit.mp4" --show   # canlı izleme penceresi açar

Gereksinim: pip install ultralytics
"""

import argparse
import os
import re
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

import db

# ============================================================
# KONFİGÜRASYON
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "runs", "pota_yolo11l", "weights", "best.pt")

CONF_THRESHOLD = 0.4      # tespit güven eşiği
EXIT_GRACE_SEC = 3.0      # bir ID bu kadar süre görünmezse "çıktı" sayılır
MIN_TRACK_SEC = 1.0       # bundan kısa süren track'ler gürültü sayılıp kaydedilmez
START_TOLERANCE_SEC = 1.0 # videonun ilk bu kadar saniyesinde görünen pota,
                          # kayıt başlamadan önce girmiş sayılır (giriş anı bilinmiyor)

# Track ID'lerine sabit renk atanır (BGR). Aynı pota hep aynı renkte görünür;
# renk ortada değişirse tracker onu kaybedip yeni ID vermiş demektir.
TRACK_COLORS = [
    (56, 168, 0), (0, 152, 255), (222, 82, 175), (255, 178, 29),
    (0, 204, 255), (140, 60, 255), (49, 210, 207), (10, 249, 72),
]

# NVR dosya adından kamera adı ve kayıt başlangıç zamanını çıkarma
# Örn: "NVR_CAM-04_20260625114352.mp4"
FILENAME_RE = re.compile(r"NVR_(.+?)_(\d{14})")


# ============================================================
# YARDIMCILAR
# ============================================================

def parse_filename(video_path):
    """Dosya adından (kamera adı, kayıt başlangıç zamanı) çıkarır; bulamazsa (None, None)."""
    m = FILENAME_RE.search(os.path.basename(video_path))
    if not m:
        return None, None
    camera = m.group(1).replace("", "").strip()
    start_time = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    return camera, start_time


# ============================================================
# ANA TAKİP DÖNGÜSÜ
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pota giriş/çıkış takibi (tek kamera)")
    parser.add_argument("--video", required=True, help="Video dosyası yolu")
    parser.add_argument("--camera", default=None, help="Kamera adı (verilmezse dosya adından çıkarılır)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Eğitilmiş model (best.pt) yolu")
    parser.add_argument("--show", action="store_true", help="Takibi canlı pencerede göster")
    parser.add_argument("--save", action="store_true",
                        help="İşaretlenmiş çıktı videosu kaydet (<video>_takip.mp4)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model bulunamadı: {args.model}\nColab'dan indirdiğiniz best.pt'yi bu klasöre koyun.")
    if not os.path.exists(args.video):
        raise SystemExit(f"Video bulunamadı: {args.video}")

    camera_from_file, video_start = parse_filename(args.video)
    camera = args.camera or camera_from_file
    if not camera:
        raise SystemExit("Kamera adı dosya adından çıkarılamadı, --camera ile belirtin.")
    if video_start is None:
        video_start = datetime.now()
        print("⚠ Dosya adında zaman bilgisi yok, video başlangıcı 'şimdi' kabul edildi.")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Kamera: {camera} | FPS: {fps:.1f} | Kare sayısı: {total_frames} | {width}x{height}")

    conn = db.connect()
    model = YOLO(args.model)

    # İşaretlenmiş çıktı videosu (sonuçları doğrulamak ve göstermek için)
    writer = None
    if args.save:
        out_path = os.path.splitext(args.video)[0] + "_takip.mp4"
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print(f"Çıktı videosu: {out_path}")

    # Aktif track'ler: track_id -> {"first": video_saniyesi, "last": video_saniyesi}
    active = {}
    frame_idx = 0
    saved_count = 0

    results = model.track(
        source=args.video,
        conf=CONF_THRESHOLD,
        tracker="bytetrack.yaml",
        stream=True,       # kareleri tek tek işle (belleği doldurmaz)
        persist=True,
        verbose=False,
    )

    def finalize(track_id, info, exit_partial=False):
        """
        Track'i olay olarak veritabanına yazar (çok kısa olanları eler).

        entry_partial: Pota video BAŞLADIĞINDA zaten kadrajdaydı — gerçek giriş
        anı kayıttan öncedir, bilinmiyor. Süre gerçeğinden KISA.
        exit_partial: Video bittiğinde pota hâlâ kadrajdaydı — gerçek çıkış bilinmiyor.

        Bu ayrım olmadan sistem eksik süreyi tam süre gibi kaydeder ve rapor
        sessizce yanlış olur.
        """
        nonlocal saved_count
        duration = info["last"] - info["first"]
        if duration < MIN_TRACK_SEC:
            return

        entry_partial = info["first"] <= START_TOLERANCE_SEC

        entry_t = video_start + timedelta(seconds=info["first"])
        exit_t = video_start + timedelta(seconds=info["last"])

        db.save_event(conn, camera, track_id, entry_t, exit_t, duration,
                      os.path.basename(args.video),
                      entry_partial=entry_partial, exit_partial=exit_partial)
        saved_count += 1

        flag = ""
        if entry_partial and exit_partial:
            flag = "  ⚠ giriş ve çıkış kayıt dışında — süre güvenilmez"
        elif entry_partial:
            flag = "  ⚠ video başında zaten kadrajdaydı — gerçek süre DAHA UZUN"
        elif exit_partial:
            flag = "  ⚠ video bitince hâlâ kadrajdaydı — gerçek süre DAHA UZUN"

        print(f"  ✔ Pota #{track_id}: {entry_t.strftime('%H:%M:%S')} → "
              f"{exit_t.strftime('%H:%M:%S')} ({duration:.1f} sn){flag}")

    for result in results:
        video_sec = frame_idx / fps
        frame_idx += 1

        seen_ids = set()
        if result.boxes is not None and result.boxes.id is not None:
            for tid in result.boxes.id.int().tolist():
                seen_ids.add(tid)
                if tid not in active:
                    active[tid] = {"first": video_sec, "last": video_sec}
                    print(f"  → Pota #{tid} girdi ({video_sec:.1f}. sn)")
                else:
                    active[tid]["last"] = video_sec

        # Uzun süredir görünmeyen track'leri kapat
        for tid in list(active.keys()):
            if tid not in seen_ids and video_sec - active[tid]["last"] > EXIT_GRACE_SEC:
                finalize(tid, active.pop(tid))

        # --- Görsel katman ---
        if args.show or writer is not None:
            frame = draw_overlay(result, active, video_sec, video_start, camera, saved_count)

            if writer is not None:
                writer.write(frame)

            if args.show:
                # Büyük kareleri ekrana sığdır
                disp = frame
                if width > 1600:
                    disp = cv2.resize(frame, (1280, int(height * 1280 / width)))
                cv2.imshow("Pota Takip", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # Video bitti: hâlâ kadrajda olan potaların gerçek çıkışı bilinmiyor
    for tid, info in active.items():
        finalize(tid, info, exit_partial=True)

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nTamamlandı. {saved_count} pota olayı kaydedildi.")
    if args.save:
        print(f"İşaretlenmiş video: {os.path.splitext(args.video)[0]}_takip.mp4")
    print("Kayıtları görmek için: python3 report.py")


def draw_overlay(result, active, video_sec, video_start, camera, saved_count):
    """
    Kutuların üstüne track ID'sini ve potanın kadrajda geçirdiği süreyi yazar.
    Süreyi kutunun üstünde canlı görmek, takibin doğru çalıştığını gözle
    doğrulamanın en hızlı yolu: ID atlıyorsa veya süre sıfırlanıyorsa
    tracker potayı kaybediyor demektir.
    """
    frame = result.orig_img.copy()

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        confs = result.boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)

            # Kadrajda geçirdiği süre
            elapsed = video_sec - active[tid]["first"] if tid in active else 0.0

            # Her ID'ye sabit bir renk — pota kadrajda gezerken rengi değişmiyorsa
            # tracker onu kaybetmemiş demektir
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"Pota #{tid}  {elapsed:.0f}sn  ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Üst bilgi paneli: kamera, gerçek saat, kadrajdaki pota sayısı, kaydedilen olay
    real_time = (video_start + timedelta(seconds=video_sec)).strftime("%H:%M:%S")
    header = f"{camera}  |  {real_time}  |  Kadrajda: {len(active)}  |  Kayit: {saved_count}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (32, 32, 32), -1)
    cv2.putText(frame, header, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


if __name__ == "__main__":
    main()
