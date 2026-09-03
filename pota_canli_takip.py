"""
============================================================
POTA CANLI TAKİP — Tek Dosya (RTSP + PostgreSQL)
============================================================
Canlı kamera akışından (RTSP) potaların işlem bölgesine giriş/çıkış
zamanlarını GERÇEK ZAMANLI takip eder ve PostgreSQL'e yazar.

Offline sürümden farkı: "tara sonra analiz et" değil, akıştan gelen
kareleri anında işleyip bölge doluluğunu sürekli izler; bir pota
bölgeye girince/çıkınca olayı o an veritabanına kaydeder.

------------------------------------------------------------
MANTIK (çok basit — kimlik takibi YOK)
------------------------------------------------------------
Her işlem bölgesi için tek soru: "şu an dolu mu, boş mu?"
  boş → dolu   : POTA GİRDİ  (giriş zamanı = şimdi)
  dolu → boş   : POTA ÇIKTI  (çıkış zamanı = son görüldüğü an)
Bölge kısa süre (buhar/anlık kayıp) boş görünürse ELENMEZ; ancak
EXIT_GRACE_SEC'ten uzun boş kalırsa gerçek çıkış sayılır.

------------------------------------------------------------
KURULUM
------------------------------------------------------------
    pip install ultralytics opencv-python psycopg2-binary

------------------------------------------------------------
BÖLGE TANIMI (önce bir kez yapılır)
------------------------------------------------------------
Her kameranın işlem bölgesi önceden çizilmelidir (bolge_editor.py ile).
Çizilen bölgeler veri/<video>_bolgeler.json dosyalarında saklanır ve bu
sistem KAMERA ADINA göre otomatik bulur. Ya da --bolge ile elle verilir.

------------------------------------------------------------
KULLANIM
------------------------------------------------------------
    # Canlı RTSP akışı
    python pota_canli_takip.py \
        --kaynak "rtsp://<user>:<pass>@<camera-host>:554/stream1" \
        --kamera "CAM-01"

    # Test için video dosyası (canlı akışmış gibi işler)
    python pota_canli_takip.py --kaynak "kayit.mp4" --kamera "CAM-01"

    # Bölgeyi elle vererek
    python pota_canli_takip.py --kaynak "rtsp://..." --kamera "X" \
        --bolge "986,255,1600,990"

    # Canlı izleme penceresiyle
    python pota_canli_takip.py --kaynak "rtsp://..." --kamera "X" --goster

ÇOKLU KAMERA: Her kamera için bu script'i AYRI çalıştırın (her biri kendi
sürecinde, kendi RTSP akışını dinler). Model her süreçte bir kez yüklenir.

Durdurmak: Ctrl+C  (o an işlemde olan potalar "hâlâ işlemde" olarak kaydedilir)
============================================================
"""

import argparse
import glob
import json
import os
import re
import signal
import sys
import time
from datetime import datetime

import psycopg2

from canli_kaynak import FrameGrabber


# ============================================================
# KONFİGÜRASYON
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "runs", "pota_yolo11l", "weights", "best.pt")
VERI_DIR = os.path.join(PROJECT_DIR, "veri")

CONF_THRESHOLD = 0.4       # tespit güven eşiği
PROCESS_FPS = 3.0          # saniyede kaç kare İŞLENSİN (pota yavaş; 2-4 yeterli, GPU'yu yormaz)
EXIT_GRACE_SEC = 15.0      # bölge bu kadar süre boş kalırsa "çıkış" sayılır (kısa kayıpları köprüler)
MIN_EVENT_SEC = 20.0       # bundan kısa süren "işlem" gürültü sayılır, kaydedilmez
                           # (RECONNECT_WAIT_SEC artık canli_kaynak.py'de)

# Veritabanı (POTA_DB_URL veya DB_* ortam değişkenleri; yoksa bunlar)
DEFAULT_DB = {
    "host": "localhost", "dbname": "pota_db",
    "user": os.environ.get("USER", "postgres"), "password": "", "port": "5432",
}

TRACK_COLORS = [(56,168,0),(0,152,255),(222,82,175),(255,178,29),(0,204,255)]


# ============================================================
# VERİTABANI
# ============================================================

SCHEMA = """
    CREATE TABLE IF NOT EXISTS pota_events (
        id SERIAL PRIMARY KEY,
        camera TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        entry_time TIMESTAMP NOT NULL,
        exit_time TIMESTAMP NOT NULL,
        duration_sec REAL NOT NULL,
        entry_partial BOOLEAN DEFAULT FALSE,
        exit_partial BOOLEAN DEFAULT FALSE,
        zone_no INTEGER DEFAULT 1,
        video_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ALTER TABLE pota_events ADD COLUMN IF NOT EXISTS zone_no INTEGER DEFAULT 1;
    CREATE INDEX IF NOT EXISTS idx_pota_events_camera ON pota_events (camera);
    CREATE INDEX IF NOT EXISTS idx_pota_events_entry ON pota_events (entry_time DESC);
"""


def db_connect():
    url = os.environ.get("POTA_DB_URL", "").strip()
    try:
        if url:
            conn = psycopg2.connect(url)
            hedef = url.split("@")[-1]
        else:
            conn = psycopg2.connect(
                host=os.environ.get("DB_HOST", DEFAULT_DB["host"]),
                dbname=os.environ.get("DB_NAME", DEFAULT_DB["dbname"]),
                user=os.environ.get("DB_USER", DEFAULT_DB["user"]),
                password=os.environ.get("DB_PASS", DEFAULT_DB["password"]),
                port=os.environ.get("DB_PORT", DEFAULT_DB["port"]),
            )
            hedef = f"{DEFAULT_DB['host']}/{DEFAULT_DB['dbname']}"
    except psycopg2.OperationalError as e:
        print(f"✘ PostgreSQL'e bağlanılamadı: {e}")
        sys.exit(1)
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    print(f"Veritabanı: PostgreSQL ({hedef})")
    return conn


def db_save_event(conn, camera, no, entry_t, exit_t, dur, zone_no, exit_partial=False):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO pota_events (camera, track_id, entry_time, exit_time, "
                "duration_sec, entry_partial, exit_partial, zone_no, video_file) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (camera, no, entry_t, exit_t, round(dur, 1), False, exit_partial, zone_no, "CANLI"),
            )
        conn.commit()
    except Exception as e:
        print(f"⚠ DB yazma hatası: {e}  (bağlantı yenilenecek)")
        try:
            conn.rollback()
        except Exception:
            pass


# ============================================================
# BÖLGE YÜKLEME (kamera adına göre, editörle çizilmiş)
# ============================================================

def _camera_from_name(name):
    m = re.search(r"NVR_(.+?)_\d{14}", name)
    if m:
        return m.group(1).replace("", "").strip()
    return None


def load_zones(camera, bolge_arg=None):
    """Kamera için işlem bölgelerini yükle: --bolge > editörle çizilmiş > hata."""
    if bolge_arg:
        # "x1,y1,x2,y2;x1,y1,x2,y2" biçiminde bir veya birden çok bölge
        zones = []
        for part in bolge_arg.split(";"):
            zones.append([int(v) for v in part.split(",")])
        return zones

    for f in sorted(glob.glob(os.path.join(VERI_DIR, "*_bolgeler.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("kaynak") != "elle_cizildi":
            continue
        ad = os.path.basename(f)[:-len("_bolgeler.json")]
        if _camera_from_name(ad) == camera:
            print(f"İşlem bölgesi yüklendi ({len(d['zones'])} bölge) ← {os.path.basename(f)}")
            return d["zones"]

    raise SystemExit(
        f"'{camera}' için çizilmiş işlem bölgesi bulunamadı.\n"
        f"Önce bolge_editor.py ile bu kameranın bölgesini çizin, ya da --bolge x1,y1,x2,y2 verin."
    )


def center_in_zone(cx, cy, zone):
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


# FrameGrabber artık canli_kaynak.py'de — hem bu dosya hem v2 (canlı modda) kullanır.

# ============================================================
# BÖLGE DURUM MAKİNESİ (giriş/çıkış tespiti)
# ============================================================

class ZoneState:
    """Bir işlem bölgesinin doluluk durumunu ve açık olayını tutar."""
    def __init__(self, zone_no):
        self.zone_no = zone_no
        self.occupied = False
        self.entry_time = None       # işleme giriş anı
        self.last_seen = None        # bölgede en son pota görüldüğü an
        self.event_count = 0         # bu bölgede kaç pota işlendi (numaralandırma için)


# ============================================================
# ANA CANLI DÖNGÜ
# ============================================================

_STOP = False
def _sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\n⏹  Durduruluyor… (açık işlemler kaydediliyor)")

signal.signal(signal.SIGINT, _sigint)


def main():
    global _STOP
    parser = argparse.ArgumentParser(description="Pota canlı takip (RTSP + PostgreSQL)")
    parser.add_argument("--kaynak", required=True, help="RTSP adresi veya test için video dosyası")
    parser.add_argument("--kamera", required=True, help="Kamera adı (bölge dosyası bununla bulunur)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bolge", default=None, help="Elle bölge: x1,y1,x2,y2 (çoklu için ; ile ayır)")
    parser.add_argument("--goster", action="store_true", help="Canlı izleme penceresi aç")
    parser.add_argument("--hiz", type=float, default=1.0,
                        help="Sadece dosya testinde: kaç kat hızlı oynatılsın (örn 4). Canlıda etkisiz.")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    if not os.path.exists(args.model):
        raise SystemExit(f"Model bulunamadı: {args.model}")

    zones = load_zones(args.kamera, args.bolge)
    for i, z in enumerate(zones, 1):
        w, h = z[2]-z[0], z[3]-z[1]
        tip = "yatay şerit" if w > h*2.2 else ("dikey şerit" if h > w*2.2 else "kutu")
        print(f"  Bölge {i}: {z}  ({tip})")

    conn = db_connect()
    model = YOLO(args.model)
    states = [ZoneState(i+1) for i in range(len(zones))]

    print(f"\n▶  Canlı takip başladı — kamera: {args.kamera}")
    print(f"   (işleme hızı ~{PROCESS_FPS:.0f} kare/sn, çıkış eşiği {EXIT_GRACE_SEC:.0f} sn)\n")

    grabber = FrameGrabber(args.kaynak)
    grabber.start()

    interval = 1.0 / PROCESS_FPS
    last_proc = 0.0
    frames_done = 0

    def finalize(st, exit_time, exit_partial=False):
        """Açık bir işlemi (bölge boşaldı) veritabanına yaz."""
        dur = (exit_time - st.entry_time).total_seconds()
        if dur >= MIN_EVENT_SEC:
            st.event_count += 1
            db_save_event(conn, args.kamera, st.event_count, st.entry_time, exit_time,
                          dur, st.zone_no, exit_partial=exit_partial)
            durum = "hâlâ işlemde" if exit_partial else "çıktı"
            print(f"  ✔ [Bölge {st.zone_no}] Pota {durum}: "
                  f"{st.entry_time.strftime('%H:%M:%S')} → {exit_time.strftime('%H:%M:%S')} "
                  f"({dur:.0f} sn) → kaydedildi")
        st.occupied = False
        st.entry_time = None

    try:
        while not _STOP:
            now = time.time()
            if now - last_proc < interval:
                time.sleep(0.01)
                continue
            last_proc = now

            frame = grabber.latest()
            if frame is None:
                time.sleep(0.1)
                continue

            T = datetime.now()   # gerçek (duvar saati) zaman
            res = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            frames_done += 1

            # tespit merkezleri
            centers = []
            if res.boxes is not None and len(res.boxes) > 0:
                centers = [(float(c[0]), float(c[1])) for c in res.boxes.xywh.cpu().numpy()]

            # her bölge için doluluk durumu güncelle
            for zi, (zone, st) in enumerate(zip(zones, states)):
                dolu = any(center_in_zone(cx, cy, zone) for cx, cy in centers)

                if dolu:
                    st.last_seen = T
                    if not st.occupied:
                        st.occupied = True
                        st.entry_time = T
                        print(f"  → [Bölge {st.zone_no}] Pota GİRDİ ({T.strftime('%H:%M:%S')})")
                elif st.occupied:
                    # bölge boş görünüyor; grace süresi doldu mu?
                    if (T - st.last_seen).total_seconds() > EXIT_GRACE_SEC:
                        finalize(st, st.last_seen)

            # izleme penceresi
            if args.goster:
                _ciz(cv2, frame, zones, states, args.kamera, T)
                disp = frame
                if frame.shape[1] > 1600:
                    disp = cv2.resize(frame, (1280, int(frame.shape[0]*1280/frame.shape[1])))
                cv2.imshow("Pota Canlı Takip", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # bağlantı durumu bilgisi (her ~30 sn)
            if frames_done % (int(PROCESS_FPS) * 30) == 0:
                acik = sum(1 for s in states if s.occupied)
                print(f"  · {frames_done} kare işlendi | bölgede işlemde: {acik} | "
                      f"akış: {'bağlı' if grabber.connected else 'KOPUK'}")

    finally:
        # kapanışta açık olan işlemleri "hâlâ işlemde" olarak kaydet
        for st in states:
            if st.occupied and st.entry_time:
                finalize(st, st.last_seen or datetime.now(), exit_partial=True)
        grabber.stop()
        if args.goster:
            import cv2
            cv2.destroyAllWindows()
        conn.close()
        print("\n⏹  Durduruldu. Kayıtlar panelde: http://localhost:8000")


def _ciz(cv2, frame, zones, states, camera, T):
    for zone, st in zip(zones, states):
        c = (0, 200, 0) if st.occupied else (120, 120, 120)
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), c, 3)
        etiket = f"Bolge {st.zone_no}: {'ISLEMDE' if st.occupied else 'bos'}"
        if st.occupied and st.entry_time:
            sure = int((T - st.entry_time).total_seconds())
            etiket += f" ({sure}sn)"
        cv2.putText(frame, etiket, (zone[0], max(30, zone[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (32, 32, 32), -1)
    cv2.putText(frame, f"{camera}  |  {T.strftime('%H:%M:%S')}  |  CANLI",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


if __name__ == "__main__":
    main()
