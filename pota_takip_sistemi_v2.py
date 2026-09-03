"""
============================================================
POTA TAKİP SİSTEMİ — Tek Dosya (PostgreSQL)
============================================================
Eğitilmiş YOLO modelini ByteTrack ile video üzerinde çalıştırır, her potanın
kameraya giriş/çıkış zamanını ve görünme süresini PostgreSQL'e kaydeder.

Model 2 sınıflıdır (pota, pota_arabasi). DB'ye sadece POTA olayları yazılır;
pota_arabasi tespiti yalnızca yardımcı sinyal olarak kullanılır: pota buhar/
duman yüzünden geçici görünmez olduğunda, altındaki araba hâlâ kadrajdaysa
track kapatılmaz, çıkış toleransı uzatılır (bkz. EXIT_GRACE_SEC_ASSISTED).

Takip, veritabanı ve raporlama tek dosyada — başka .py dosyasına ihtiyaç yok.

------------------------------------------------------------
KURULUM
------------------------------------------------------------
    pip install ultralytics opencv-python psycopg2-binary

------------------------------------------------------------
VERİTABANI BAĞLANTISI
------------------------------------------------------------
Bağlantı bilgisi iki yoldan verilebilir:

  1) Tek satır URL (önerilen):
     export POTA_DB_URL="postgresql://kullanici:sifre@sunucu:5432/potadb"

  2) Ayrı ayrı ortam değişkenleri:
     export DB_HOST=89.252.153.101
     export DB_NAME=ladle_db
     export DB_USER=postgres
     export DB_PASS=sifre
     export DB_PORT=5432

Hiçbiri verilmezse aşağıdaki DEFAULT_DB değerleri kullanılır.

------------------------------------------------------------
KULLANIM
------------------------------------------------------------
    # Bir videoyu işle ve olayları PostgreSQL'e yaz
    python pota_takip_sistemi.py track --video "kayit.mp4"

    # İşaretlenmiş çıktı videosu da üret
    python pota_takip_sistemi.py track --video "kayit.mp4" --save

    # Canlı izleme penceresiyle
    python pota_takip_sistemi.py track --video "kayit.mp4" --show

    # Kaydedilen olayları listele
    python pota_takip_sistemi.py report
    python pota_takip_sistemi.py report --camera "CAM-09"
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
from collections import deque
from datetime import datetime, timedelta

import psycopg2

LIVE_SOURCE_PREFIXES = ("rtsp://", "http://", "https://", "tcp://", "udp://")

# Canlı moddan Ctrl+C ile çıkış: bare "except KeyboardInterrupt" bu ortamda
# (ultralytics/torch kendi sinyal davranışına sahip) güvenilir çalışmadı —
# pota_canli_takip.py'deki gibi açık signal handler + bayrak kullanılıyor.
_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\n⏹  Durduruluyor… (açık track'ler 'hâlâ işlemde' olarak kaydedilecek)")


signal.signal(signal.SIGINT, _handle_sigint)


# ============================================================
# KONFİGÜRASYON
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "runs", "pota_arabasi_v2", "weights", "best.pt")
VERI_DIR = os.path.join(PROJECT_DIR, "veri")   # bolge_editor.py'nin kaydettiği kamera-bazlı kalibrasyon

# Model sınıfları (eğitimde bu sırayla tanımlı — bkz. dataset/data.yaml)
CLASS_POTA = 0
CLASS_ARABASI = 1

DEFAULT_STRIDE = 4         # her N. kareyi işle (20fps kaynakta stride=4 → 5fps efektif, 0.2sn çözünürlük;
                           # potanın giriş/çıkışı saniyeler sürdüğünden bu çözünürlük fazlasıyla yeterli
                           # ve MPS/CPU üzerindeki işlem yükünü ~stride kat azaltır)

CONF_THRESHOLD = 0.4       # tespit güven eşiği
MIN_VALID_Y_FRAC = 0.5     # kadrajın üst yarısı vinç/tavan yapısı — pota fiziksel olarak oraya
                           # çıkamaz. Tam video analizinde ORADA (y<698px, 2048x1536 kadrajda)
                           # tekrarlayan bir köşe yansıması + saçılmış kıvılcım gürültüsü
                           # bulundu (253 vakası: 885sn'lik SAHTE bir pota'ya kadar zincirlendi),
                           # gerçek pota/kapı aktivitesi hep y>824'te — aradaki boşluğa göre
                           # y merkezinin frame yüksekliğinin bu oranının altındaki her tespit
                           # (CLASS_POTA için) tamamen göz ardı edilir, ham track'e bile girmez.
STABLE_POINT_N = 5         # bir ham track başlarken/biterken konum ORTALAMASI için kullanılan
                           # tespit sayısı. Tek kareye (özellikle tekrar-belirmenin İLK karesine)
                           # bakmak riskli: tracker o anda geçici olarak yanlış/kaymış bir kutu
                           # üretebiliyor (9/35 vakası: 0.18 eşiğini 0.038 farkla kaçırmıştı).
                           # İlk/son birkaç karenin ortalaması bu tek karelik gürültüyü filtreler.
LIVE_PROCESS_FPS = 3.0     # canlı (RTSP) modda saniyede kaç kare İŞLENSİN — pota_canli_takip.py'de
                           # de aynı değer kullanılıyor, GPU'yu yormadan yeterli çözünürlük sağlıyor.
                           # Dosya modunda etkisi yok (--stride kullanılır).
LIVE_CONNECT_TIMEOUT_SEC = 30.0  # canlı kaynağa bağlanıp ilk kareyi almak için azami bekleme
EXIT_GRACE_SEC = 3.0       # bir pota ID'si bu kadar süre görünmezse "çıktı" sayılır
EXIT_GRACE_SEC_ASSISTED = 15.0  # pota kaybolduğu anda pota_arabasi hâlâ yakında
                                 # görünüyorsa bu daha uzun tolerans kullanılır
                                 # (pota buhar/duman arkasında geçici kayboldu, araba yerinde duruyor)
ASSIST_DIST_FRAC = 0.15    # pota_arabasi'nin, kaybolan potanın son konumuna olan mesafesi
                           # kadraj köşegeninin bu oranından yakın olmalı ki "aynı yerde" sayılsın
MIN_TRACK_SEC = 180.0      # birleştirme SONRASI bundan kısa kalan potalar gürültü sayılır.
                           # Tam video testinde 60sn'de bile 1-2.5dk'lık kalıntılar geçiyordu — çoğu ya
                           # potanın durduğu ana bölgenin dışında (farklı obje/yanlış pozitif) ya da
                           # birleşme eşiğini birkaç saniye kaçıran parçalar. Gerçek olaylar 373sn'den
                           # başlıyor, aradaki geniş boşluğa (180sn) konuldu.
START_TOLERANCE_SEC = 1.0  # videonun ilk bu kadar saniyesinde görünen pota,
                           # kayıt başlamadan önce girmiş sayılır (giriş anı bilinmiyor)

# --- Konum bazlı birleştirme ---
# Çelikhanede pota görünümü döküm evresine göre tamamen değişiyor (gri↔kızgın turuncu),
# bu yüzden görünüm bazlı ReID güvenilmez. Onun yerine KONUM kullanıyoruz: bir track
# bitip kısa süre sonra AYNI konumda yeni track başlarsa, bu aynı potadır → birleştir.
MERGE_TIME_GAP = 150.0     # track bitişi ile yeni track başlangıcı arası en fazla bu kadar sn olabilir.
                           # 30sn iken 76/80 ve 103/130 vakaları kaçmıştı: teşhis videosunda görüldü ki
                           # şiddetli reaksiyon alevi kamerayı ~65-70sn boyunca TAMAMEN beyaza kesiyor
                           # (pota VE arabasi ikisi de kayboluyor) — bu yüzden görsel sinyale güvenilemez,
                           # 120sn iken de track #235 129sn'lik boşlukla kıl payı kaçmıştı, pay artırıldı.
                           # sadece konum+zaman bazlı eşiği gözlemlenen alev süresinin üstüne çıkardık.
MERGE_DIST_FRAC = 0.18     # iki track'in birleşmesi için merkezleri kadraj köşegeninin en fazla bu oranı kadar uzak olmalı.
                           # NOT: 9/35 vakasında bu eşik tek karelik konum gürültüsü yüzünden kıl
                           # payı (0.218) kaçırılmıştı — çözüm eşiği büyütmek değil, STABLE_POINT_N
                           # ile ilk/son birkaç karenin ORTALAMASINI kullanmak oldu (bkz. yukarı).
OVERLAP_TOLERANCE_SEC = 8.0  # Gerçek DB verisinde görüldü: kıvılcım/döküm anında tracker aynı fiziksel
                              # potaya bir anlığına HAYALET ikinci bir ID veriyor (tek kare çakışma).
                              # Böyle KISA çakışmalar konuma bakılarak yine birleştirilir; sadece bu
                              # süreden UZUN, gerçek eşzamanlı varlık farklı pota sayılır — AMA BUNUN
                              # İSTİSNASI VAR, bkz. GATE_LEFT/GATE_RIGHT.

# --- Kapı doğrulaması (giriş yönü = çıkış yönü) ---
# Pota kadraja ekranın iki alt köşesindeki raylı boşluklardan (kapı) girip çıkar.
# Bir ham track bu kapılardan BİRİNİN dışında (kadrajın ortasında) kayboluyorsa,
# pota aslında kadrajdan çıkmamış, sadece duman/kıvılcım/buhar yüzünden bir anlığına
# görünmez olmuştur — bu durumda OVERLAP_TOLERANCE_SEC sınırı uygulanmaz, süre farkı
# ne olursa olsun konuma bakılarak birleştirilir (103/149 vakası: 43sn çakışma, ikisi
# de kadraj ortasında, aynı potaydı).
# Kayboluş noktası gerçekten bir kapıdaysa, bu GERÇEK bir çıkıştır ve yeni track yalnızca
# AYNI kapıdan giriyorsa aynı pota sayılır — sağdan çıkan sağdan girmeli, soldan çıkan
# soldan girmeli; ters kapıdan "giren" asla aynı pota sayılmaz.
# Koordinatlar TEST-14SAAT-v2-full kamerasının 2048x1536 görüntüsüne göre kalibre edildi,
# oranlı (fraksiyon) olarak tutulur ki farklı çözünürlükte de orantılı kalsın.
GATE_LEFT_FRAC = (0.0, 0.41, 0.586, 1.0)     # (x_min, x_max, y_min, y_max) — kare/fraksiyon
GATE_RIGHT_FRAC = (0.769, 1.0, 0.563, 1.0)
# --- Geçişli düzen (soldan gir / sağdan çık) ---
# Sahada doğrulandı: pota girdiği kapıdan çıkmak ZORUNDA DEĞİL, bir kameradan
# soldan girip sağdan çıkabiliyor. Bu yüzden "aynı kapıdan dönmeli" kuralı kalktı:
#   - bir parça GERÇEKTEN kapıdan çıktıysa o pota gitmiştir, bir daha birleşmez
#   - GERÇEKTEN kapıdan yeni bir giriş varsa o BAŞKA bir potadır, birleşmez
#   - ikisi de değilse (kadraj ortasında kayboldu) duman/alev örtmesidir, birleşir
# Kamera bazında kapatmak için GECISLI_ISTISNA'ya kamera adı + False yazılır.
GECISLI_DUZEN = True
GECISLI_ISTISNA = {}       # örn. {"CAM-10": False} -> o kamerada eski "aynı kapı" kuralı

# Bir parçanın kapı dikdörtgeni içinde bitmesi TEK BAŞINA "çıktı" demek değildir:
# kapı sınırında duran ya da yeni girmiş pota da orada görünür. Gerçek çıkış/giriş
# sayılması için parçanın DIŞA/İÇE doğru en az bu kadar (kare genişliğinin oranı)
# yatay yol almış olması gerekir. Bu olmadan kapı sınırındaki piksel titremesi
# aynı potayı iki ayrı kayda bölüyordu.
GATE_DIR_MIN_DX_FRAC = 0.02

GATE_NONE_MAX_GAP = 900.0  # kapının dışında (mid-frame) kaybolma durumunda üst sınır (15dk).
                           # Doğrulanan en uzun gerçek duman/buhar vakası 852sn (233/235) idi;
                           # sınırsız bırakınca kadrajın merkezî bekleme noktasına GÜN BOYU farklı
                           # zamanlarda gelen tamamen farklı potalar (muhtemelen vinçle inip
                           # kalkıyorlar, kapılara hiç değmiyorlar) yanlışlıkla tek pota sayıldı
                           # (bir vakada 7 saate kadar birleşti) — bu üst sınır o hatayı önler.

# --- Akıllı birleştirme: pota_arabasi sürekliliği ---
# Boşluk MERGE_TIME_GAP'i aşsa bile, o boşluk boyunca pota_arabasi AYNI yerde hiç
# kaybolmadan durduysa (sadece pota duman/ışıktan görünmüyordu), süre sınırı olmadan
# birleştir — araba yerinden ayrılmadıysa başka bir potanın gelmiş olması imkansız.
# Araba da o boşlukta kayboldan (yer tamamen boşaldıysa) sıkı MERGE_TIME_GAP geçerli.
ARABASI_GAP_TOLERANCE = 20.0   # arabasi'nin KENDİ görünürlüğünde tolere edilen maks. boşluk
                                # (bundan uzun kaybolursa "sürekli duruyordu" denemez)
ARABASI_COVER_SLACK = 5.0      # arabasi span'ının, pota'nın kapanış/açılış anını tam kapsaması
                                # gerekmez — bu kadar saniyelik pay bırakılır

# Veritabanı bağlantısı verilmezse kullanılacak varsayılanlar
# Pota takibi için ayrı bir veritabanı (diğer sistemlerden bağımsız)
DEFAULT_DB = {
    "host": "localhost",
    "dbname": "pota_db",
    "user": os.environ.get("USER", "postgres"),
    "password": "",
    "port": "5432",
}

# Track ID'lerine sabit renk atanır (BGR). Aynı pota hep aynı renkte görünür;
# renk ortada değişirse tracker onu kaybedip yeni ID vermiş demektir.
TRACK_COLORS = [
    (56, 168, 0), (0, 152, 255), (222, 82, 175), (255, 178, 29),
    (0, 204, 255), (140, 60, 255), (49, 210, 207), (10, 249, 72),
]

# NVR dosya adından kamera adı ve kayıt başlangıç zamanını çıkarma
# Örn: "NVR_CAM-04_20260625114352.mp4"
FILENAME_RE = re.compile(r"NVR_(.+?)_(\d{14})")

# NVR formatında olmayan (örn. birden çok parçadan birleştirilmiş) videolar için:
# yol içinde YYYY-MM-DD geçiyorsa o günün gece yarısını video başlangıcı kabul et.
# Böylece giriş/çıkış saatleri "video'nun kaçıncı dakikası" olarak okunabilir hale gelir
# (gerçek NVR saati bilinmediğinden gerçek saat DEĞİL, video pozisyonudur).
DATE_IN_PATH_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# bolge_editor.py'nin ürettiği dosya adları da NVR kalıbını kullanır (pota_canli_takip.py
# ile aynı eşleştirme mantığı — 27 kameranın her biri kendi dosyasını bu şekilde bulur)
CAMERA_FROM_FILENAME_RE = re.compile(r"NVR_(.+?)_\d{14}")


def _camera_from_zone_filename(name):
    m = CAMERA_FROM_FILENAME_RE.search(name)
    if m:
        return m.group(1).replace("", "").strip()
    return None


def load_camera_config(camera, width, height):
    """
    Kamera adına göre veri/<...>_bolgeler.json'dan (bolge_editor.py ile çizilmiş)
    kapı ve geçersiz-bölge dikdörtgenlerini yükler.

    Bulunamazsa None döner — çağıran taraf GATE_LEFT_FRAC/GATE_RIGHT_FRAC/
    MIN_VALID_Y_FRAC hardcoded varsayılanlarına düşer (mevcut, doğrulanmış
    TEST-14SAAT-v2-full davranışı hiç değişmez).

    Dönen (bulunursa): {"gate_left": (x1,y1,x2,y2) | None,
                        "gate_right": (x1,y1,x2,y2) | None,
                        "invalid_zones": [(x1,y1,x2,y2), ...]}
    — piksel cinsinden, ÇİZİM YAPILAN referans karenin boyutuyla GERÇEK
    kare boyutu farklıysa orantılı olarak yeniden ölçeklenir.
    """
    # Aynı kameraya ait BİRDEN ÇOK bölge dosyası olabilir (kamera yeniden
    # konumlandırılıp yeni referans kare alındığında eskisi diskte kalıyor).
    # Eskiden burada sorted(glob)[ilk] alınıyordu; dosya adları tarih içerdiği
    # için bu EN ESKİ dosyayı seçiyordu ve yeni çizilen kapılar sessizce
    # yok sayılıyordu (CAM-09 bu yüzden kapısız çalışıyordu).
    # Doğru tercih sırası: (1) referans karesi hâlâ duran dosya, (2) en yeni.
    adaylar = []
    for fp in glob.glob(os.path.join(VERI_DIR, "*_bolgeler.json")):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        if data.get("kaynak") != "elle_cizildi":
            continue
        base = os.path.basename(fp)[:-len("_bolgeler.json")]
        if _camera_from_zone_filename(base) != camera:
            continue
        frame_var = os.path.exists(os.path.join(VERI_DIR, base + "_frame.jpg"))
        adaylar.append((frame_var, os.path.getmtime(fp), fp, data))

    for _frame_var, _mtime, _fp, data in sorted(adaylar, reverse=True):
        zones = data.get("zones", [])
        tipler = data.get("tipler") or ["islem"] * len(zones)
        ref_w = data.get("width") or width
        ref_h = data.get("height") or height
        sx, sy = width / ref_w, height / ref_h

        def scaled(z):
            return (z[0] * sx, z[1] * sy, z[2] * sx, z[3] * sy)

        cfg = {"gate_left": None, "gate_right": None, "invalid_zones": []}
        for z, t in zip(zones, tipler):
            if t == "kapi_sol":
                cfg["gate_left"] = scaled(z)
            elif t == "kapi_sag":
                cfg["gate_right"] = scaled(z)
            elif t == "gecersiz":
                cfg["invalid_zones"].append(scaled(z))
        return cfg
    return None


def _point_in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


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
    CREATE INDEX IF NOT EXISTS idx_pota_events_camera ON pota_events (camera);
    CREATE INDEX IF NOT EXISTS idx_pota_events_entry ON pota_events (entry_time DESC);
    -- Eski tabloda zone_no yoksa ekle (çift taraflı düzenek için)
    ALTER TABLE pota_events ADD COLUMN IF NOT EXISTS zone_no INTEGER DEFAULT 1;
"""


def db_connect():
    """
    PostgreSQL'e bağlanır ve tabloyu (yoksa) oluşturur.
    Bağlantı: önce POTA_DB_URL, sonra DB_* ortam değişkenleri, sonra DEFAULT_DB.
    """
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
            hedef = f"{os.environ.get('DB_HOST', DEFAULT_DB['host'])}/{os.environ.get('DB_NAME', DEFAULT_DB['dbname'])}"
    except psycopg2.OperationalError as e:
        print(f"✘ PostgreSQL'e bağlanılamadı: {e}")
        print("  POTA_DB_URL veya DB_HOST/DB_NAME/DB_USER/DB_PASS ortam değişkenlerini kontrol edin.")
        sys.exit(1)

    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    print(f"Veritabanı: PostgreSQL ({hedef})")
    return conn


def db_save_event(conn, camera, track_id, entry_time, exit_time, duration_sec,
                  video_file, entry_partial=False, exit_partial=False, zone_no=1):
    """Yeni olay ekler, eklenen satırın id'sini döner (sonradan UPDATE edebilmek için)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pota_events "
            "(camera, track_id, entry_time, exit_time, duration_sec, "
            " entry_partial, exit_partial, zone_no, video_file) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (camera, track_id, entry_time, exit_time, round(duration_sec, 1),
             entry_partial, exit_partial, zone_no, video_file),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def db_update_event(conn, event_id, exit_time, duration_sec, exit_partial):
    """Devam eden (birleşen) bir olayın çıkış zamanını/süresini günceller — canlı takip için."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pota_events SET exit_time = %s, duration_sec = %s, exit_partial = %s WHERE id = %s",
            (exit_time, round(duration_sec, 1), exit_partial, event_id),
        )
    conn.commit()


def db_delete_by_video(conn, video_file):
    """Bir videonun eski kayıtlarını sil — aynı video tekrar işlenince çift kayıt olmasın."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM pota_events WHERE video_file = %s", (video_file,))
    conn.commit()


def db_fetch_events(conn, camera=None):
    sql = ("SELECT camera, track_id, entry_time, exit_time, duration_sec, "
           "entry_partial, exit_partial, video_file FROM pota_events")
    params = ()
    if camera:
        sql += " WHERE camera = %s"
        params = (camera,)
    sql += " ORDER BY entry_time"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


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


def draw_overlay(cv2, result, active, video_sec, video_start, camera, saved_count):
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
        clss = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().numpy()

        for box, tid, cls_id, conf in zip(boxes, ids, clss, confs):
            x1, y1, x2, y2 = map(int, box)

            if cls_id == CLASS_ARABASI:
                # Yardımcı sinyal: ince gri kutu, sadece görsel doğrulama için
                cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)
                cv2.putText(frame, "araba", (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
                continue

            elapsed = video_sec - active[tid]["first"] if tid in active else 0.0
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"Pota #{tid}  {elapsed:.0f}sn  ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    real_time = (video_start + timedelta(seconds=video_sec)).strftime("%H:%M:%S")
    header = f"{camera}  |  {real_time}  |  Kadrajda: {len(active)}  |  Kayit: {saved_count}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (32, 32, 32), -1)
    cv2.putText(frame, header, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


# ============================================================
# TAKİP (track komutu)
# ============================================================

def cmd_track(args):
    # Ağır kütüphaneler sadece bu komutta yüklenir (report çok daha hızlı açılır)
    import cv2
    from ultralytics import YOLO

    is_live = args.live or args.video.lower().startswith(LIVE_SOURCE_PREFIXES)

    if not os.path.exists(args.model):
        raise SystemExit(f"Model bulunamadı: {args.model}")
    if not is_live and not os.path.exists(args.video):
        raise SystemExit(f"Video bulunamadı: {args.video}")

    grabber = None
    if is_live:
        # Canlı akışta dosya adından kamera/zaman çıkarımı anlamsız — kamera adı
        # zorunlu, zaman "şimdi"den başlar (giriş/çıkışlar GERÇEK saat olur).
        if not args.camera:
            raise SystemExit("Canlı kaynakta --camera zorunlu (dosya adından çıkarım yapılamaz).")
        camera = args.camera
        video_start = datetime.now()
        if args.save:
            print("⚠ Canlı modda --save yok sayılıyor (süresiz akışı tek dosyaya yazmak uygun değil).")
            args.save = False
        if args.resume:
            print("⚠ Canlı modda --resume anlamsız, yok sayılıyor.")
            args.resume = False

        from canli_kaynak import FrameGrabber
        print(f"Canlı kaynağa bağlanılıyor: {args.video}")
        grabber = FrameGrabber(args.video)
        grabber.start()
        t0 = time.time()
        first_frame = None
        while time.time() - t0 < LIVE_CONNECT_TIMEOUT_SEC:
            first_frame = grabber.latest()
            if first_frame is not None:
                break
            time.sleep(0.2)
        if first_frame is None:
            grabber.stop()
            raise SystemExit(f"Canlı kaynaktan {LIVE_CONNECT_TIMEOUT_SEC:.0f}sn içinde kare alınamadı.")
        height, width = first_frame.shape[:2]
        fps = 25.0            # sadece kozmetik yazdırma/VideoWriter için — merge mantığı wall-clock kullanır
        total_frames = 0
        print(f"Kamera: {camera} | {width}x{height} (canlı akış)")
    else:
        camera_from_file, video_start = parse_filename(args.video)
        camera = args.camera or camera_from_file
        if not camera:
            raise SystemExit("Kamera adı dosya adından çıkarılamadı, --camera ile belirtin.")
        if video_start is None:
            # NVR formatında zaman bilgisi yok (örn. birleştirilmiş video). Yol içinde
            # YYYY-MM-DD varsa o günün gece yarısını "video'nun 0. saniyesi" kabul et —
            # böylece kaydedilen giriş/çıkış saatleri GERÇEK saat değil, VİDEO POZİSYONU olur.
            m = DATE_IN_PATH_RE.search(args.video)
            if m:
                y, mo, d = map(int, m.groups())
                video_start = datetime(y, mo, d)
                print(f"⚠ Dosya adında NVR zaman bilgisi yok. Video başlangıcı {video_start.strftime('%Y-%m-%d')} "
                      f"00:00:00 kabul edildi — giriş/çıkış saatleri GERÇEK SAAT değil, VİDEONUN İÇİNDEKİ "
                      f"POZİSYONDUR (örn. 03:45:12 = videonun 3sa 45dk 12sn'si).")
            else:
                video_start = datetime.now()
                print("⚠ Dosya adında zaman bilgisi yok, video başlangıcı 'şimdi' kabul edildi.")

        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"Kamera: {camera} | FPS: {fps:.1f} | Kare: {total_frames} | {width}x{height}")

    conn = db_connect()
    if is_live:
        # Canlı modda "yeniden işleme" diye bir şey yok — her başlangıç, o kameranın
        # akışına yeni katılmaktır. video_file="CANLI" TÜM kameralar için ortak
        # olacağından burada ASLA silme yapılmaz (pota_canli_takip.py ile aynı kural).
        video_file = "CANLI"
    else:
        video_file = os.path.basename(args.video)
        if not args.resume:
            db_delete_by_video(conn, video_file)  # aynı video tekrar işlenirse çift kayıt olmasın
    model = YOLO(args.model)

    # Device'ı AÇIKÇA belirt: ultralytics device="" bırakılınca (varsayılan çağrı)
    # Apple Silicon'da MPS'e otomatik geçmiyor, sessizce CPU'da çalışıyor — YOLO11l
    # 960px'te CPU'da çok yavaş, canlı modda hiç yetişmez.
    # Sıra: CUDA (saha sunucusu) → MPS (geliştirme Mac'i) → CPU (son çare).
    import torch
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        print("⚠ GPU bulunamadı, CPU'ya düşüldü — YOLO11l CPU'da çok yavaştır, "
              "canlı (RTSP) kullanımda kareler birikir. GPU'lu makine kullanın.")
    if is_live:
        print(f"Device: {device} | Canlı işleme hızı: ~{LIVE_PROCESS_FPS:.0f} kare/sn")
    else:
        print(f"Device: {device} | Frame stride: {args.stride} (efektif {fps/args.stride:.1f} fps)"
              + (f" | Start offset: {args.start_offset:.0f}sn" if args.start_offset else ""))

    writer = None
    if args.save:
        out_path = os.path.splitext(args.video)[0] + "_takip.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"Çıktı videosu: {out_path}")

    active = {}          # track_id -> {"first","last","first_c","last_c"}
    segments = []        # bitmiş ham track parçaları (sadece debug JSON dökümü için tutulur)
    pending = []         # canlı birleştirilmiş mantıksal potalar: {first,last,first_c,last_c,
                         # entry_partial,exit_partial,parts,row_id}. row_id=None → henüz DB'de yok.
    arabasi_active = {}  # pota_arabasi'nın kendi ham track'leri (tid -> info), DB'ye yazılmaz
    arabasi_spans = []   # arabasi'nın konum bazlı birleşik "sürekli varlık" aralıkları — pota
                         # segmentleri arasındaki uzun boşluklarda "araba ayrılmadı" kanıtı
    frame_idx = int(round(args.start_offset * fps))  # klip orijinalin bir parçasıysa gerçek pozisyondan başla

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def avg_point(pts):
        n = len(pts)
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

    frame_diag = (width**2 + height**2) ** 0.5

    # Kamera-bazlı kalibrasyon (bolge_editor.py ile çizilmiş) varsa onu kullan,
    # yoksa TEST-14SAAT-v2-full için doğrulanmış hardcoded fraksiyonlara düş.
    cam_cfg = load_camera_config(camera, width, height)
    if cam_cfg is not None:
        print(f"Kamera kalibrasyonu: veri/*_bolgeler.json ← kapı_sol={'var' if cam_cfg['gate_left'] else 'yok'}, "
              f"kapı_sağ={'var' if cam_cfg['gate_right'] else 'yok'}, "
              f"geçersiz_bölge={len(cam_cfg['invalid_zones'])} adet")
        gate_left = cam_cfg["gate_left"]
        gate_right = cam_cfg["gate_right"]
        invalid_zones = cam_cfg["invalid_zones"]
    else:
        print("Kamera kalibrasyonu: bulunamadı, hardcoded varsayılanlar kullanılıyor "
              "(TEST-14SAAT-v2-full için kalibre edilmiş)")
        gate_left = (GATE_LEFT_FRAC[0] * width, GATE_LEFT_FRAC[1] * width,
                     GATE_LEFT_FRAC[2] * height, GATE_LEFT_FRAC[3] * height)
        gate_right = (GATE_RIGHT_FRAC[0] * width, GATE_RIGHT_FRAC[1] * width,
                      GATE_RIGHT_FRAC[2] * height, GATE_RIGHT_FRAC[3] * height)
        invalid_zones = [(0, 0, width, MIN_VALID_Y_FRAC * height)]

    def gate_of(point):
        """Nokta sol/sağ kapı bölgesindeyse 'sol'/'sag' döner, ikisinde de değilse None
        (kadrajın ortasında kaybolmuş demektir — gerçek çıkış değil)."""
        x, y = point
        if gate_left is not None and _point_in_rect(x, y, gate_left):
            return "sol"
        if gate_right is not None and _point_in_rect(x, y, gate_right):
            return "sag"
        return None

    gecisli = GECISLI_ISTISNA.get(camera, GECISLI_DUZEN)
    min_dx = GATE_DIR_MIN_DX_FRAC * width

    def real_exit_gate(last_c, dx):
        """Parça GERÇEKTEN kapıdan çıktıysa kapı adını döner.
        Kapı dikdörtgeninde bitmek yetmez; dışa doğru (sol kapı → sola, sağ kapı → sağa)
        en az min_dx yol almış olmalı. Aksi halde kapı yakınında örtülme sayılır."""
        g = gate_of(last_c)
        if g == "sol" and dx <= -min_dx:
            return "sol"
        if g == "sag" and dx >= min_dx:
            return "sag"
        return None

    def real_entry_gate(first_c, dx):
        """Parça GERÇEKTEN kapıdan içeri girdiyse kapı adını döner (içe doğru hareket)."""
        g = gate_of(first_c)
        if g == "sol" and dx >= min_dx:
            return "sol"
        if g == "sag" and dx <= -min_dx:
            return "sag"
        return None

    def close_arabasi(info):
        """
        pota_arabasi için ayrı, hafif bir 'sürekli varlık' takibi. DB'ye yazmaz;
        sadece arabasi_spans'ta konum+zaman bazlı birleşik aralıklar tutar. Bu
        aralıklar, pota segmentleri arasındaki UZUN boşluklarda "araba yerinden
        hiç ayrılmadı" kanıtı olarak close_track içinde kullanılır.
        """
        seg = {"first": info["first"], "last": info["last"],
               "first_c": info["first_c"], "last_c": info["last_c"]}
        for s in arabasi_spans:
            if seg["first"] < s["last"] - 0.5:
                continue
            gap = seg["first"] - s["last"]
            if gap > ARABASI_GAP_TOLERANCE:
                continue
            if dist(s["last_c"], seg["first_c"]) / frame_diag > MERGE_DIST_FRAC:
                continue
            s["last"] = seg["last"]
            s["last_c"] = seg["last_c"]
            return
        arabasi_spans.append(seg)

    def close_track(tid, info, exit_partial=False):
        """
        Biten bir ham track'i konum bazlı kuralla mevcut bir mantıksal potaya
        ekler ya da yeni bir mantıksal pota başlatır — VE sonucu HEMEN PostgreSQL'e
        yazar (ilk parça INSERT, sonraki birleşen parçalar UPDATE). Böylece dashboard
        video işlenirken canlı güncellenir, saatler süren video bitene kadar beklenmez.

        Birleştirme iki aşamalı:
          1. SIKI: zaman çakışmıyor, boşluk MERGE_TIME_GAP'ten az, son konumlar
             MERGE_DIST_FRAC'tan yakın. Görünüme bakmaz, sadece konum+zaman.
             Kapı doğrulaması: önceki track kadraj ortasında (kapı dışında) kaybolduysa
             çakışma süresi önemsenmez (duman/kıvılcım varsayılır). Gerçekten bir kapıdan
             çıktıysa yeni track yalnızca AYNI kapıdan giriyorsa birleşir.
          2. YARDIMCI (sıkı eşleşme yoksa): boşluk MERGE_TIME_GAP'i aşsa bile,
             pota_arabasi o boşluk boyunca AYNI yerde hiç kaybolmadan durduysa
             (arabasi_spans'ta kanıtlanır), süre sınırı olmadan birleştir — araba
             yerinden ayrılmadıysa başka bir potanın gelmiş olması imkansız.
        """
        seg = {
            "first": info["first"], "last": info["last"],
            "first_c": avg_point(info["first_pts"]), "last_c": avg_point(info["last_pts"]),
            "entry_partial": info["first"] <= START_TOLERANCE_SEC,
            "exit_partial": exit_partial,
        }
        segments.append({**seg, "tid": tid})   # sadece ham_track.json dökümü için

        target, best_gap = None, None
        seg_dx = seg["last_c"][0] - seg["first_c"][0]
        entry_gate = real_entry_gate(seg["first_c"], seg_dx)
        for p in pending:
            overlap = p["last"] - seg["first"]           # >0 ise çakışma var
            gap = seg["first"] - p["last"]                # çakışma varsa negatif olur, sorun değil
            exit_gate = real_exit_gate(p["last_c"], p.get("exit_dx", 0.0))
            if exit_gate is not None:
                # p GERÇEKTEN kapıdan çıktı (kapı dikdörtgeninde + dışa doğru hareket).
                if gecisli:
                    # Geçişli düzen: pota girdiği kapıdan çıkmak zorunda değil, dolayısıyla
                    # kadrajı terk eden pota geri dönmez — hiçbir yeni parçayla birleşmez.
                    continue
                # Eski "aynı kapı" düzeni: sadece aynı kapıdan geri girerse aynı potadır.
                if entry_gate != exit_gate:
                    continue
                if overlap > OVERLAP_TOLERANCE_SEC:       # UZUN çakışma → gerçekten farklı pota
                    continue
                if gap > MERGE_TIME_GAP:
                    continue
            elif entry_gate is not None and gate_of(p["last_c"]) != entry_gate:
                # Kapıdan GERÇEK bir giriş var ama p o kapıda değil (kadrajın içinde
                # bekliyor) → giren BAŞKA bir potadır, konumca yakın olsa bile birleşmez.
                # p de AYNI kapıdaysa bu, tek bir girişin parçalanmasıdır (pota kapıdan
                # içeri süzülürken tracker'ın onu bir anlığına kaybetmesi) → birleşir.
                # Bu ayrım olmadan giriş anı 15-30sn geç kaydediliyordu.
                continue
            else:
                # ikisi de kapı dışındaysa (kadraj ortasında duman/kıvılcım/buhar yüzünden
                # kayboldu, GERÇEK ÇIKIŞ DEĞİL): OVERLAP_TOLERANCE_SEC/MERGE_TIME_GAP yerine
                # daha geniş bir üst sınır uygulanır. Sınırsız bırakılamaz: kadrajın merkezî
                # bekleme noktasına potalar muhtemelen vinçle iniyor/kalkıyor (kapılara hiç
                # değmeden) — bu yüzden konum yakınlığı TEK BAŞINA "aynı pota" kanıtı olamaz,
                # farklı saatlerde gelen tamamen farklı potalar da aynı yerde durur.
                if gap > GATE_NONE_MAX_GAP:
                    continue
            if dist(p["last_c"], seg["first_c"]) / frame_diag > MERGE_DIST_FRAC:
                continue
            if target is None or gap < best_gap:
                target, best_gap = p, gap

        assisted_merge = False
        if target is None:
            for p in pending:
                # Kapı kuralı burada da aynen geçerli (bkz. yukarıdaki sıkı eşleşme):
                # gerçekten çıkmış pota geri dönmez, gerçek yeni giriş başka potadır.
                exit_gate = real_exit_gate(p["last_c"], p.get("exit_dx", 0.0))
                if exit_gate is not None:
                    if gecisli:
                        continue
                    if entry_gate != exit_gate:
                        continue
                    if p["last"] - seg["first"] > OVERLAP_TOLERANCE_SEC:
                        continue
                elif entry_gate is not None and gate_of(p["last_c"]) != entry_gate:
                    continue
                if dist(p["last_c"], seg["first_c"]) / frame_diag > MERGE_DIST_FRAC:
                    continue
                for s in arabasi_spans:
                    if (s["first"] <= p["last"] + ARABASI_COVER_SLACK
                            and s["last"] >= seg["first"] - ARABASI_COVER_SLACK
                            and dist(s["last_c"], p["last_c"]) / frame_diag <= ASSIST_DIST_FRAC):
                        target, assisted_merge = p, True
                        break
                if target is not None:
                    break

        if target is not None:
            if seg["last"] > target["last"]:              # kısa çakışmalı birleşmede süreyi GERİYE almasın
                target["last"] = seg["last"]
                target["last_c"] = seg["last_c"]
                target["exit_dx"] = seg_dx                # kapı yön testi son parçanın hareketine bakar
            target["exit_partial"] = seg["exit_partial"]
            target["parts"].append(tid)
            duration = target["last"] - target["first"]
            entry_t = video_start + timedelta(seconds=target["first"])
            exit_t = video_start + timedelta(seconds=target["last"])
            etiket = "  [ARABA SÜREKLİLİĞİYLE BİRLEŞTİ, uzun boşluk]" if assisted_merge else ""
            if target["row_id"] is None:
                if duration >= MIN_TRACK_SEC:
                    target["row_id"] = db_save_event(
                        conn, camera, target["parts"][0], entry_t, exit_t, duration, video_file,
                        entry_partial=target["entry_partial"], exit_partial=target["exit_partial"])
                    print(f"  ✔ [DB] Pota #{target['parts'][0]}: {entry_t.strftime('%H:%M:%S')} → "
                          f"{exit_t.strftime('%H:%M:%S')} ({duration:.0f} sn)  "
                          f"[{len(target['parts'])} parça birleşti]{etiket}")
            else:
                db_update_event(conn, target["row_id"], exit_t, duration, target["exit_partial"])
                print(f"  ↻ [DB] Pota #{target['parts'][0]} güncellendi: çıkış "
                      f"{exit_t.strftime('%H:%M:%S')} ({duration:.0f} sn)  "
                      f"[{len(target['parts'])} parça birleşti]{etiket}")
        else:
            new = {**seg, "parts": [tid], "row_id": None, "exit_dx": seg_dx}
            duration = new["last"] - new["first"]
            if duration >= MIN_TRACK_SEC:
                entry_t = video_start + timedelta(seconds=new["first"])
                exit_t = video_start + timedelta(seconds=new["last"])
                new["row_id"] = db_save_event(
                    conn, camera, tid, entry_t, exit_t, duration, video_file,
                    entry_partial=new["entry_partial"], exit_partial=new["exit_partial"])
                print(f"  ✔ [DB] Pota #{tid}: {entry_t.strftime('%H:%M:%S')} → "
                      f"{exit_t.strftime('%H:%M:%S')} ({duration:.0f} sn)")
            pending.append(new)

    # Tracker ayarı: proje içindeki özel dosyaysa tam yolunu ver, değilse ismi
    # (bytetrack.yaml gibi ultralytics'in hazır configleri) doğrudan geçir.
    tracker_cfg = args.tracker
    local_cfg = os.path.join(PROJECT_DIR, args.tracker)
    if os.path.exists(local_cfg):
        tracker_cfg = local_cfg
    print(f"Tracker: {os.path.basename(tracker_cfg)}")

    def process_frame_result(result, video_sec):
        """Tek bir karenin (dosyadan ya da canlı akıştan fark etmez) tespit sonucunu
        işler: ham track'leri günceller/kapatır, overlay çizer. 'q' ile çıkılırsa
        False döner (çağıran taraf döngüyü durdurur)."""
        seen_ids = set()
        seen_arabasi_ids = set()
        arabasi_centers = []   # bu karedeki pota_arabasi merkezleri (yardımcı sinyal)

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clss = result.boxes.cls.int().tolist()
            centers = result.boxes.xywh.cpu().numpy()   # [x_center, y_center, w, h]
            for tid, cls_id, c in zip(ids, clss, centers):
                cx, cy = float(c[0]), float(c[1])
                if cls_id == CLASS_ARABASI:
                    arabasi_centers.append((cx, cy))
                    seen_arabasi_ids.add(tid)
                    if tid not in arabasi_active:
                        arabasi_active[tid] = {"first": video_sec, "last": video_sec,
                                                "first_c": (cx, cy), "last_c": (cx, cy)}
                    else:
                        arabasi_active[tid]["last"] = video_sec
                        arabasi_active[tid]["last_c"] = (cx, cy)
                    continue
                if cls_id != CLASS_POTA:
                    continue
                if any(_point_in_rect(cx, cy, z) for z in invalid_zones):
                    continue   # bilinen gürültü bölgesi (köşe yansıması/kıvılcım gibi) —
                               # kamera kalibrasyonu varsa "gecersiz" etiketli çizim,
                               # yoksa varsayılan üst-yarı sınırı (bkz. MIN_VALID_Y_FRAC)
                seen_ids.add(tid)
                if tid not in active:
                    active[tid] = {"first": video_sec, "last": video_sec,
                                   "first_c": (cx, cy), "last_c": (cx, cy),
                                   "first_pts": [(cx, cy)],
                                   "last_pts": deque([(cx, cy)], maxlen=STABLE_POINT_N)}
                    print(f"  → ham track #{tid} ({video_sec:.1f}. sn)")
                else:
                    active[tid]["last"] = video_sec
                    active[tid]["last_c"] = (cx, cy)
                    active[tid]["last_pts"].append((cx, cy))
                    if len(active[tid]["first_pts"]) < STABLE_POINT_N:
                        active[tid]["first_pts"].append((cx, cy))

        for tid in list(active.keys()):
            if tid in seen_ids:
                continue
            gap = video_sec - active[tid]["last"]
            assisted = any(
                dist(active[tid]["last_c"], ac) / frame_diag < ASSIST_DIST_FRAC
                for ac in arabasi_centers
            )
            grace = EXIT_GRACE_SEC_ASSISTED if assisted else EXIT_GRACE_SEC
            if gap > grace:
                close_track(tid, active.pop(tid))

        for tid in list(arabasi_active.keys()):
            if tid in seen_arabasi_ids:
                continue
            if video_sec - arabasi_active[tid]["last"] > ARABASI_GAP_TOLERANCE:
                close_arabasi(arabasi_active.pop(tid))

        if args.show or writer is not None:
            frame = draw_overlay(cv2, result, active, video_sec, video_start, camera, len(segments))
            if writer is not None:
                writer.write(frame)
            if args.show:
                disp = frame
                if width > 1600:
                    disp = cv2.resize(frame, (1280, int(height * 1280 / width)))
                cv2.imshow("Pota Takip", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False
        return True

    if is_live:
        # Canlı mod: FrameGrabber'dan gelen kareyi tek tek model.track(persist=True)'a
        # besliyoruz (ultralytics'in kendi akış yönetimini değil, canli_kaynak'ın her
        # zaman EN GÜNCEL kareyi tutan/otomatik yeniden bağlanan mantığını kullanmak
        # için). video_sec = gerçek geçen süre (wall-clock) — close_track'in
        # video_start + timedelta(seconds=...) formülü böylece hem dosya hem canlı
        # modda DEĞİŞMEDEN çalışır.
        print(f"▶ Canlı takip başladı — kamera: {camera}  (Ctrl+C ile durdurun)")
        interval = 1.0 / LIVE_PROCESS_FPS
        last_proc = 0.0
        try:
            while not _STOP and not grabber.bitti:
                now = time.time()
                if now - last_proc < interval:
                    time.sleep(0.01)
                    continue
                last_proc = now
                frame = grabber.latest()
                if frame is None:
                    time.sleep(0.1)
                    continue
                video_sec = (datetime.now() - video_start).total_seconds()
                results = model.track(
                    source=frame, conf=CONF_THRESHOLD, tracker=tracker_cfg,
                    persist=True, verbose=False, device=device,
                )
                if not process_frame_result(results[0], video_sec):
                    break
            if grabber.bitti:
                print("Test dosyası bitti (gerçek canlı akış hiçbir zaman bitmez).")
        finally:
            grabber.stop()
    else:
        results = model.track(
            source=args.video, conf=CONF_THRESHOLD, tracker=tracker_cfg,
            stream=True, persist=True, verbose=False,
            device=device, vid_stride=args.stride,
        )
        for result in results:
            video_sec = frame_idx / fps
            frame_idx += args.stride   # vid_stride ile atlanan kareler yüzünden 1 değil stride kadar ilerle
            if not process_frame_result(result, video_sec):
                break

    # Video/akış bitti (ya da durduruldu): hâlâ kadrajda olan track'lerin gerçek çıkışı bilinmiyor
    for tid, info in active.items():
        close_track(tid, info, exit_partial=True)
    for info in arabasi_active.values():
        close_arabasi(info)

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    saved = sum(1 for p in pending if p["row_id"] is not None)
    noise = len(pending) - saved

    if not is_live:
        # --- Ham track verisini diske dök (birleştirme ayarını ısı yaratmadan denemek için) ---
        # Canlı modda yapılmaz: akış URL'sinden anlamlı bir dosya yolu türetilemez ve
        # 7/24 çalışan bir süreçte "segments" listesi zaten sınırsız büyür — bu sadece
        # dosya-modu debug/analiz amaçlı bir döküm.
        raw_path = os.path.splitext(args.video)[0] + "_ham_track.json"
        with open(raw_path, "w") as f:
            json.dump({"camera": camera, "video_start": video_start.isoformat(),
                       "frame_diag": frame_diag,
                       "video_file": video_file,
                       "segments": segments,
                       "arabasi_spans": arabasi_spans}, f, indent=2)
        print(f"Ham track verisi: {os.path.basename(raw_path)}")
        if args.save:
            print(f"İşaretlenmiş video: {os.path.splitext(args.video)[0]}_takip.mp4")

    conn.close()
    print(f"\n{len(segments)} ham track → {len(pending)} mantıksal pota "
          f"({saved} PostgreSQL'e yazıldı, {noise} çok kısa/gürültü sayılıp atlandı)")
    print("Kayıtları görmek için: python pota_takip_sistemi_v2.py report")


# ============================================================
# RAPOR (report komutu)
# ============================================================

def cmd_report(args):
    conn = db_connect()
    rows = db_fetch_events(conn, camera=args.camera)
    conn.close()

    if not rows:
        print("\nKayıt bulunamadı.")
        return

    print(f"\n{'Kamera':<24} {'ID':>3} {'Giriş':<10} {'Çıkış':<10} {'Süre':>8}  Not")
    print("-" * 82)

    partial = 0
    for camera, tid, entry, exit_, duration, e_part, x_part, _video in rows:
        entry_s = entry.strftime("%H:%M:%S")
        exit_s = exit_.strftime("%H:%M:%S")

        note = ""
        if e_part and x_part:
            note = "⚠ giriş+çıkış kayıt dışında"
        elif e_part:
            note = "⚠ kayıt başlarken zaten kadrajdaydı (süre daha uzun)"
        elif x_part:
            note = "⚠ kayıt biterken hâlâ kadrajdaydı (süre daha uzun)"
        if note:
            partial += 1

        print(f"{camera:<24} {tid:>3} {entry_s:<10} {exit_s:<10} {duration:>6.1f}s  {note}")

    print(f"\nToplam {len(rows)} olay.")
    if partial:
        print(f"{partial} tanesinin süresi eksik — pota, kayıt penceresinin dışında girmiş/çıkmış.")


# ============================================================
# GİRİŞ NOKTASI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pota Takip Sistemi — YOLO + ByteTrack + PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("track", help="Videoyu işle ve olayları PostgreSQL'e yaz")
    t.add_argument("--video", required=True,
                    help="Video dosyası yolu, YA DA canlı kaynak (rtsp://, http(s)://, tcp://, udp://)")
    t.add_argument("--camera", default=None,
                    help="Kamera adı (dosya modunda verilmezse dosya adından çıkarılır; "
                         "canlı modda ZORUNLU)")
    t.add_argument("--model", default=DEFAULT_MODEL, help="Eğitilmiş model (best.pt) yolu")
    t.add_argument("--show", action="store_true", help="Takibi canlı pencerede göster")
    t.add_argument("--save", action="store_true", help="İşaretlenmiş çıktı videosu kaydet")
    t.add_argument("--tracker", default="bytetrack.yaml",
                   help="Tracker ayarı: bytetrack.yaml (varsayılan) veya botsort_reid.yaml "
                        "(görünüm hafızalı — kadrajdan çıkıp dönen potayı aynı ID ile tanır)")
    t.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help=f"Her N. kareyi işle, hız/doğruluk dengesi (varsayılan {DEFAULT_STRIDE}). "
                        "Doğrulama için --stride 1 ile tam kare işleyip karşılaştırın.")
    t.add_argument("--start-offset", type=float, default=0.0,
                   help="Bu video, orijinal kaynağın şu kadar saniye ilerisinden kesilmiş "
                        "(örn. bir aralık test klibi) — giriş/çıkış saatleri buna göre kayar.")
    t.add_argument("--resume", action="store_true",
                   help="Aynı videonun önceki kayıtlarını SİLME (çökme sonrası kaldığı yerden "
                        "--start-offset ile devam ederken önceki saatleri korumak için).")
    t.add_argument("--live", action="store_true",
                   help="Kaynak rtsp://,http(s):// vb. ile başlamasa bile canlı-mod kodunu zorla "
                        "(örn. bir video dosyasını canlı akışmış gibi test etmek, ya da standart "
                        "olmayan bir aygıt/protokol için).")

    r = sub.add_parser("report", help="Kaydedilen olayları listele")
    r.add_argument("--camera", default=None, help="Sadece bu kameranın kayıtları")

    args = parser.parse_args()

    if args.command == "track":
        cmd_track(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
