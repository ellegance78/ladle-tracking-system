"""
============================================================
POTA TAKİP — Bölge Tarama + Otomatik İşlem Bölgesi
============================================================
Amaç: "İşleme giren pota"yı, kenarda boşta duran potadan ayırmak.
Çözüm: her kameranın İŞLEM BÖLGESİ'ni (potanın döküm/kamera pozisyonu)
otomatik bul. Sadece o bölgedeki potalar sayılır.

Bu script videoyu SEYREK tarar (saniyede ~2 kare) — 25 dk videoda 30 bin
yerine ~3 bin kare işler, çok daha az ısı. Tüm tespitleri diske kaydeder;
gerisi (bölge bulma, giriş/çıkış hesabı) ısı gerektirmeyen offline analiz.

KULLANIM
    # 1) Videoyu tara, tespitleri kaydet + ısı ısı haritası + önerilen bölge
    python bolge_tarama.py scan --video "kayit.mp4"

    Çıktılar (video adının yanına):
      <video>_tespitler.json   — ham tespitler (offline analiz için)
      <video>_isihalitasi.jpg  — potaların nerede göründüğü (ısı haritası)
      <video>_onerilen_bolge.jpg — otomatik önerilen işlem bölgesi (bir kare üstünde)
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np


CONF_THRESHOLD = 0.4
SAMPLE_FPS = 2.0     # saniyede kaç kare işlensin (düşük = az ısı; pota dakikalarca durduğu için yeterli)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "runs", "pota_yolo11l", "weights", "best.pt")

# Tüm analiz verileri burada tutulur (tespitler, bölgeler, kareler).
# Böylece videonun diskteki yeri değişse de veri kaybolmaz / bağ kopmaz.
VERI_DIR = os.path.join(PROJECT_DIR, "veri")


def data_base(video):
    """Bir video için veri/ altındaki temel yolu döndürür (uzantısız, klasörsüz ad)."""
    os.makedirs(VERI_DIR, exist_ok=True)
    return os.path.join(VERI_DIR, os.path.splitext(os.path.basename(video))[0])


def cmd_scan(args):
    import cv2
    from ultralytics import YOLO

    if not os.path.exists(args.video):
        raise SystemExit(f"Video bulunamadı: {args.video}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    stride = max(1, int(round(fps / SAMPLE_FPS)))
    print(f"Video: {width}x{height} | {fps:.0f}fps | {total} kare")
    print(f"Seyrek tarama: her {stride}. kare (~{fps/stride:.1f} kare/sn) → ~{total//stride} kare işlenecek\n")

    model = YOLO(args.model)

    # Isı haritası: düşük çözünürlüklü ızgara (potaların nerede göründüğü)
    GRID = 64
    heat = np.zeros((GRID, GRID), dtype=np.float64)

    detections = []   # her tespit: {"sec", "cx", "cy", "w", "h"}
    frame_idx = 0
    sample_frame = None   # bölge önizlemesi için ortadan bir kare

    results = model.predict(
        source=args.video, conf=CONF_THRESHOLD, stream=True, verbose=False,
        vid_stride=stride,
    )

    for r in results:
        sec = (frame_idx * stride) / fps
        frame_idx += 1

        if frame_idx == max(1, (total // stride) // 2):
            sample_frame = r.orig_img.copy()   # videonun ortasından temsili kare

        if r.boxes is not None and len(r.boxes) > 0:
            for c in r.boxes.xywh.cpu().numpy():
                cx, cy, w, h = float(c[0]), float(c[1]), float(c[2]), float(c[3])
                detections.append({"sec": round(sec, 1), "cx": round(cx), "cy": round(cy),
                                   "w": round(w), "h": round(h)})
                # ısı haritasına ekle (kutunun kapladığı alan)
                gx1 = int((cx - w/2) / width * GRID); gx2 = int((cx + w/2) / width * GRID)
                gy1 = int((cy - h/2) / height * GRID); gy2 = int((cy + h/2) / height * GRID)
                heat[max(0,gy1):min(GRID,gy2+1), max(0,gx1):min(GRID,gx2+1)] += 1

        if frame_idx % 200 == 0:
            print(f"  {frame_idx} kare işlendi, {len(detections)} tespit…")

    if sample_frame is None:
        cap = cv2.VideoCapture(args.video)
        ok, sample_frame = cap.read()
        cap.release()

    print(f"\nTarama bitti: {frame_idx} kare, {len(detections)} tespit.")

    base = data_base(args.video)

    # --- tespitleri kaydet ---
    det_path = base + "_tespitler.json"
    with open(det_path, "w") as f:
        json.dump({"width": width, "height": height, "fps": fps,
                   "sample_fps": fps/stride, "video_file": os.path.basename(args.video),
                   "detections": detections}, f)
    print(f"✔ Tespitler: {os.path.basename(det_path)}")

    # --- ısı haritası görseli ---
    heat_norm = (heat / heat.max() * 255).astype(np.uint8) if heat.max() > 0 else heat.astype(np.uint8)
    heat_big = cv2.resize(heat_norm, (width, height), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_big, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(sample_frame, 0.55, heat_color, 0.45, 0)
    cv2.imwrite(base + "_isihalitasi.jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"✔ Isı haritası: {os.path.basename(base)}_isihalitasi.jpg")

    # --- otomatik bölge öner ---
    zone = propose_zone(heat, width, height)
    prev = sample_frame.copy()
    x1,y1,x2,y2 = zone
    cv2.rectangle(prev, (x1,y1), (x2,y2), (0,255,0), 4)
    cv2.putText(prev, "ONERILEN ISLEM BOLGESI", (x1, max(30,y1-12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv2.imwrite(base + "_onerilen_bolge.jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"✔ Önerilen bölge: {os.path.basename(base)}_onerilen_bolge.jpg")
    print(f"  bölge (x1,y1,x2,y2) = {zone}")

    # bölgeyi de kaydet (sonraki adım kullanacak)
    with open(base + "_bolge.json", "w") as f:
        json.dump({"zone": zone, "width": width, "height": height}, f)


def propose_zone(heat, width, height, coverage=0.6):
    """
    Isı haritasından işlem bölgesini öner.
    En yoğun hücreden başlayıp, toplam yoğunluğun `coverage` oranını kapsayan
    en küçük dikdörtgeni bul. Kenardaki dağınık (boşta pota) tespitler değil,
    potaların en çok toplandığı merkezi bölge seçilir.
    """
    GRID = heat.shape[0]
    total = heat.sum()
    if total == 0:
        return [width//4, height//4, 3*width//4, 3*height//4]

    # yoğunluk ağırlık merkezi
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    cx = (xs * heat).sum() / total
    cy = (ys * heat).sum() / total

    # merkezden dışa doğru büyüyen kutu, coverage oranına ulaşınca dur
    order = np.argsort(-heat.ravel())
    cumulative = 0.0
    sel = np.zeros_like(heat, dtype=bool)
    for idx in order:
        gy, gx = divmod(idx, GRID)
        sel[gy, gx] = True
        cumulative += heat[gy, gx]
        if cumulative >= total * coverage:
            break

    ys_sel, xs_sel = np.where(sel)
    gx1, gx2 = xs_sel.min(), xs_sel.max()
    gy1, gy2 = ys_sel.min(), ys_sel.max()

    # ızgaradan piksele
    x1 = int(gx1 / GRID * width);  x2 = int((gx2+1) / GRID * width)
    y1 = int(gy1 / GRID * height); y2 = int((gy2+1) / GRID * height)
    return [x1, y1, x2, y2]


# ============================================================
# ANALİZ — kaydedilmiş tespitlerden giriş/çıkış hesabı (OFFLINE, ısı yok)
# ============================================================

# İşlem bölgesi bu kadar saniyeden uzun boş kalırsa "çıkış" sayılır.
# Kısa tespit kayıplarını (buhar, anlık kaybolma) köprüler; kamera gerçekten
# boşalıp yeni pota gelince iki ayrı olay üretir.
EXIT_GRACE_SEC = 15.0
MIN_EVENT_SEC = 20.0   # bundan kısa süren "işlem" gürültü sayılır (kenardan geçen vb.)


def compute_events(detections, zone, sample_fps):
    """
    Bir bölgenin doluluk zaman çizelgesinden giriş/çıkış olaylarını çıkarır.
    Dönüş: [{"first","last","entry_partial","exit_partial","dur"}]  (saniye cinsinden)
    """
    zx1, zy1, zx2, zy2 = zone
    all_secs = sorted(set(d["sec"] for d in detections))
    occupied = set()
    for d in detections:
        if zx1 <= d["cx"] <= zx2 and zy1 <= d["cy"] <= zy2:
            occupied.add(d["sec"])

    step = 1.0 / sample_fps
    occ_times = sorted(occupied)

    intervals = []
    if occ_times:
        start = prev = occ_times[0]
        for t in occ_times[1:]:
            if t - prev > EXIT_GRACE_SEC:
                intervals.append((start, prev))
                start = t
            prev = t
        intervals.append((start, prev))

    video_end = max(all_secs) if all_secs else 0

    events = []
    for (a, b) in intervals:
        dur = (b - a) + step
        if dur < MIN_EVENT_SEC:
            continue
        events.append({
            "first": a, "last": b, "dur": dur,
            "entry_partial": a <= step + 0.01,
            "exit_partial": b >= video_end - step - 0.01,
        })
    return events


def _elle_cizilen_bolge_ayni_kamera(camera):
    """
    Aynı KAMERADAN elle çizilmiş bir bölge dosyası ara.
    Bölgeler kayda değil kameraya aittir: bir kamerada bir kez çizilen bölge,
    o kameranın TÜM kayıtlarında geçerli olur (farklı saatlerdeki videolar dahil).
    """
    import glob
    for f in sorted(glob.glob(os.path.join(VERI_DIR, "*_bolgeler.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("kaynak") != "elle_cizildi":
            continue
        # dosya adından kamerayı çöz
        ad = os.path.basename(f)[:-len("_bolgeler.json")]
        kam, _ = parse_filename_local(ad)
        if kam == camera:
            return d["zones"], os.path.basename(f)
    return None, None


def _load_zones(base, camera=None, single_zone_arg=None):
    """Bölgeleri yükle: elle verilen > bu videonun dosyası > AYNI KAMERANIN çizimi > otomatik."""
    if single_zone_arg:
        return [[int(v) for v in single_zone_arg.split(",")]]
    # 1. bu videonun kendi bölge dosyası
    if os.path.exists(base + "_bolgeler.json"):
        return json.load(open(base + "_bolgeler.json"))["zones"]
    # 2. aynı kameradan elle çizilmiş bölge (başka bir kayıtta çizilmiş olabilir)
    if camera:
        zones, kaynak_dosya = _elle_cizilen_bolge_ayni_kamera(camera)
        if zones is not None:
            print(f"Bu kayıt için bölge yok; AYNI KAMERANIN çizimi kullanılıyor: {kaynak_dosya}")
            return zones
    # 3. otomatik saptanmış tek bölge
    if os.path.exists(base + "_bolge.json"):
        return [json.load(open(base + "_bolge.json"))["zone"]]
    raise SystemExit("Bölge bulunamadı. Editörle çizin, ya da 'scan' çalıştırıp otomatik saptayın.")


def cmd_analiz(args):
    base = data_base(args.video)
    det_path = base + "_tespitler.json"
    if not os.path.exists(det_path):
        raise SystemExit(f"Tespit dosyası yok: {det_path}\nÖnce 'scan' çalıştırın.")

    det = json.load(open(det_path))
    camera, video_start = parse_filename_local(det["video_file"])
    if video_start is None:
        video_start = datetime(2000, 1, 1)
    zones = _load_zones(base, camera=camera, single_zone_arg=args.zone)

    conn = None
    if args.kaydet:
        from pota_takip_sistemi import db_connect, db_save_event, db_delete_by_video
        conn = db_connect()
        db_delete_by_video(conn, det["video_file"])   # aynı video tekrar işlenince çift kayıt olmasın

    print(f"\nKamera: {camera} | {len(zones)} işlem bölgesi")
    print(f"{'Bölge':<6}{'Pota':<5}{'Giriş':<10}{'Çıkış':<10}{'Süre':>8}  Not")
    print("-" * 62)

    toplam = 0
    for zi, zone in enumerate(zones, 1):
        events = compute_events(det["detections"], zone, det.get("sample_fps", 2.0))
        for n, e in enumerate(events, 1):
            et = video_start + timedelta(seconds=e["first"])
            xt = video_start + timedelta(seconds=e["last"] + 1.0/det.get("sample_fps", 2.0))
            note = ""
            if e["entry_partial"] and e["exit_partial"]:
                note = "⚠ giriş+çıkış kayıt dışında"
            elif e["entry_partial"]:
                note = "⚠ başta zaten işlemde"
            elif e["exit_partial"]:
                note = "⚠ sonda hâlâ işlemde"
            print(f"{zi:<6}{n:<5}{et.strftime('%H:%M:%S'):<10}{xt.strftime('%H:%M:%S'):<10}"
                  f"{e['dur']:>6.0f}s  {note}")
            if conn:
                db_save_event(conn, camera, n, et, xt, e["dur"], det["video_file"],
                              entry_partial=e["entry_partial"], exit_partial=e["exit_partial"],
                              zone_no=zi)
            toplam += 1

    if conn:
        conn.close()
        print(f"\n✔ {toplam} işlem olayı PostgreSQL'e kaydedildi.")
    else:
        print(f"\n{toplam} işlem olayı bulundu (kaydetmek için --kaydet ekleyin).")
    print(f"(eşikler: çıkış boşluğu >{EXIT_GRACE_SEC:.0f}s, en kısa olay {MIN_EVENT_SEC:.0f}s)")


def parse_filename_local(name):
    import re
    m = re.search(r"NVR_(.+?)_(\d{14})", name)
    if not m:
        return name, None
    cam = m.group(1).replace("", "").strip()
    return cam, datetime.strptime(m.group(2), "%Y%m%d%H%M%S")


# ============================================================
# ÇOKLU BÖLGE — çift taraflı düzenek (kamera: döküm + bekleme) saptama
# ============================================================

def cmd_bolgeler(args):
    """
    Kaydedilmiş tespitlerden ISI HARİTASINI yeniden kurup BİRDEN FAZLA
    işlem noktası saptar (ısı yaratmaz — video yeniden işlenmez).
    Çift taraflı düzeneklerde (kamera) hem döküm hem bekleme pozisyonunu bulur.
    """
    import cv2

    base = data_base(args.video)

    # Elle çizilmiş bölge varsa OTOMATİK saptamayı atla — kullanıcının çizimi geçerli.
    zones_path = base + "_bolgeler.json"
    if os.path.exists(zones_path):
        existing = json.load(open(zones_path))
        if existing.get("kaynak") == "elle_cizildi":
            print(f"Elle çizilmiş {len(existing.get('zones',[]))} bölge bulundu "
                  f"(bolge_editor ile). Otomatik saptama atlanıyor.")
            return

    det_path = base + "_tespitler.json"
    if not os.path.exists(det_path):
        raise SystemExit(f"Tespit dosyası yok: {det_path}\nÖnce 'scan' çalıştırın.")

    det = json.load(open(det_path))
    width, height = det["width"], det["height"]
    detections = det["detections"]

    # AYNI KAMERANIN elle çizilmiş bölgesi varsa (başka kayıtta), otomatik saptama yapma.
    camera, _ = parse_filename_local(det["video_file"])
    ec_zones, ec_dosya = _elle_cizilen_bolge_ayni_kamera(camera)
    if ec_zones is not None:
        print(f"'{camera}' kamerası için elle çizilmiş bölge var ({ec_dosya}). "
              f"Otomatik saptama atlanıyor, o bölgeler kullanılacak.")
        return

    # ısı haritasını tespitlerden yeniden kur
    GRID = 96
    heat = np.zeros((GRID, GRID), dtype=np.float64)
    for d in detections:
        cx, cy, w, h = d["cx"], d["cy"], d["w"], d["h"]
        gx1 = int((cx - w/2)/width*GRID); gx2 = int((cx + w/2)/width*GRID)
        gy1 = int((cy - h/2)/height*GRID); gy2 = int((cy + h/2)/height*GRID)
        heat[max(0,gy1):min(GRID,gy2+1), max(0,gx1):min(GRID,gx2+1)] += 1

    zones = propose_multi_zones(heat, width, height,
                                thresh_frac=args.thresh, min_frac=args.minfrac)

    print(f"{len(zones)} işlem noktası saptandı:\n")
    for i, z in enumerate(zones, 1):
        pay = z["pay"]
        print(f"  Bölge {i}: (x1,y1,x2,y2)={z['box']}  | toplam tespitin %{pay*100:.0f}'i")

    # görselleştir
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)//2))
    ok, frame = cap.read()
    cap.release()
    if ok:
        colors = [(0,255,0),(0,180,255),(255,120,0),(200,0,255)]
        for i, z in enumerate(zones, 1):
            c = colors[(i-1) % len(colors)]
            x1,y1,x2,y2 = z["box"]
            cv2.rectangle(frame, (x1,y1),(x2,y2), c, 4)
            cv2.putText(frame, f"BOLGE {i} (%{z['pay']*100:.0f})", (x1, max(30,y1-12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3)
        out = base + "_coklu_bolge.jpg"
        cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\n✔ Görsel: {os.path.basename(out)}")

    # kaydet
    with open(base + "_bolgeler.json", "w") as f:
        json.dump({"zones": [z["box"] for z in zones], "width": width, "height": height}, f)


def propose_multi_zones(heat, width, height, thresh_frac=0.20, min_frac=0.05):
    """
    Isı haritasındaki AYRI sıcak bölgeleri bulur (bağlı bileşen analizi).
    thresh_frac: tepe değerinin bu oranı üstü "sıcak" sayılır
    min_frac:    bir bölge toplam ısının en az bu kadarını içermeli (gürültü elemesi)
    """
    import cv2
    GRID = heat.shape[0]
    total = heat.sum()
    if total == 0:
        return [{"box": [width//4, height//4, 3*width//4, 3*height//4], "pay": 1.0}]

    mask = (heat >= heat.max() * thresh_frac).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    zones = []
    for lbl in range(1, n):
        comp = (labels == lbl)
        pay = heat[comp].sum() / total
        if pay < min_frac:
            continue
        ys, xs = np.where(comp)
        x1 = int(xs.min()/GRID*width);  x2 = int((xs.max()+1)/GRID*width)
        y1 = int(ys.min()/GRID*height); y2 = int((ys.max()+1)/GRID*height)
        zones.append({"box": [x1,y1,x2,y2], "pay": pay})

    zones.sort(key=lambda z: -z["pay"])
    return zones if zones else [{"box":[width//4,height//4,3*width//4,3*height//4],"pay":1.0}]


def main():
    parser = argparse.ArgumentParser(description="Pota işlem bölgesi tarama ve analiz")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Videoyu tara, tespitleri + ısı haritası + önerilen bölge üret")
    s.add_argument("--video", required=True)
    s.add_argument("--model", default=DEFAULT_MODEL)

    a = sub.add_parser("analiz", help="Kaydedilmiş tespitlerden giriş/çıkış hesapla (ısı yok)")
    a.add_argument("--video", required=True, help="scan yapılan videonun yolu (tespit dosyasını bulmak için)")
    a.add_argument("--zone", default=None, help="Elle bölge: x1,y1,x2,y2 (verilmezse otomatik öneri)")
    a.add_argument("--kaydet", action="store_true", help="Sonuçları PostgreSQL'e yaz")

    b = sub.add_parser("bolgeler", help="Çoklu işlem noktası sapta (çift taraflı düzenek) — ısı yok")
    b.add_argument("--video", required=True)
    b.add_argument("--thresh", type=float, default=0.20, help="Tepe değerinin bu oranı üstü sıcak sayılır")
    b.add_argument("--minfrac", type=float, default=0.05, help="Bir bölge toplam ısının en az bu kadarını içermeli")

    i = sub.add_parser("isle", help="TEK KOMUT: tara → bölge sapta → giriş/çıkış → PostgreSQL'e yaz")
    i.add_argument("--video", required=True)
    i.add_argument("--model", default=DEFAULT_MODEL)
    i.add_argument("--yeniden-tara", action="store_true",
                   help="Tespit dosyası varsa bile videoyu yeniden tara")

    args = parser.parse_args()
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "analiz":
        cmd_analiz(args)
    elif args.command == "bolgeler":
        cmd_bolgeler(args)
    elif args.command == "isle":
        cmd_isle(args)


def cmd_isle(args):
    """
    Tek komutla tam akış:
      1. Tara (tespit dosyası yoksa veya --yeniden-tara ile)  [GPU, ısı]
      2. Çoklu işlem bölgesi sapta                            [offline, ısı yok]
      3. Giriş/çıkış hesapla + PostgreSQL'e yaz               [offline, ısı yok]
    """
    base = data_base(args.video)
    det_path = base + "_tespitler.json"

    print("=" * 60)
    print("  ADIM 1/3 — Video taranıyor (tespitler çıkarılıyor)")
    print("=" * 60)
    if os.path.exists(det_path) and not args.yeniden_tara:
        print(f"Tespit dosyası zaten var, atlanıyor: {os.path.basename(det_path)}")
        print("(yeniden taramak için --yeniden-tara ekleyin)")
    else:
        cmd_scan(args)

    print("\n" + "=" * 60)
    print("  ADIM 2/3 — İşlem bölgeleri saptanıyor")
    print("=" * 60)
    cmd_bolgeler(argparse.Namespace(video=args.video, thresh=0.20, minfrac=0.05))

    print("\n" + "=" * 60)
    print("  ADIM 3/3 — Giriş/çıkış hesaplanıyor ve kaydediliyor")
    print("=" * 60)
    cmd_analiz(argparse.Namespace(video=args.video, zone=None, kaydet=True))

    print("\n✔ TAMAMLANDI. Sonuçları panelde görün: http://localhost:8000")


if __name__ == "__main__":
    main()
