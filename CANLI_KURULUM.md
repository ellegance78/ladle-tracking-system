# Pota Takip — Canlı/Çoklu Kamera Kurulum ve Devir Dokümanı

Bu doküman, `pota_takip_sistemi_v2.py`'nin 27 canlı kameralı sisteme
entegrasyonu için mühendise devir amacıyla hazırlanmıştır.

## 1. Ne yapıyor, mimari özeti

`pota_takip_sistemi_v2.py` tek bir dosyada:
- YOLO (pota + pota_arabası, 2 sınıf) + ByteTrack ile takip yapar,
- Konum bazlı, kapı-doğrulamalı bir mantıkla ham track parçalarını
  mantıksal "pota" olaylarına birleştirir (bkz. `close_track()` — kod
  içindeki yorumlar birleştirme kurallarının GEREKÇESİNİ ayrıntılı anlatır,
  bunlar gerçek veri üzerinde bulunan hatalardan çıkarılmış dersledir,
  değiştirmeden önce okuyun),
- Sonucu PostgreSQL `pota_events` tablosuna CANLI yazar (video/akış
  işlenirken, sonunu beklemeden).

Aynı script hem **dosya modunda** (`--video kayit.mp4`, geriye dönük
uyumlu, hiçbir şey değişmedi) hem **canlı modda** çalışır:

```bash
# Dosya (offline analiz/test)
python pota_takip_sistemi_v2.py track --video kayit.mp4 --camera "CAM-01"

# Canlı (RTSP)
python pota_takip_sistemi_v2.py track \
  --video "rtsp://<user>:<pass>@<camera-host>:554/stream1" \
  --camera "CAM-01"
```

`--video` değeri `rtsp://`, `http://`, `https://`, `tcp://` ya da `udp://`
ile başlıyorsa OTOMATİK canlı moda geçer (kamera adı bu modda ZORUNLU).
Standart olmayan bir kaynak için `--live` bayrağıyla canlı modu elle
zorlayabilirsiniz.

Canlı modda kare okuma `canli_kaynak.py`'deki `FrameGrabber` sınıfıyla
yapılır: ayrı thread'de sürekli okur, her zaman EN GÜNCEL kareyi tutar
(işleme yavaş kalırsa gecikme birikmez), bağlantı koparsa otomatik
yeniden bağlanır. `pota_canli_takip.py` (eski, basit bölge-doluluk
sistemi — ID takibi yok) da aynı sınıfı kullanır.

## 2. Kurulum

```bash
python3.11 -m venv venv
venv/bin/pip install ultralytics opencv-python psycopg2-binary flask
```

PostgreSQL bağlantısı üç yoldan biriyle verilir (öncelik sırasıyla):
1. `POTA_DB_URL="postgresql://kullanici:sifre@sunucu:5432/potadb"`
2. `DB_HOST` / `DB_NAME` / `DB_USER` / `DB_PASS` / `DB_PORT`
3. Hiçbiri yoksa `DEFAULT_DB` (kod içinde, localhost/pota_db)

Model: `runs/pota_arabasi_v2/weights/best.pt` (varsayılan, `--model` ile
değiştirilebilir). Sınıflar: 0=pota, 1=pota_arabası.

## 3. Yeni kamera ekleme (HER kamera için tek seferlik kalibrasyon)

Her kameranın açısı farklı olduğundan giriş/çıkış kapıları ve varsa
bilinen gürültü bölgeleri (yansıma, sabit ışık kaynağı vb.) o kameraya
özel çizilmelidir.

1. O kameradan örnek bir kare alıp `veri/<KAMERA_ADI>_frame.jpg` olarak
   kaydedin. Dosya adı `NVR_<kamera adı>_<14 haneli tarih>` kalıbını
   içermeli (örn. `NVR_CAM-01_20260713042959_frame.jpg`)
   — sistem kamera adını bu kalıptan çıkarır.
2. `python bolge_editor.py` çalıştırın, tarayıcıda **http://localhost:8001**
   açın, ilgili kamerayı seçin.
3. Toolbar'daki tip seçiciden **Kapı (Sol)** / **Kapı (Sağ)** seçip
   potanın kadraja giriş/çıkış yaptığı gerçek rayları/boşlukları çizin.
   Varsa bilinen gürültü kaynaklarını **Geçersiz Bölge** olarak işaretleyin.
   **Kaydet**'e basın.
4. Bu kamera için `--camera "<KAMERA_ADI>"` ile track komutunu çalıştırdığınızda
   kalibrasyon otomatik yüklenir (konsolda "Kamera kalibrasyonu: ..." satırını
   göreceksiniz — bulunamazsa hardcoded varsayılanlara düşer, bu SADECE
   `TEST-14SAAT-v2-full` için doğrulanmıştır, başka kamerada yanlış olabilir —
   **her yeni kamera için bu adımı atlamayın**).

`veriseti_potano_v6 kopyası` klasöründeki mevcut eğitim verisinden 29
kameranın birer örnek karesi zaten `veri/` klasörüne kopyalanmış ve
`kameralar.json`'a eklenmiş durumda — sadece RTSP adreslerini girip
kapı/geçersiz bölgeleri çizmeniz yeterli.

## 4. Çalıştırma

**Tek kamera (test için):**
```bash
./run_live_resilient.sh "CAM-01" "rtsp://<user>:<pass>@<camera-host>:554/stream1"
```
Bu, MPS/model çökerse otomatik yeniden başlatan bir supervisor'dur (dosya
modundaki `run_full_resilient.sh` ile aynı mantık, sonsuz döngü hâli).

**Tüm kameralar:**
1. `kameralar.json`'da her kameranın `"rtsp"` alanını gerçek adresle
   doldurun (placeholder `KULLANICI:SIFRE@IP_ADRESI` içeren satırlar
   başlatıcı tarafından atlanır).
2. `./baslat_tum_kameralar.sh` — her kamerayı ayrı arka plan sürecinde,
   kademeli (1sn arayla) başlatır.
3. Durdurmak için `./durdur_tum_kameralar.sh` (zarif durdurma — açık
   track'ler "hâlâ işlemde" olarak kaydedilir, süreçler yeniden başlamaz).

Loglar: `logs_canli/<kamera>_attemptN.log` (kamera başına en son 5 deneme
tutulur, eskiler otomatik silinir).

## 5. Entegrasyon noktası: PostgreSQL

Sisteme kod seviyesinde bağımlılık GEREKMEZ — mühendisin (GPT tabanlı)
sistemi doğrudan `pota_events` tablosunu okuyarak entegre olabilir:

```sql
CREATE TABLE pota_events (
    id SERIAL PRIMARY KEY,
    camera TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    duration_sec REAL NOT NULL,
    entry_partial BOOLEAN DEFAULT FALSE,   -- kayıt/izleme BAŞLARKEN pota zaten kadrajdaydı
    exit_partial BOOLEAN DEFAULT FALSE,    -- kayıt/izleme BİTERKEN pota hâlâ kadrajdaydı (süre kesin değil)
    zone_no INTEGER DEFAULT 1,             -- çift taraflı düzenekler için (şu an hep 1)
    video_file TEXT,                       -- canlı modda hep "CANLI", dosya modunda dosya adı
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Satırlar canlı olarak INSERT/UPDATE edilir (bir pota kadrajdayken süre
her birkaç saniyede bir UPDATE ile güncellenir — `exit_time` "kesinleşmiş"
değil, en son bilinen konumdur; pota gerçekten çıktığında son UPDATE
kesinleşir). Mevcut `pota_dashboard.py` (Flask, port 8000) bu tabloyu
canlı gösteren basit bir örnek — mühendis kendi arayüzü için referans alabilir.

## 6. Bilinen sınırlamalar / dikkat edilmesi gerekenler

- **Bellek**: `segments`/`pending` listeleri süreç boyunca sadece büyür
  (asla budanmaz). Bir kamera günlerce/haftalarca kesintisiz çalışırsa bu
  bellek kullanımı zamanla artar. Bugünkü kapsamda ele alınmadı — uzun
  vadeli 7/24 kullanımda periyodik yeniden başlatma (örn. günlük, boş bir
  saatte) ya da `pending` listesinin eski/kapanmış girdilerini periyodik
  temizleyen bir mekanizma eklenmesi önerilir.
- **`--save` canlı modda yok**: süresiz akışı tek bir MP4'e yazmak uygun
  değil, otomatik olarak yok sayılır.
- **`ham_track.json` dökümü canlı modda yazılmaz**: dosya-modu debug/analiz
  amaçlıdır, canlıda anlamsız (bkz. yukarıdaki bellek notu).
- **Gate/geçersiz-bölge kalibrasyonu kamera-bazlıdır**: yeni bir kamera
  eklerken adım 3'ü atlamayın, aksi halde `TEST-14SAAT-v2-full` için
  kalibre edilmiş varsayılanlar kullanılır (o kameranın açısına göre yanlış
  olabilir).
- **MERGE_TIME_GAP / GATE_NONE_MAX_GAP / MERGE_DIST_FRAC vb. sabitler**
  (kod başındaki KONFİGÜRASYON bölümü) TÜM kameralar için ORTAKTIR —
  bunlar pota'nın fiziksel davranışıyla ilgilidir (ne kadar süre
  duman/buhar arkasında kalabilir vb.), kamera açısıyla değil. Farklı bir
  süreçte (örn. çok daha uzun/kısa işlem süreleri olan bir istasyon) bu
  sabitlerin de gözden geçirilmesi gerekebilir.

## 7. Sorun giderme

- Süreç MPS/Metal hatasıyla çöküyorsa: `run_live_resilient.sh` zaten
  otomatik yeniden başlatır, `logs_canli/` içindeki loglardan sıklığı
  takip edin.
- RTSP bağlantısı sık kopuyorsa: `canli_kaynak.py`'deki
  `RECONNECT_WAIT_SEC` (varsayılan 5sn) ayarlanabilir.
- Bir kameranın kalibrasyonu yanlış görünüyorsa (örn. gerçek çıkışlar
  "farklı pota" sayılıyor): `bolge_editor.py`'de o kameranın kapı
  bölgelerini kontrol edin/yeniden çizin.
