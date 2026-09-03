"""
============================================================
CANLI KAYNAK — Paylaşılan RTSP/dosya kare okuyucu
============================================================
`pota_canli_takip.py` ve `pota_takip_sistemi_v2.py` (canlı mod) tarafından
ortak kullanılır. Tek sorumluluğu: bir video kaynağını (RTSP akışı ya da
test için yerel dosya) ayrı bir thread'de sürekli okumak, HER ZAMAN en son
kareyi tutmak (işleme yavaş kalsa bile akış birikmez, gecikme olmaz) ve
bağlantı koparsa otomatik yeniden bağlanmak.
============================================================
"""

import threading
import time

RECONNECT_WAIT_SEC = 5.0   # RTSP koptuğunda yeniden bağlanma bekleme süresi


class FrameGrabber(threading.Thread):
    """
    RTSP akışını ayrı thread'de sürekli okur, HER ZAMAN en son kareyi tutar.
    Böylece işleme yavaş olsa bile akış birikmez (gecikme olmaz) ve
    bağlantı koparsa otomatik yeniden bağlanır.
    """
    def __init__(self, kaynak, hiz=1.0):
        super().__init__(daemon=True)
        self.kaynak = kaynak
        # RTSP/HTTP akışı mı yoksa yerel dosya mı?
        self.is_stream = str(kaynak).lower().startswith(("rtsp://", "http://", "https://", "tcp://", "udp://"))
        self.hiz = hiz               # sadece dosya testinde: kaç kat hızlı oynatılsın
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.bitti = False           # dosya sonuna gelindi (canlıda hiç True olmaz)

    def run(self):
        import cv2
        while self.running:
            cap = cv2.VideoCapture(self.kaynak)
            if self.is_stream:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # canlıda gecikmeyi azalt
                except Exception:
                    pass

            if not cap.isOpened():
                self.connected = False
                if not self.is_stream:
                    print(f"✘ Dosya açılamadı: {self.kaynak}")
                    self.bitti = True
                    return
                print(f"⚠ Kaynağa bağlanılamadı, {RECONNECT_WAIT_SEC:.0f} sn sonra tekrar denenecek…")
                time.sleep(RECONNECT_WAIT_SEC)
                continue

            self.connected = True
            print("✔ " + ("Akışa bağlanıldı." if self.is_stream else "Dosya açıldı (canlı simülasyon)."))

            # dosya testinde gerçek zaman hızını taklit et
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            kare_araligi = (1.0 / fps) / max(0.1, self.hiz)

            while self.running:
                ok, f = cap.read()
                if not ok:
                    if self.is_stream:
                        print("⚠ Akış koptu, yeniden bağlanılıyor…")
                        self.connected = False
                        break
                    else:
                        print("Dosya bitti.")
                        self.bitti = True
                        cap.release()
                        return
                with self.lock:
                    self.frame = f
                if not self.is_stream:
                    time.sleep(kare_araligi)   # dosyayı gerçek zaman hızında oynat

            cap.release()
            if self.is_stream:
                time.sleep(RECONNECT_WAIT_SEC)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
