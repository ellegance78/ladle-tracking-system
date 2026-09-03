#!/bin/bash
# ============================================================
# TEK KAMERA — Sürekli çalışan canlı takip (7/24 supervisor)
# ============================================================
# pota_takip_sistemi_v2.py'yi bir kameranın canlı (RTSP) akışında SÜREKLİ
# çalışır tutar: süreç MPS/model çökmesi ya da RTSP bağlantı sorunu yüzünden
# sonlanırsa birkaç saniye bekleyip otomatik olarak yeniden başlatır.
# (RTSP bağlantı kopmaları zaten canli_kaynak.py'nin FrameGrabber'ı tarafından
# otomatik tolere edilir — bu betik SADECE process'in kendisi çökerse devreye
# girer, örn. MPS/Metal hatası gibi.)
#
# Durdurmak için: Ctrl+C (açık track'ler "hâlâ işlemde" olarak kaydedilir).
#
# KULLANIM:
#   ./run_live_resilient.sh "<kamera adı>" "<rtsp adresi>"
#
# Örnek:
#   ./run_live_resilient.sh "CAM-01" "rtsp://<user>:<pass>@<camera-host>:554/stream1"
#
# 27 kameranın HEPSİNİ aynı anda başlatmak için baslat_tum_kameralar.sh kullanın
# (her kamera bu betiğin ayrı bir kopyasını, ayrı bir arka plan sürecinde çalıştırır).
# ============================================================

set -u
cd "$(dirname "$0")"

CAMERA="${1:?Kullanım: ./run_live_resilient.sh \"<kamera adı>\" \"<rtsp adresi>\"}"
RTSP="${2:?Kullanım: ./run_live_resilient.sh \"<kamera adı>\" \"<rtsp adresi>\"}"

RESTART_WAIT_SEC=10
MAX_LOG_KEEP=5        # kamera başına en fazla bu kadar eski log dosyası tutulur (disk şişmesin)

LOG_DIR="logs_canli"
mkdir -p "$LOG_DIR"
SAFE_NAME=$(echo "$CAMERA" | tr -c 'A-Za-z0-9_-' '_')
STOP_FLAG="$LOG_DIR/${SAFE_NAME}.stop"
rm -f "$STOP_FLAG"   # önceki bir çalıştırmadan kalmış olabilir

ATTEMPT=1
while true; do
  if [ -f "$STOP_FLAG" ]; then
    echo "[$CAMERA] Durdurma isteği bulundu, çıkılıyor."
    rm -f "$STOP_FLAG"
    break
  fi

  LOG="$LOG_DIR/${SAFE_NAME}_attempt${ATTEMPT}.log"
  echo "=== [$CAMERA] Deneme $ATTEMPT ($(date '+%Y-%m-%d %H:%M:%S')) ==="

  ./venv/bin/python -u pota_takip_sistemi_v2.py track \
    --video "$RTSP" --camera "$CAMERA" \
    > "$LOG" 2>&1
  EXIT=$?

  # durdur_tum_kameralar.sh (ya da elle) STOP_FLAG bırakıp SIGINT gönderdiyse,
  # bu GERÇEK bir durdurma isteğidir — yeniden başlatma, döngüden çık.
  if [ -f "$STOP_FLAG" ]; then
    echo "[$CAMERA] Durduruldu (istek üzerine)."
    rm -f "$STOP_FLAG"
    break
  fi

  echo "[$CAMERA] Süreç beklenmedik şekilde sonlandı (exit=$EXIT). ${RESTART_WAIT_SEC}sn sonra yeniden başlatılacak…"

  # Eski logları buda — sadece en son MAX_LOG_KEEP tanesini tut
  ls -t "$LOG_DIR/${SAFE_NAME}_attempt"*.log 2>/dev/null | tail -n +$((MAX_LOG_KEEP + 1)) | xargs -r rm -f

  ATTEMPT=$((ATTEMPT + 1))
  sleep "$RESTART_WAIT_SEC"
done
