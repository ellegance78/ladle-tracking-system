#!/bin/bash
# ============================================================
# baslat_tum_kameralar.sh ile başlatılan tüm kamera süreçlerini
# ZARİF şekilde durdurur (açık track'ler "hâlâ işlemde" olarak kaydedilir).
# ============================================================
# Her kamera için: önce bir "durdurma bayrağı" bırakılır (run_live_resilient.sh
# bunu görünce süreç bitince YENİDEN BAŞLATMAZ), sonra o kameranın python
# sürecine SIGINT gönderilir (v2'nin kendi signal handler'ı ile aynı — Ctrl+C
# ile aynı etkiyi yapar).
# ============================================================

set -u
cd "$(dirname "$0")"

LOG_DIR="logs_canli"
PIDS_FILE="$LOG_DIR/pids.txt"

if [ ! -f "$PIDS_FILE" ]; then
  echo "PID listesi bulunamadı ($PIDS_FILE) — zaten çalışmıyor olabilir."
  exit 0
fi

while IFS=$'\t' read -r PID KAMERA; do
  SAFE_NAME=$(echo "$KAMERA" | tr -c 'A-Za-z0-9_-' '_')
  touch "$LOG_DIR/${SAFE_NAME}.stop"
  echo "⏹  Durduruluyor: $KAMERA"
  # run_live_resilient.sh'nin başlattığı python sürecine doğrudan SIGINT gönder
  # (wrapper bash script'e değil — sinyal iletiminin garanti olduğu yer burası).
  pkill -INT -f "pota_takip_sistemi_v2.py track --video .* --camera $KAMERA" 2>/dev/null
done < "$PIDS_FILE"

echo ""
echo "Sinyaller gönderildi. Açık track'lerin kaydedilmesi için birkaç saniye bekleyin,"
echo "sonra 'ps aux | grep pota_takip_sistemi_v2' ile temizlendiğini doğrulayabilirsiniz."
