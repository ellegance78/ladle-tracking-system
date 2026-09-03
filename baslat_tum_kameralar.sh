#!/bin/bash
# ============================================================
# TÜM KAMERALAR — kameralar.json'daki her kamera için ayrı bir
# run_live_resilient.sh süreci başlatır (her kamera kendi arka plan
# sürecinde, kendi RTSP akışını dinler — pota_canli_takip.py'de
# belirtilen "her kamera ayrı süreç" prensibiyle aynı).
# ============================================================
# KULLANIM:
#   ./baslat_tum_kameralar.sh                 (kameralar.json'daki hepsini başlatır)
#   ./baslat_tum_kameralar.sh kameralar.json   (başka bir dosya vermek için)
#
# Durdurmak için: ./durdur_tum_kameralar.sh
# ============================================================

set -u
cd "$(dirname "$0")"

CONFIG="${1:-kameralar.json}"
LOG_DIR="logs_canli"
PIDS_FILE="$LOG_DIR/pids.txt"

if [ ! -f "$CONFIG" ]; then
  echo "✘ Bulunamadı: $CONFIG"
  exit 1
fi

mkdir -p "$LOG_DIR"
: > "$PIDS_FILE"

venv/bin/python3 -c "
import json, sys
d = json.load(open('$CONFIG'))
for k in d['kameralar']:
    print(k['kamera'] + '\t' + k['rtsp'])
" | while IFS=$'\t' read -r KAMERA RTSP; do
  if [[ "$RTSP" == *"KULLANICI:SIFRE@IP_ADRESI"* ]]; then
    echo "⚠ Atlandı (RTSP adresi henüz girilmemiş): $KAMERA"
    continue
  fi
  echo "▶ Başlatılıyor: $KAMERA"
  nohup ./run_live_resilient.sh "$KAMERA" "$RTSP" >> "$LOG_DIR/baslatici.log" 2>&1 &
  printf "%s\t%s\n" "$!" "$KAMERA" >> "$PIDS_FILE"
  sleep 1   # aynı anda hepsi model yüklemeye başlamasın diye kademeli aç
done

echo ""
echo "Tüm kameralar başlatıldı (PID listesi: $PIDS_FILE)."
echo "Her kameranın logu: $LOG_DIR/<kamera>_attemptN.log"
echo "Durdurmak için: ./durdur_tum_kameralar.sh"
