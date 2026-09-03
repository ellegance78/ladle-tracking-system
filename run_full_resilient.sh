#!/bin/bash
# 14sa50dk'lık videoyu, MPS/Metal çökmesi olursa otomatik devam ederek işler.
set -u
cd <proje-dizini>

VIDEO="<proje-dizini>"
CAMERA="TEST-14SAAT-v2-full"
TOTAL=53405
OFFSET=0
ATTEMPT=1

while (( $(echo "$OFFSET < $TOTAL" | bc -l) )); do
  LOG="full_run_attempt${ATTEMPT}.log"
  RESUME_FLAG=""
  if [ "$ATTEMPT" -gt 1 ]; then
    RESUME_FLAG="--resume"
  fi
  echo "=== Attempt $ATTEMPT, start-offset=$OFFSET ($(date '+%H:%M:%S')) ==="
  ./venv/bin/python -u pota_takip_sistemi_v2.py track \
    --video "$VIDEO" --camera "$CAMERA" --stride 8 \
    --start-offset "$OFFSET" $RESUME_FLAG \
    > "$LOG" 2>&1
  EXIT=$?
  if [ "$EXIT" -eq 0 ]; then
    echo "Tamamlandı (attempt $ATTEMPT, exit 0)."
    break
  fi
  echo "Çöktü (exit=$EXIT), son işlenen pozisyon $LOG içinde aranıyor..."
  LAST=$(grep -oE '\([0-9]+\.[0-9]+\. sn\)' "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  if [ -z "$LAST" ]; then
    echo "Son pozisyon bulunamadı, duruyorum."
    break
  fi
  OFFSET=$(echo "$LAST - 150" | bc)
  if (( $(echo "$OFFSET < 0" | bc -l) )); then OFFSET=0; fi
  ATTEMPT=$((ATTEMPT+1))
done

echo "=== BİTTİ (toplam $ATTEMPT deneme) ==="
