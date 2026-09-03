"""
Pota Takip - Kayıt Raporu
==========================
Veritabanındaki pota giriş/çıkış olaylarını tablo halinde listeler.

Kullanım:
    python3 report.py                           # tüm kayıtlar
    python3 report.py --camera "CAM-09"  # tek kamera
"""

import argparse

import db


def main():
    parser = argparse.ArgumentParser(description="Pota olay kayıtlarını listele")
    parser.add_argument("--camera", default=None, help="Sadece bu kameranın kayıtları")
    args = parser.parse_args()

    conn = db.connect()
    rows = db.fetch_events(conn, camera=args.camera)

    if not rows:
        print("\nKayıt bulunamadı.")
        return

    print(f"\n{'Kamera':<24} {'ID':>3} {'Giriş':<10} {'Çıkış':<10} {'Süre':>8}  Not")
    print("-" * 78)

    partial_count = 0
    for camera, tid, entry, exit_, duration, e_partial, x_partial, _video in rows:
        # Sadece saat kısmını göster (tarih zaten aynı gün)
        entry_s = str(entry)[11:19]
        exit_s = str(exit_)[11:19]

        note = ""
        if e_partial and x_partial:
            note = "⚠ giriş+çıkış kayıt dışında"
        elif e_partial:
            note = "⚠ kayıt başlarken zaten kadrajdaydı (süre daha uzun)"
        elif x_partial:
            note = "⚠ kayıt biterken hâlâ kadrajdaydı (süre daha uzun)"

        if note:
            partial_count += 1

        print(f"{camera:<24} {tid:>3} {entry_s:<10} {exit_s:<10} {duration:>6.1f}s  {note}")

    print(f"\nToplam {len(rows)} olay.")
    if partial_count:
        print(f"{partial_count} tanesinin süresi eksik — pota, kayıt penceresinin "
              f"dışında girmiş veya çıkmış.")
        print("Tam süre için potanın giriş ve çıkışını kapsayan bir kayıt gerekir.")


if __name__ == "__main__":
    main()
