"""
============================================================
POTA TAKİP — İzleme Paneli (Web UI)
============================================================
pota_takip_sistemi.py'nin PostgreSQL'e yazdığı olayları canlı gösteren
web panosu. Yeni videolar işlendikçe tablo otomatik güncellenir.

KURULUM
    pip install flask psycopg2-binary

ÇALIŞTIRMA
    python pota_dashboard.py
    # sonra tarayıcıda:  http://localhost:8000

Veritabanı bağlantısı pota_takip_sistemi.py ile aynı ortam değişkenlerini
kullanır (POTA_DB_URL veya DB_HOST/DB_NAME/...). Varsayılan: pota_db.
============================================================
"""

import os
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, jsonify, render_template_string

DEFAULT_DB = {
    "host": "localhost",
    "dbname": "pota_db",
    "user": os.environ.get("USER", "postgres"),
    "password": "",
    "port": "5432",
}

app = Flask(__name__)


def get_conn():
    url = os.environ.get("POTA_DB_URL", "").strip()
    if url:
        return psycopg2.connect(url, cursor_factory=RealDictCursor)
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", DEFAULT_DB["host"]),
        dbname=os.environ.get("DB_NAME", DEFAULT_DB["dbname"]),
        user=os.environ.get("DB_USER", DEFAULT_DB["user"]),
        password=os.environ.get("DB_PASS", DEFAULT_DB["password"]),
        port=os.environ.get("DB_PORT", DEFAULT_DB["port"]),
        cursor_factory=RealDictCursor,
    )


def table_exists(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.pota_events') IS NOT NULL AS ok")
        return cur.fetchone()["ok"]


@app.route("/api/data")
def api_data():
    """Panonun beslendiği JSON: özet istatistikler + olay listesi."""
    try:
        conn = get_conn()
    except Exception as e:
        return jsonify({"error": f"Veritabanına bağlanılamadı: {e}"}), 500

    try:
        if not table_exists(conn):
            return jsonify({"stats": None, "events": [], "cameras": [],
                            "empty": True})

        with conn.cursor() as cur:
            # Özet istatistikler
            cur.execute("""
                SELECT
                    COUNT(*)                                    AS toplam,
                    COUNT(DISTINCT camera)                      AS kamera_sayisi,
                    COALESCE(AVG(duration_sec), 0)              AS ort_sure,
                    COALESCE(MAX(duration_sec), 0)              AS max_sure,
                    COUNT(*) FILTER (WHERE entry_partial OR exit_partial) AS eksik
                FROM pota_events
            """)
            stats = cur.fetchone()

            # Kamera bazlı özet
            cur.execute("""
                SELECT camera,
                       COUNT(*)                       AS olay,
                       COALESCE(AVG(duration_sec), 0) AS ort_sure,
                       MAX(entry_time)                AS son_kayit
                FROM pota_events
                GROUP BY camera
                ORDER BY son_kayit DESC
            """)
            cameras = cur.fetchall()

            # Son olaylar (en yeni üstte)
            cur.execute("""
                SELECT id, camera, track_id, entry_time, exit_time,
                       duration_sec, entry_partial, exit_partial, video_file
                FROM pota_events
                ORDER BY entry_time DESC
                LIMIT 200
            """)
            events = cur.fetchall()

        # datetime -> ISO string
        for e in events:
            e["entry_time"] = e["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
            e["exit_time"] = e["exit_time"].strftime("%H:%M:%S")
        for c in cameras:
            c["son_kayit"] = c["son_kayit"].strftime("%d.%m %H:%M") if c["son_kayit"] else "—"

        return jsonify({
            "stats": stats, "events": events, "cameras": cameras,
            "empty": False,
            "updated": datetime.now().strftime("%H:%M:%S"),
        })
    finally:
        conn.close()


@app.route("/")
def index():
    return render_template_string(PAGE)


# ============================================================
# ARAYÜZ (tek sayfa — HTML/CSS/JS gömülü)
# ============================================================

PAGE = r"""
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pota Takip — İzleme Paneli</title>
<style>
  :root {
    --bg: #0e1113; --card: #171b1e; --card2: #1c2124;
    --fg: #e8ecea; --muted: #98a5a0; --border: #2a3033; --border2: #3b4347;
    --accent: #2f9c67; --warn: #fbbf24; --warn-bg: #2a2113;
    --temp: #fb923c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased; font-size: 14px;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 20px 16px 60px; }

  header { display: flex; align-items: center; justify-content: space-between;
           margin-bottom: 20px; flex-wrap: wrap; gap: 8px; }
  h1 { font-size: 18px; font-weight: 600; letter-spacing: -.01em; }
  h1 .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
            background: var(--accent); margin-right: 8px; vertical-align: middle; }
  .updated { font-size: 12px; color: var(--muted); }
  .updated b { color: var(--fg); font-variant-numeric: tabular-nums; }

  /* İstatistik kutuları */
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
           gap: 10px; margin-bottom: 20px; }
  .stat { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 14px 16px; }
  .stat .k { font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
             color: var(--muted); font-weight: 600; }
  .stat .v { font-size: 26px; font-weight: 600; margin-top: 4px;
             font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .stat .v small { font-size: 14px; color: var(--muted); font-weight: 400; }

  section { margin-bottom: 26px; }
  h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase;
       letter-spacing: .04em; margin-bottom: 10px; }

  /* Kamera filtresi */
  .cams { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
  .cam { background: var(--card); border: 1px solid var(--border); color: var(--muted);
         border-radius: 6px; padding: 5px 11px; font-size: 13px; cursor: pointer;
         transition: .12s; user-select: none; }
  .cam:hover { border-color: var(--border2); color: var(--fg); }
  .cam.active { background: var(--accent); border-color: var(--accent); color: #06120c;
                font-weight: 600; }
  .cam .n { opacity: .6; margin-left: 5px; font-variant-numeric: tabular-nums; }

  /* Tablo */
  .tablewrap { background: var(--card); border: 1px solid var(--border);
               border-radius: 8px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
       color: var(--muted); font-weight: 600; padding: 10px 14px;
       border-bottom: 1px solid var(--border); background: var(--card2); }
  td { padding: 10px 14px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--card2); }
  td.cam-name { font-variant-numeric: normal; }
  .id { display: inline-block; min-width: 26px; text-align: center; padding: 1px 7px;
        border-radius: 5px; background: var(--card2); border: 1px solid var(--border2);
        font-weight: 600; font-size: 12px; }
  .dur { font-weight: 600; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 5px; font-size: 12px;
           background: var(--warn-bg); color: var(--warn);
           border: 1px solid rgba(251,191,36,.25); }
  .ok { color: var(--muted); font-size: 12px; }

  .empty, .err { text-align: center; padding: 50px 20px; color: var(--muted); }
  .err { color: #f87171; }
  .hint { font-size: 12px; color: var(--muted); margin-top: 8px; line-height: 1.6; }
  code { background: var(--card2); padding: 1px 6px; border-radius: 4px;
         border: 1px solid var(--border); font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="dot"></span>Pota Takip — İzleme Paneli</h1>
    <div class="updated">Son güncelleme: <b id="upd">—</b> · <span id="live">otomatik yenileniyor</span></div>
  </header>

  <div id="content"><div class="empty">Yükleniyor…</div></div>
</div>

<script>
let selectedCam = null;
let allEvents = [];

async function load() {
  try {
    const r = await fetch('/api/data');
    const d = await r.json();
    if (d.error) { showError(d.error); return; }
    render(d);
    document.getElementById('upd').textContent = d.updated || '—';
  } catch (e) {
    showError('Panele veri gelmiyor: ' + e.message);
  }
}

function showError(msg) {
  document.getElementById('content').innerHTML =
    '<div class="err">' + msg + '</div>';
}

function fmt(sec) {
  sec = Math.round(sec);
  if (sec < 60) return sec + ' sn';
  const m = Math.floor(sec/60), s = sec%60;
  return m + ' dk ' + (s ? s + ' sn' : '').trim();
}

function render(d) {
  if (d.empty || !d.stats || d.stats.toplam == 0) {
    document.getElementById('content').innerHTML =
      '<div class="empty">Henüz kayıt yok.<div class="hint">' +
      'Bir video işleyin:<br><code>python pota_takip_sistemi.py track --video "kayit.mp4"</code>' +
      '</div></div>';
    return;
  }

  const s = d.stats;
  allEvents = d.events;

  let html = '';

  // İstatistikler
  html += '<div class="stats">';
  html += stat('Toplam olay', s.toplam);
  html += stat('Kamera', s.kamera_sayisi);
  html += stat('Ortalama süre', fmt(s.ort_sure));
  html += stat('En uzun', fmt(s.max_sure));
  html += stat('Süresi eksik', s.eksik + ' <small>/ ' + s.toplam + '</small>');
  html += '</div>';

  // Kamera filtresi
  html += '<section><h2>Kameralar</h2><div class="cams">';
  html += '<div class="cam ' + (selectedCam===null?'active':'') +
          '" onclick="filterCam(null)">Tümü<span class="n">' + s.toplam + '</span></div>';
  for (const c of d.cameras) {
    html += '<div class="cam ' + (selectedCam===c.camera?'active':'') +
            '" onclick="filterCam(\'' + c.camera.replace(/'/g,"\\'") + '\')">' +
            c.camera + '<span class="n">' + c.olay + '</span></div>';
  }
  html += '</div></section>';

  // Olay tablosu
  html += '<section><h2>Olaylar</h2>' + renderTable() + '</section>';

  document.getElementById('content').innerHTML = html;
}

function stat(k, v) {
  return '<div class="stat"><div class="k">' + k + '</div><div class="v">' + v + '</div></div>';
}

function renderTable() {
  let rows = allEvents;
  if (selectedCam !== null) rows = rows.filter(e => e.camera === selectedCam);

  if (rows.length === 0) return '<div class="empty">Bu kamerada kayıt yok.</div>';

  let h = '<div class="tablewrap"><table><thead><tr>' +
    '<th>Kamera</th><th>Pota</th><th>Giriş</th><th>Çıkış</th><th>Süre</th><th>Durum</th>' +
    '</tr></thead><tbody>';

  for (const e of rows) {
    let durum = '<span class="ok">tam kayıt</span>';
    if (e.entry_partial && e.exit_partial)
      durum = '<span class="badge">giriş+çıkış kayıt dışında</span>';
    else if (e.entry_partial)
      durum = '<span class="badge">girişte zaten kadrajda</span>';
    else if (e.exit_partial)
      durum = '<span class="badge">çıkışta hâlâ kadrajda</span>';

    h += '<tr>' +
      '<td class="cam-name">' + e.camera + '</td>' +
      '<td><span class="id">#' + e.track_id + '</span></td>' +
      '<td>' + e.entry_time + '</td>' +
      '<td>' + e.exit_time + '</td>' +
      '<td class="dur">' + fmt(e.duration_sec) + '</td>' +
      '<td>' + durum + '</td>' +
    '</tr>';
  }
  h += '</tbody></table></div>';
  return h;
}

function filterCam(cam) {
  selectedCam = cam;
  load();
}

load();
setInterval(load, 8000);   // 8 saniyede bir otomatik yenile
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  Pota Takip İzleme Paneli")
    print("  Tarayıcıda aç:  http://localhost:8000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False)
