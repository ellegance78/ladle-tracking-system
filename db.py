"""
Pota Takip - Veritabanı Katmanı
================================
Prototipte SQLite (kurulumsuz, yerel test), ana sistemde PostgreSQL.
Hangisinin kullanılacağı POTA_DB_URL ortam değişkeniyle belirlenir:

    (boş / tanımsız)                          → SQLite (pota_takip.db)
    postgresql://user:sifre@host:5432/potadb  → PostgreSQL

PostgreSQL için gereksinim: pip install psycopg2-binary
"""

import os
import sqlite3

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(PROJECT_DIR, "pota_takip.db")

DB_URL = os.environ.get("POTA_DB_URL", "").strip()
IS_POSTGRES = DB_URL.startswith("postgres")

# SQLite '?' , PostgreSQL '%s' placeholder kullanır
PLACEHOLDER = "%s" if IS_POSTGRES else "?"

# entry_partial / exit_partial:
#   Pota video BAŞLADIĞINDA zaten kadrajdaysa, gerçek giriş anı kayıttan
#   öncedir — bilinmiyor. Aynı şekilde video BİTTİĞİNDE hâlâ kadrajdaysa
#   gerçek çıkış anı bilinmiyor. Bu durumlarda süre GERÇEĞİNDEN KISA çıkar.
#   Bu bayraklar olmadan sistem eksik süreyi tam süre gibi kaydeder ve
#   raporlar sessizce yanlış olur.

SCHEMA_SQLITE = """
    CREATE TABLE IF NOT EXISTS pota_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT NOT NULL,
        duration_sec REAL NOT NULL,
        entry_partial INTEGER DEFAULT 0,
        exit_partial INTEGER DEFAULT 0,
        video_file TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
"""

SCHEMA_POSTGRES = """
    CREATE TABLE IF NOT EXISTS pota_events (
        id SERIAL PRIMARY KEY,
        camera TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        entry_time TIMESTAMP NOT NULL,
        exit_time TIMESTAMP NOT NULL,
        duration_sec REAL NOT NULL,
        entry_partial BOOLEAN DEFAULT FALSE,
        exit_partial BOOLEAN DEFAULT FALSE,
        video_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""


def connect():
    """Veritabanına bağlanır ve tabloyu (yoksa) oluşturur."""
    if IS_POSTGRES:
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute(SCHEMA_POSTGRES)
        conn.commit()
        print(f"Veritabanı: PostgreSQL ({DB_URL.split('@')[-1]})")
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.execute(SCHEMA_SQLITE)
        # Eski veritabanında bu sütunlar yoksa ekle
        for col in ("entry_partial", "exit_partial"):
            try:
                conn.execute(f"ALTER TABLE pota_events ADD COLUMN {col} INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # zaten var
        conn.commit()
        print(f"Veritabanı: SQLite ({SQLITE_PATH})")
    return conn


def save_event(conn, camera, track_id, entry_time, exit_time, duration_sec, video_file,
               entry_partial=False, exit_partial=False):
    sql = (
        f"INSERT INTO pota_events "
        f"(camera, track_id, entry_time, exit_time, duration_sec, entry_partial, exit_partial, video_file) "
        f"VALUES ({PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER}, "
        f"{PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER}, {PLACEHOLDER})"
    )
    params = (
        camera, track_id,
        entry_time.isoformat(sep=" ", timespec="seconds"),
        exit_time.isoformat(sep=" ", timespec="seconds"),
        round(duration_sec, 1),
        entry_partial if IS_POSTGRES else int(entry_partial),
        exit_partial if IS_POSTGRES else int(exit_partial),
        video_file,
    )
    if IS_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        conn.execute(sql, params)
    conn.commit()


def fetch_events(conn, camera=None):
    sql = (
        "SELECT camera, track_id, entry_time, exit_time, duration_sec, "
        "entry_partial, exit_partial, video_file FROM pota_events"
    )
    params = ()
    if camera:
        sql += f" WHERE camera = {PLACEHOLDER}"
        params = (camera,)
    sql += " ORDER BY entry_time"

    if IS_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    cur = conn.execute(sql, params)
    return cur.fetchall()
