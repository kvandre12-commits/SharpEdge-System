import sqlite3

DB_PATH = "data/spy_truth.db"

TARIFF_DATES = [
    ("2024-04-03", "Tariff enforcement announcement"),
    ("2024-04-11", "Presidential tariff threat"),
    ("2024-04-24", "Trade policy remarks"),
]

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Create table if missing
    cur.execute("""
    CREATE TABLE IF NOT EXISTS overlays_daily (
      date TEXT NOT NULL,
      symbol TEXT NOT NULL,
      overlay_type TEXT NOT NULL,
      overlay_strength REAL NOT NULL,
      notes TEXT,
      PRIMARY KEY (symbol, date, overlay_type)
    );
    """)

    for d, note in TARIFF_DATES:
        cur.execute("""
        INSERT OR IGNORE INTO overlays_daily
        (date, symbol, overlay_type, overlay_strength, notes)
        VALUES (?, 'SPY', 'tariff', 1.0, ?)
        """, (d, note))

    con.commit()
    con.close()
    print("Tariff overlays inserted.")

if __name__ == "__main__":
    main()
