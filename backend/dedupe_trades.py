"""
Database Deduplication Script for Paper-Trading-Bot
Cleans duplicate trade records from paper_trading.db resulting from prior backfill operations.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_trading.db"

def deduplicate_trades():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    before_count = cursor.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    print(f"Initial total rows in trades table: {before_count}")

    # Delete duplicates keeping the earliest row (MIN(id)) for each unique trade
    cursor.execute("""
        DELETE FROM trades
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM trades
            GROUP BY strategy_id, price, quantity, timestamp
        )
    """)

    conn.commit()

    after_count = cursor.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    deleted_count = before_count - after_count

    print(f"Deduplication Complete!")
    print(f"Rows deleted: {deleted_count}")
    print(f"Remaining unique trades: {after_count}")

    # Verify reconciliation
    v4_pnls = cursor.execute("""
        SELECT pnl FROM trades
        WHERE strategy_id = 'v4' AND pnl IS NOT NULL
    """).fetchall()

    v4_net_pnl = sum(p[0] for p in v4_pnls)
    print(f"Reconciled V4 Net Closed PnL: ${v4_net_pnl:.2f} across {len(v4_pnls)} closed trades")

    conn.close()

if __name__ == "__main__":
    deduplicate_trades()
