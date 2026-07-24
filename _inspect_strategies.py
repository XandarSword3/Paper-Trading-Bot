import sqlite3
c = sqlite3.connect('data/paper_trading.db')
cur = c.cursor()
print(cur.execute("SELECT id, name, timeframe FROM strategies").fetchall())
print(cur.execute("SELECT strategy_id, category, level, message FROM system_event_logs").fetchall())
print(cur.execute("SELECT strategy_id, current_stage FROM execution_pipeline_steps").fetchall())
