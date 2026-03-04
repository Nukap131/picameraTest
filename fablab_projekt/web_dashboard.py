from flask import Flask, render_template, jsonify, send_file
import sqlite3
import pandas as pd
from datetime import datetime
import io

app = Flask(__name__)
DB_FILE = "fablab_people.db"

@app.route("/")
def dashboard():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Total ind
    cursor.execute("SELECT COUNT(*) FROM people WHERE direction='←'")
    total_ind = cursor.fetchone()[0]
    
    # Seneste 20 events
    cursor.execute("SELECT timestamp, track_id, direction, total FROM people ORDER BY id DESC LIMIT 20")
    events = cursor.fetchall()
    
    # Dagens total
    today = datetime.now().strftime('%d-%m-%y')
    cursor.execute("SELECT COUNT(*) FROM people WHERE direction='←' AND timestamp LIKE ?", (f"{today}%",))
    today_total = cursor.fetchone()[0]
    
    conn.close()
    return render_template("dashboard.html", total=total_ind, today=today_total, events=events)

@app.route("/api")
def api():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.execute("SELECT * FROM people ORDER BY id DESC LIMIT 100")
    data = [{"id": row[0], "time": row[1], "track": row[2], "dir": row[3], "total": row[4]} for row in cursor.fetchall()]
    conn.close()
    return jsonify(data)

# ← NY: DOWNLOAD KNAP!
@app.route("/download/csv")
def download_csv():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM people ORDER BY id", conn)
    conn.close()
    
    # CSV i hukommelsen (ikke fil på disk)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"fablab_people_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    )

@app.route("/download/json")
def download_json():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM people ORDER BY id", conn)
    conn.close()
    
    return send_file(
        io.BytesIO(df.to_json(orient='records', indent=2).encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name=f"fablab_people_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
