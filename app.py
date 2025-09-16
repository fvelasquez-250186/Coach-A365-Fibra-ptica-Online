# -*- coding: utf-8 -*-
import os, re, csv, json, sqlite3, base64, mimetypes
from functools import wraps
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_from_directory, flash, make_response
)

# ========================= RUTAS BASE =========================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads" / "audios"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Zona horaria fija: Lima ---
APP_TZ = ZoneInfo("America/Lima")
def now_tz() -> datetime:
    return datetime.now(APP_TZ)

# Detecta DB existente para NO perder auditorías previas
_env_db = os.environ.get("DB_PATH", "").strip()
candidates = []
if _env_db:
    candidates.append(BASE_DIR / _env_db)
candidates += [
    BASE_DIR / "auditorias.db",
    BASE_DIR / "audits.db",
    BASE_DIR / "auditorias.sqlite",
    BASE_DIR / "data" / "auditorias.db",
]
_DB_PICK = next((p for p in candidates if str(p) and Path(p).exists()), None)
DB_PATH = _DB_PICK or (BASE_DIR / "auditorias.db")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-please-change")

# ========================= DB HELPERS =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        advisor TEXT NOT NULL,
        campaign TEXT,
        mobile_number TEXT,
        tipificacion TEXT,
        audio_url TEXT,
        detail_3_6_8 TEXT,
        advisor_confirmed INTEGER DEFAULT 0
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS advisor_emails (
        advisor TEXT PRIMARY KEY,
        email TEXT NOT NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS advisors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asesor TEXT NOT NULL,
        campaign TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

init_db()

# ========================= AUTH =========================
def require_login(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "dni" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

SUPERVISORES = {
    "70105421": "Castañeda Iraola Sergio",
    "71528928": "Chuquizuta  Torres  Christian Delmman",
    "44023703": "Alberca Presentacion Jesus Miguel",
    "72199613": "Munoz  Bejar Jhair Franco Rey",
}

# ========================= LECTORES =========================
def _read_csv_any(paths):
    for p in paths:
        if p and Path(p).exists():
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    with open(p, "r", encoding=enc, newline="") as f:
                        reader = csv.DictReader(f)
                        out = []
                        for r in reader:
                            nr = { (k or "").strip().lower(): (v or "").strip() for k, v in (r or {}).items() }
                            out.append(nr)
                        if out:
                            return out
                except Exception:
                    continue
    return None

def _read_excel_any(paths, sheet=None):
    try:
        import pandas as pd
    except Exception:
        return None
    for p in paths:
        if p and Path(p).exists():
            try:
                import pandas as pd
                df = pd.read_excel(p, sheet_name=sheet) if sheet else pd.read_excel(p)
                df.columns = [str(c).strip().lower() for c in df.columns]
                df = df.dropna(how="all")
                for c in df.columns:
                    df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) else "")
                return df.to_dict(orient="records")
            except Exception:
                continue
    return None

def _read_json_if_exists(*paths):
    for p in paths:
        if p and Path(p).exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def _load_advisors_any():
    items = _read_excel_any([
        BASE_DIR / "dotación_asesores.xlsx",
        BASE_DIR / "dotacion_asesores.xlsx",
        BASE_DIR / "dotacion" / "dotación_asesores.xlsx",
        BASE_DIR / "dotacion" / "dotacion_asesores.xlsx",
        BASE_DIR / "data" / "advisors.xlsx",
    ], sheet="asesores")
    if not items:
        items = _read_csv_any([BASE_DIR / "data" / "advisors.csv", BASE_DIR / "advisors.csv"])
    if not items:
        js = _read_json_if_exists(BASE_DIR / "data" / "advisors.json", BASE_DIR / "advisors.json")
        if isinstance(js, list):
            items = js
    if not items:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT asesor, campaign FROM advisors ORDER BY asesor")
        rows = cur.fetchall()
        conn.close()
        items = [{"asesor": r["asesor"], "campaign": r["campaign"]} for r in rows]

    norm = []
    for r in (items or []):
        asesor = (r.get("asesor") or r.get("advisor") or r.get("nombre") or "").strip()
        campaign = (r.get("campaign") or r.get("campaña") or r.get("campana") or "").strip()
        if not asesor or not campaign:
            continue
        norm.append({"advisor": asesor, "campaign": campaign})

    sup_names = set(SUPERVISORES.values())
    norm = [x for x in norm if x["advisor"] not in sup_names]

    seen, out = set(), []
    for it in norm:
        k = (it["advisor"], it["campaign"])
        if k in seen: continue
        seen.add(k); out.append(it)
    return out

def _norm_name(s:str)->str:
    return " ".join((s or "").strip().lower().split())

def _find_campaign_for_advisor(advisor_name:str)->str:
    key = _norm_name(advisor_name)
    for it in _load_advisors_any():
        if _norm_name(it["advisor"]) == key:
            return it["campaign"]
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT campaign FROM advisors WHERE lower(trim(asesor)) = ? LIMIT 1", (key,))
    row = cur.fetchone()
    conn.close()
    return row["campaign"] if row else ""

# ========================= VISTAS =========================
@app.route("/", methods=["GET"])
def home():
    if "dni" in session:
        return redirect(url_for("supervisor_dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        dni = (request.form.get("dni") or "").strip()
        pwd = (request.form.get("password") or "").strip()
        name = SUPERVISORES.get(dni)
        if not name:
            flash("DNI no reconocido", "warning")
            return render_template("login.html")
        if pwd != dni:
            flash("Contraseña incorrecta. Recuerda: es igual a tu DNI.", "danger")
            return render_template("login.html")
        session["dni"] = dni
        session["sup_name"] = name
        session["allowed_campaigns"] = {
            "70105421": ["C2C HOGAR PRIVADO", "WEB HOGAR PRIVADO"],
            "72199613": ["C2C HOGAR PRIVADO", "WEB HOGAR PRIVADO"],
            "71528928": ["FONOCOMPRAS HOGAR"],
            "44023703": ["FONOCOMPRAS HOGAR"],
        }.get(dni, [])
        flash(f"¡Bienvenido, {name}!", "success")
        return redirect(url_for("supervisor_dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/supervisor_dashboard")
@require_login
def supervisor_dashboard():
    return render_template("supervisor_dashboard.html")

@app.route("/advisor_dashboard")
@require_login
def advisor_dashboard():
    advisors = _load_advisors_any()
    return render_template("asesor_dashboard.html", advisors=advisors)

# ========================= UPLOAD =========================
@app.route("/upload", methods=["POST"])
@require_login
def upload():
    f = request.files.get("audio")
    advisor = (request.form.get("advisor") or "").strip()
    mobile = (request.form.get("mobile") or "").strip()
    tipificacion_grouped = (request.form.get("tipificacion") or "").strip()

    if not f or f.filename == "":
        return "Archivo no recibido", 400
    if not advisor:
        return "Asesor es requerido", 400

    ts = now_tz().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\.-]+", "_", f.filename)
    filename = f"{ts}_{safe_name}"
    path = UPLOAD_DIR / filename
    f.save(path)
    audio_url = url_for("serve_upload", filename=filename)

    transcript = ""  # aquí iría la transcripción
    campaign = _find_campaign_for_advisor(advisor)
    detail_3_6_8 = "Coaching generado (demo)"

    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO audits (created_at, advisor, campaign, mobile_number, tipificacion, audio_url, detail_3_6_8, advisor_confirmed)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
    """, (
        now_tz().strftime("%Y-%m-%d %H:%M:%S"),
        advisor, campaign, mobile, tipificacion_grouped,
        audio_url, detail_3_6_8
    ))
    conn.commit(); conn.close()

    flash("Audio subido y analizado.", "success")
    return ("", 204)

@app.route("/uploads/<path:filename>")
@require_login
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

# ========================= EXPORT =========================
@app.route("/export_supervisor")
@require_login
def export_supervisor():
    try:
        import pandas as pd
    except Exception:
        return "Falta dependencia pandas", 500

    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT created_at, advisor, campaign, mobile_number, tipificacion,
               detail_3_6_8 AS feedback
        FROM audits ORDER BY id DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]; conn.close()

    now = now_tz()
    ym = f"{now.year}-{str(now.month).zfill(2)}"
    rows = [r for r in rows if (r.get("created_at") or "").startswith(ym)]
    if not rows:
        rows = [{"created_at":"", "advisor":"", "campaign":"", "mobile_number":"", "tipificacion":"", "feedback":""}]

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = BASE_DIR / f"export_supervisor_{ym}.xlsx"
    df.to_excel(out_path, index=False)
    with open(out_path, "rb") as f:
        data = f.read()
    resp = make_response(data)
    resp.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    resp.headers["Content-Disposition"] = f'attachment; filename="{out_path.name}"'
    return resp

# ========================= RUN =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
