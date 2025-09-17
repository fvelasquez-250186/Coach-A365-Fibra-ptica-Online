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
from google.oauth2 import service_account
from googleapiclient.discovery import build

# === Google Drive (Service Account) ===
SERVICE_ACCOUNT_FILE = str(Path(__file__).resolve().parent / "credentials" / "coach-a365-drive.json")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)
def drive_upload_audio(file_storage):
    # imports locales para no tocar más imports arriba
    from googleapiclient.http import MediaIoBaseUpload
    from werkzeug.utils import secure_filename
    import io

    service = get_drive_service()

    # Nombre seguro y único
    orig = secure_filename(file_storage.filename or "audio.mp3")
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f"{ts}_{orig}"

    # Subir contenido
    media = MediaIoBaseUpload(
        file_storage.stream,
        mimetype=(file_storage.mimetype or "audio/mpeg"),
        resumable=False
    )
    meta = {"name": name}
    created = service.files().create(
        body=meta, media_body=media, fields="id,webViewLink,webContentLink"
    ).execute()

    file_id = created["id"]

    # Hacer accesible con enlace
    service.permissions().create(
        fileId=file_id, body={"role": "reader", "type": "anyone"}
    ).execute()

    # URL directa de descarga (si no, la de vista)
    url = created.get("webContentLink") or created.get("webViewLink")
    return url, file_id

import boto3
from botocore.config import Config  # 👈 nuevo import

# --- Cliente S3 (Backblaze B2) ---
B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "").strip()
B2_KEY_ID = os.environ.get("B2_KEY_ID", "").strip()
B2_APP_KEY = os.environ.get("B2_APP_KEY", "").strip()
B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "").strip()

s3_client = boto3.client(
    "s3",
    endpoint_url=f"https://{B2_ENDPOINT}",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY,
    region_name="us-west-004",
    config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
)

# ---- Hora local Lima (Perú) ----
APP_TZ = ZoneInfo("America/Lima")
def now_pe():
    return datetime.now(APP_TZ)

# ========================= RUTAS BASE =========================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads" / "audios"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

# ========================= B2 CLIENT =========================
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY")
B2_BUCKET = os.environ.get("B2_BUCKET_NAME")
B2_ENDPOINT = os.environ.get("B2_ENDPOINT")

s3_client = boto3.client(
    "s3",
    endpoint_url=f"https://{B2_ENDPOINT}",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY,
)

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
        advisor_confirmed INTEGER DEFAULT 0,
        uploaded_by TEXT
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

def _migrate_add_uploaded_by():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(audits)")
    cols = [r["name"] for r in cur.fetchall()]
    if "uploaded_by" not in cols:
        cur.execute("ALTER TABLE audits ADD COLUMN uploaded_by TEXT")
        conn.commit()
    conn.close()

init_db()
_migrate_add_uploaded_by()

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

# ========================= LECTORES (advisors / tipificaciones) =========================
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
        import pandas as pd  # noqa: F401
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
        campaign = (r.get("campaign") or r.get("campaña") or r.get("campana") or r.get("campana ") or "").strip()
        if not asesor or not campaign:
            continue
        norm.append({"advisor": asesor, "campaign": campaign})

    # quita nombres de supervisores si aparecieran
    sup_names = set(SUPERVISORES.values())
    norm = [x for x in norm if x["advisor"] not in sup_names]

    # únicos
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

def _load_tipificaciones_any():
    """
    Excel/CSV con:
    - Columna A: 'tipificacion'  -> display (desplegable)
    - Columna B: 'tipi agrupada' -> grouped (se guarda/exporta)
    """
    excel_rows = _read_excel_any([
        BASE_DIR / "tipificacion" / "tipificaciones.xlsx",
        BASE_DIR / "tipificacion" / "tipificacion.xlsx",
        BASE_DIR / "tipificaciones.xlsx",
        BASE_DIR / "tipificacion.xlsx",
        BASE_DIR / "data" / "tipificaciones.xlsx",
    ])
    if excel_rows:
        norm = []
        for r in excel_rows:
            display = (r.get("tipificacion") or r.get("tipificación") or r.get("display") or "").strip()
            grouped = (r.get("tipi agrupada") or r.get("tipi_agrupada") or r.get("grouped") or "").strip()
            if display:
                norm.append({"display": display, "grouped": grouped})
        if norm: return norm

    js = _read_json_if_exists(BASE_DIR / "data" / "tipificaciones.json", BASE_DIR / "tipificaciones.json")
    if isinstance(js, list) and js:
        norm = []
        for t in js:
            display = (t.get("display") or t.get("nombre") or t.get("name") or "").strip()
            grouped = (t.get("grouped") or t.get("grupo") or t.get("group") or "").strip()
            if display:
                norm.append({"display": display, "grouped": grouped})
        if norm: return norm

    csv_rows = _read_csv_any([BASE_DIR / "data" / "tipificaciones.csv", BASE_DIR / "tipificaciones.csv"])
    if csv_rows:
        norm = []
        for r in csv_rows:
            display = (r.get("display") or r.get("nombre") or r.get("name") or "").strip()
            grouped = (r.get("grouped") or r.get("grupo") or r.get("group") or "").strip()
            if display:
                norm.append({"display": display, "grouped": grouped})
        if norm: return norm

    # Fallback mínimo visual si no hay fuentes
    return [
        {"display": "Contactado-No Televendible-Adulto mayor", "grouped": "SERVICIOS MÓVILES"},
        {"display": "No Interesa", "grouped": "SERVICIOS MÓVILES"},
        {"display": "Agendado para volver a llamar-Cliente está ocupado", "grouped": "SERVICIOS MÓVILES"},
        {"display": "Ajena de Gestión", "grouped": "SERVICIOS MÓVILES"},
        {"display": "Sin Factibilidad", "grouped": "SERVICIOS MÓVILES"},
    ]

# ========================= LOGOS =========================
def _logo_url(name: str):
    for ext in ("svg", "png", "jpg", "jpeg"):
        p = BASE_DIR / "static" / "brand" / f"{name}.{ext}"
        if p.exists():
            return url_for("static", filename=f"brand/{name}.{ext}")
    if name.lower() == "entel":
        return "https://upload.wikimedia.org/wikipedia/commons/2/2c/Entel_logo.svg"
    if name.lower() == "impulsa":
        return "https://via.placeholder.com/120x32?text=Impulsa365"
    return "https://via.placeholder.com/120x32?text=Logo"

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
            return render_template("login.html", entel_logo=_logo_url("entel"), impulsa_logo=_logo_url("impulsa"))
        if pwd != dni:
            flash("Contraseña incorrecta. Recuerda: es igual a tu DNI.", "danger")
            return render_template("login.html", entel_logo=_logo_url("entel"), impulsa_logo=_logo_url("impulsa"))
        session["dni"] = dni
        session["sup_name"] = name
        # campañas permitidas para filtros
        session["allowed_campaigns"] = {
            "70105421": ["C2C HOGAR PRIVADO", "WEB HOGAR PRIVADO"],
            "72199613": ["C2C HOGAR PRIVADO", "WEB HOGAR PRIVADO"],
            "71528928": ["FONOCOMPRAS HOGAR", "FONOCOMPRAS"],  # Chuquizuta
            "44023703": ["FONOCOMPRAS HOGAR", "FONOCOMPRAS"],  # Alberca
        }.get(dni, [])
        flash(f"¡Bienvenido, {name}!", "success")
        return redirect(url_for("supervisor_dashboard"))
    return render_template("login.html", entel_logo=_logo_url("entel"), impulsa_logo=_logo_url("impulsa"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/supervisor")
@app.route("/supervisor_dashboard")
@require_login
def supervisor_dashboard():
    return render_template("supervisor_dashboard.html",
                           entel_logo=_logo_url("entel"), impulsa_logo=_logo_url("impulsa"))

@app.route("/advisor")
@app.route("/advisor_dashboard")
@require_login
def advisor_dashboard():
    advisors = _load_advisors_any()
    return render_template("asesor_dashboard.html",
                           advisors=advisors,
                           entel_logo=_logo_url("entel"), impulsa_logo=_logo_url("impulsa"))

# ========================= APIs =========================
def _filter_allowed_campaigns(rows):
    allowed = [ (c or "").strip().upper() for c in (session.get("allowed_campaigns") or []) ]
    if not allowed:
        return rows
    out = []
    for r in rows:
        camp = ((r.get("campaign") or "").strip().upper())
        # acepta igualdad exacta, prefijo o subcadena
        if any(camp == a or camp.startswith(a) or a.startswith(camp) or (a in camp) or (camp in a) for a in allowed):
            out.append(r)
    return out

@app.route("/api_advisors")
@require_login
def api_advisors():
    items = _load_advisors_any()
    items = _filter_allowed_campaigns(items)
    items.sort(key=lambda x: (x["advisor"].upper(), x["campaign"].upper()))
    return jsonify(items)

@app.route("/api/advisors")
@require_login
def api_advisors_compat():
    return api_advisors()

@app.route("/api_tipificaciones")
@require_login
def api_tipificaciones():
    return jsonify(_load_tipificaciones_any())

@app.route("/api_audits")
@require_login
def api_audits():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, created_at, advisor, campaign, mobile_number, tipificacion,
               audio_url, detail_3_6_8, advisor_confirmed, uploaded_by
        FROM audits
        ORDER BY id DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    rows = _filter_allowed_campaigns(rows)
    for r in rows:
        r["advisor_confirmed"] = bool(r.get("advisor_confirmed"))
    return jsonify(rows)

# ========================= GEMINI (transcribir + generar 3/6 A365) =========================
try:
    import google.generativeai as genai
    _GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    if _GEMINI_KEY:
        genai.configure(api_key=_GEMINI_KEY)
except Exception:
    genai = None
    _GEMINI_KEY = ""

def _mime_for(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "audio/mpeg"

def transcribe_audio_gemini(audio_path: Path) -> str:
    """Transcribe con Gemini. Si no hay clave/config, lanzamos error para no ocultar el problema."""
    if not (genai and _GEMINI_KEY):
        raise RuntimeError("Gemini no está configurado. Revisa GEMINI_API_KEY.")

    b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    mime = _mime_for(audio_path)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

    resp = model.generate_content([
        {"text": "Transcribe en español (latino) con buena puntuación. Devuelve SOLO el texto, sin etiquetas."},
        {"inline_data": {"mime_type": mime, "data": b64}}
    ])

    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini devolvió transcripción vacía.")
    return text

COACHING_PROMPT = """
Eres el COACH A365 para Fibra Óptica Entel Chile.
Tarea: a partir de la TRANSCRIPCIÓN (abajo), la TIPIFICACIÓN («{{TIPIFICACION}}») y la CAMPAÑA («{{CAMPANIA}}»),
devuelve SOLO el contenido dentro de estas secciones ya escritas.
No cambies los títulos, no inventes nuevas secciones, no elimines nada.
No inventes datos que no estén en la transcripción. Usa frases breves, concretas y en viñetas cuando corresponda.
Solo completa debajo de cada encabezado.

**3) Contexto y Resultado Detallado:**
(Escribe 4–6 oraciones con el resumen de la llamada y el resultado concreto. No inventes datos.)

**4) Análisis Táctico Detallado de la Interacción (Foco en Factibilidad y Conversión):**
(Completa cada subapartado con 1–3 líneas específicas tomadas de la transcripción.)

- Punto Crítico: Consulta de Factibilidad:
  o Momento y Justificación:
  o Manejo de la Reticencia:
  o Observaciones del Audio (Tono/Ritmo):
    • Tono de Voz del Agente:
    • Tono de Voz del Cliente:

- Sondeo para Necesidad de Fibra:
  o

- Presentación de Valor de Fibra:
  o

- Manejo de Objeciones:
  o

- Cierre y Programación de Instalación:
  o

- Oportunidades Perdidas:
  o

**6) Recomendaciones Estratégicas y Ejemplos de Guion (Acción Inmediata):**
(Escribe guiones y ejemplos listos para usar. Incluye subtítulos:
1. Guion de Pivote y Consulta de Factibilidad Inmediata
2. Estrategia para Generar Confianza al Pedir Datos (Refuerzo)
3. Profundizar en el Sondeo con Preguntas Abiertas sobre “Dolores”
y completa cada uno con frases de ejemplo).

DATOS CONTEXTO:
- Tipificación: {{TIPIFICACION}}
- Campaña: {{CAMPANIA}}

TRANSCRIPCIÓN:
{{TRANSCRIPCION}}
"""

def _cleanup_to_a365(text: str) -> str:
    t = (text or "").strip()
    # Normaliza detalles mínimos de formato sin cambiar contenido
    t = re.sub(r'\s*•\s*', r'\n• ', t)
    t = re.sub(r'\n{3,}', '\n\n', t).strip()
    return t

def generate_coaching_from_prompt(transcript: str, tipificacion: str, campaign: str) -> str:
    if not (genai and _GEMINI_KEY):
        raise RuntimeError("Gemini no está configurado. Revisa GEMINI_API_KEY.")

    trans = (transcript or "").strip()
    if not trans:
        raise RuntimeError("Transcripción vacía. No se puede generar coaching.")

    prompt = (
        COACHING_PROMPT
        .replace("{{TIPIFICACION}}", tipificacion or "No especificada")
        .replace("{{CAMPANIA}}", campaign or "No especificada")
        .replace("{{TRANSCRIPCION}}", trans)
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
    )
    resp = model.generate_content(prompt, safety_settings=None)

    # Extrae texto desde candidates/parts (sin fallback)
    raw = ""
    for c in (getattr(resp, "candidates", []) or []):
        content = getattr(c, "content", None)
        parts = (getattr(content, "parts", None) or [])
        txt = "".join((getattr(p, "text", "") or "") for p in parts).strip()
        if txt:
            raw = txt
            break

    if not raw:
        raise RuntimeError("Gemini respondió sin contenido utilizable (parts vacíos).")

    return raw

# ========================= SUBIDA / SERVICIOS =========================
@app.route("/upload", methods=["POST"])
@require_login
def upload():
    f = request.files.get("audio")
    advisor = (request.form.get("advisor") or "").strip()
    mobile = (request.form.get("mobile") or "").strip()
    # El <select> envía la agrupada (columna B) como valor para guardarse:
    tipificacion_grouped = (request.form.get("tipificacion") or "").strip()
    uploaded_by = session.get("dni") or ""

    if not f or f.filename == "":
        return "Archivo no recibido", 400
    if not advisor:
        return "Asesor es requerido", 400

    ts = now_pe().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\.-]+", "_", f.filename)
    filename = f"{ts}_{safe_name}"
    path = UPLOAD_DIR / filename
        # Subir a Backblaze B2
    s3_client.upload_file(str(path), B2_BUCKET, filename)
    audio_url = f"https://{B2_ENDPOINT}/{B2_BUCKET}/{filename}"

    # Transcribe + genera coaching 3/6 (sin fallback)
    transcript = transcribe_audio_gemini(path)
    campaign = _find_campaign_for_advisor(advisor)
    detail_3_6_8 = generate_coaching_from_prompt(transcript, tipificacion_grouped, campaign)

    # Inserta incluyendo uploaded_by
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO audits (created_at, advisor, campaign, mobile_number, tipificacion,
                            audio_url, detail_3_6_8, advisor_confirmed, uploaded_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
    """, (
        now_pe().strftime("%Y-%m-%d %H:%M:%S"),
        advisor, campaign, mobile, tipificacion_grouped,
        audio_url, detail_3_6_8, uploaded_by
    ))
    conn.commit(); conn.close()

    flash("Audio subido y analizado.", "success")
    return ("", 204)

# @app.route("/uploads/<path:filename>")
# @require_login
# def serve_upload(filename):
#     return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.route("/advisor/confirm/<int:audit_id>", methods=["POST"])
@require_login
def advisor_confirm(audit_id: int):
    conn = get_db(); cur = conn.cursor()
    cur.execute("UPDATE audits SET advisor_confirmed = 1 WHERE id = ?", (audit_id,))
    conn.commit(); conn.close()
    flash("Confirmación registrada.", "success")
    ref = request.headers.get("Referer") or url_for("supervisor_dashboard")
    return redirect(ref)

@app.route("/advisor_email_update", methods=["POST"])
@require_login
def advisor_email_update():
    advisor = (request.form.get("advisor") or "").strip()
    email = (request.form.get("email") or "").strip()
    if not advisor or not email:
        flash("Completa asesor y correo.", "warning")
        return redirect(url_for("advisor_dashboard"))
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO advisor_emails (advisor, email) VALUES (?,?)
        ON CONFLICT(advisor) DO UPDATE SET email=excluded.email
    """, (advisor, email))
    conn.commit(); conn.close()
    flash("Correo actualizado.", "success")
    return redirect(url_for("advisor_dashboard"))

@app.route("/export_supervisor")
@require_login
def export_supervisor():
    """Exporta: uploaded_by (1ra), created_at, advisor, campaign, mobile_number, tipificacion (agrupada), feedback (detail_3_6_8)."""
    try:
        import pandas as pd
    except Exception:
        return "Falta dependencia pandas (pip install pandas openpyxl)", 500

    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT uploaded_by, created_at, advisor, campaign, mobile_number, tipificacion,
               detail_3_6_8 AS feedback
        FROM audits ORDER BY id DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]; conn.close()

    # Filtra por mes actual
    now = now_pe(); ym = f"{now.year}-{str(now.month).zfill(2)}"
    rows = [r for r in rows if (r.get("created_at") or "").startswith(ym)]
    if not rows:
        rows = [{
            "uploaded_by":"", "created_at": "", "advisor": "", "campaign": "",
            "mobile_number": "", "tipificacion": "", "feedback": ""
        }]

    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "uploaded_by","created_at","advisor","campaign","mobile_number","tipificacion","feedback"
    ])
    out_path = BASE_DIR / f"export_supervisor_{ym}.xlsx"
    df.to_excel(out_path, index=False)
    with open(out_path, "rb") as f:
        data = f.read()
    resp = make_response(data)
    resp.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    resp.headers["Content-Disposition"] = f'attachment; filename="{out_path.name}"'
    return resp

@app.route("/resumen_semanal.txt")
@require_login
def resumen_semanal():
    path = BASE_DIR / "resumen_semanal.txt"
    if not path.exists():
        return "Aún no hay resumen semanal.", 404
    return send_from_directory(BASE_DIR, "resumen_semanal.txt", as_attachment=True)

# ========================= RUN =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
