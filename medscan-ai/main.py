import base64
import hashlib
import io
import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)
CATEGORIES = weights.meta.get("categories", [])

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

DB_PATH = Path(__file__).with_name("medscan.db")
SESSION_TTL_SECONDS = 60 * 60 * 12
CRITICAL_CONFIDENCE_THRESHOLD = 0.8


class TopPrediction(BaseModel):
    class_id: int
    label: str
    confidence: float


class AnalyzeResponse(BaseModel):
    analysis_id: str
    prediction_class: int
    prediction_label: str
    confidence: float
    risk_label: str
    risk_score: float
    recommendation: str
    triage_level: str
    requires_urgent_review: bool
    alert_message: str
    queue_status: str
    patient_id: str | None = None
    exam_type: str | None = None
    clinical_note: str | None = None
    top_predictions: list[TopPrediction]
    heatmap_base64_png: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


RISK_BUCKETS = [
    (0.0, 0.2, "Normal", "Suivi standard recommande."),
    (0.2, 0.5, "Suspicion faible", "Controle clinique et surveillance suggeres."),
    (0.5, 0.8, "Suspicion moderee", "Avis specialise conseille."),
    (0.8, 1.01, "Suspicion elevee", "Orientation urgente vers un specialiste."),
]


def _password_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                prediction_class INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_label TEXT NOT NULL,
                risk_score REAL NOT NULL,
                recommendation TEXT NOT NULL,
                patient_id TEXT,
                exam_type TEXT,
                clinical_note TEXT,
                top_predictions_json TEXT NOT NULL,
                image_bytes BLOB,
                heatmap_png_bytes BLOB,
                queue_status TEXT NOT NULL DEFAULT 'none',
                resolved_by TEXT,
                resolved_at REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY(username) REFERENCES users(username)
            )
            """
        )
        conn.commit()

    _migrate_analysis_assets()
    _seed_users()


def _migrate_analysis_assets() -> None:
    with _get_connection() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(analyses)").fetchall()
        }
        if "image_bytes" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN image_bytes BLOB")
        if "heatmap_png_bytes" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN heatmap_png_bytes BLOB")
        if "queue_status" not in columns:
            conn.execute(
                "ALTER TABLE analyses ADD COLUMN queue_status TEXT NOT NULL DEFAULT 'none'"
            )
        if "resolved_by" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN resolved_by TEXT")
        if "resolved_at" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN resolved_at REAL")
        conn.commit()


def _seed_users() -> None:
    seed = [
        ("admin", _password_hash("admin123"), "admin"),
        ("medecin", _password_hash("med123"), "medecin"),
    ]
    with _get_connection() as conn:
        for username, password_hash, role in seed:
            conn.execute(
                """
                INSERT OR IGNORE INTO users (username, password_hash, role, active)
                VALUES (?, ?, ?, 1)
                """,
                (username, password_hash, role),
            )
        conn.commit()


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    return parts[1]


def _get_current_user(authorization: str | None = Header(default=None)) -> dict:
    token = _extract_token(authorization)
    now = time.time()
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT s.token, s.username, s.expires_at, u.role, u.active
            FROM sessions s
            JOIN users u ON u.username = s.username
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Invalid session")
        if int(row["active"]) != 1:
            raise HTTPException(status_code=403, detail="Inactive user")
        if float(row["expires_at"]) < now:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            raise HTTPException(status_code=401, detail="Session expired")
        return {"username": row["username"], "role": row["role"], "token": row["token"]}


def _require_admin(user: dict = Depends(_get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _tensor_to_heatmap(cam: np.ndarray, orig_img: Image.Image) -> tuple[str, bytes]:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig = np.array(orig_img.resize((cam.shape[1], cam.shape[0])))
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        return "", b""
    png_bytes = buf.tobytes()
    return base64.b64encode(png_bytes).decode("utf-8"), png_bytes


def _grad_cam(model_: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = model_.layer4.register_forward_hook(forward_hook)
    handle_bwd = model_.layer4.register_full_backward_hook(backward_hook)

    logits = model_(input_tensor)
    class_idx = int(torch.argmax(logits, dim=1).item())
    model_.zero_grad()
    logits[0, class_idx].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0]
    grad = gradients[0]
    weights_ = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights_ * act, dim=1)
    cam = torch.relu(cam)[0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _risk_from_confidence(conf: float) -> tuple[str, float, str]:
    risk_score = float(conf)
    for lo, hi, label, reco in RISK_BUCKETS:
        if lo <= risk_score < hi:
            return label, risk_score, reco
    return "Indetermine", risk_score, "Nouvelle acquisition d'image recommandee."


def _triage_from_confidence(confidence: float) -> tuple[str, bool, str]:
    if confidence >= CRITICAL_CONFIDENCE_THRESHOLD:
        return (
            "critical",
            True,
            f"Probabilite d'anomalie: {round(confidence * 100)}% - Consultation urgente recommandee.",
        )
    if confidence >= 0.5:
        return (
            "high",
            False,
            f"Probabilite d'anomalie: {round(confidence * 100)}% - Evaluation specialisee conseillee.",
        )
    return (
        "routine",
        False,
        f"Probabilite d'anomalie: {round(confidence * 100)}% - Suivi standard.",
    )


def _class_label(class_id: int) -> str:
    if 0 <= class_id < len(CATEGORIES):
        return str(CATEGORIES[class_id])
    return f"class_{class_id}"


def _to_top_predictions(probs: torch.Tensor, k: int = 3) -> list[TopPrediction]:
    top_conf, top_idx = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    result = []
    for conf, idx in zip(top_conf[0].tolist(), top_idx[0].tolist()):
        cid = int(idx)
        result.append(
            TopPrediction(
                class_id=cid,
                label=_class_label(cid),
                confidence=float(conf),
            )
        )
    return result


def _save_analysis(
    row: AnalyzeResponse, image_bytes: bytes, heatmap_png_bytes: bytes | None
) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                analysis_id, timestamp, prediction_class, prediction_label,
                confidence, risk_label, risk_score, recommendation,
                patient_id, exam_type, clinical_note, top_predictions_json,
                image_bytes, heatmap_png_bytes, queue_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.analysis_id,
                time.time(),
                row.prediction_class,
                row.prediction_label,
                row.confidence,
                row.risk_label,
                row.risk_score,
                row.recommendation,
                row.patient_id,
                row.exam_type,
                row.clinical_note,
                json.dumps([p.model_dump() for p in row.top_predictions]),
                sqlite3.Binary(image_bytes),
                sqlite3.Binary(heatmap_png_bytes) if heatmap_png_bytes else None,
                row.queue_status,
            ),
        )
        conn.commit()


def _history_row_to_json(row: sqlite3.Row) -> dict:
    triage_level, requires_urgent_review, alert_message = _triage_from_confidence(
        float(row["confidence"])
    )
    return {
        "analysis_id": row["analysis_id"],
        "timestamp": row["timestamp"],
        "prediction_class": row["prediction_class"],
        "prediction_label": row["prediction_label"],
        "confidence": row["confidence"],
        "risk_label": row["risk_label"],
        "risk_score": row["risk_score"],
        "recommendation": row["recommendation"],
        "triage_level": triage_level,
        "requires_urgent_review": requires_urgent_review,
        "alert_message": alert_message,
        "queue_status": row["queue_status"] if "queue_status" in row.keys() else "none",
        "resolved_by": row["resolved_by"] if "resolved_by" in row.keys() else None,
        "resolved_at": row["resolved_at"] if "resolved_at" in row.keys() else None,
        "patient_id": row["patient_id"],
        "exam_type": row["exam_type"],
        "clinical_note": row["clinical_note"],
        "top_predictions": json.loads(row["top_predictions_json"]),
    }


def _build_report_pdf(row: sqlite3.Row) -> bytes:
    triage_level, urgent, alert_message = _triage_from_confidence(float(row["confidence"]))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, y, "MedScan AI - Rapport d'analyse")
    y -= 28

    pdf.setFont("Helvetica", 11)
    lines = [
        f"ID analyse: {row['analysis_id']}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row['timestamp']))}",
        f"Patient ID: {row['patient_id'] or 'N/A'}",
        f"Type examen: {row['exam_type'] or 'N/A'}",
        f"Note clinique: {row['clinical_note'] or 'N/A'}",
        f"Prediction: {row['prediction_label']} (classe {row['prediction_class']})",
        f"Confiance: {round(float(row['confidence']) * 100)}%",
        f"Niveau risque: {row['risk_label']}",
        f"Triage: {triage_level} ({'urgent' if urgent else 'non urgent'})",
        f"Alerte: {alert_message}",
        f"Recommandation: {row['recommendation']}",
    ]
    for line in lines:
        pdf.drawString(40, y, line[:110])
        y -= 16

    image_bytes = row["image_bytes"]
    heatmap_bytes = row["heatmap_png_bytes"]
    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.thumbnail((250, 250))
            pdf.drawImage(ImageReader(img), 40, 130, width=220, height=220, preserveAspectRatio=True)
            pdf.drawString(40, 112, "Image originale")
        except Exception:
            pass

    if heatmap_bytes:
        try:
            hm = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")
            hm.thumbnail((250, 250))
            pdf.drawImage(ImageReader(hm), 300, 130, width=220, height=220, preserveAspectRatio=True)
            pdf.drawString(300, 112, "Heatmap")
        except Exception:
            pass

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@app.get("/health")
async def health():
    return {"status": "ok", "db_path": str(DB_PATH)}


@app.post("/auth/login")
async def login(payload: LoginRequest):
    with _get_connection() as conn:
        user = conn.execute(
            """
            SELECT username, password_hash, role, active
            FROM users
            WHERE username = ?
            """,
            (payload.username.strip(),),
        ).fetchone()
        if user is None or user["password_hash"] != _password_hash(payload.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if int(user["active"]) != 1:
            raise HTTPException(status_code=403, detail="User disabled")

        token = secrets.token_urlsafe(32)
        expires_at = time.time() + SESSION_TTL_SECONDS
        conn.execute(
            "INSERT INTO sessions (token, username, expires_at) VALUES (?, ?, ?)",
            (token, user["username"], expires_at),
        )
        conn.commit()
        return {
            "access_token": token,
            "token_type": "bearer",
            "role": user["role"],
            "username": user["username"],
            "expires_at": expires_at,
        }


@app.post("/auth/logout")
async def logout(user: dict = Depends(_get_current_user)):
    with _get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (user["token"],))
        conn.commit()
    return {"ok": True}


@app.get("/auth/me")
async def me(user: dict = Depends(_get_current_user)):
    return {"username": user["username"], "role": user["role"]}


@app.get("/admin/users")
async def list_users(_: dict = Depends(_require_admin)):
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT username, role, active FROM users ORDER BY username ASC"
        ).fetchall()
    return [
        {"username": row["username"], "role": row["role"], "active": int(row["active"])}
        for row in rows
    ]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    heatmap: bool = False,
    patient_id: str | None = Form(default=None),
    exam_type: str | None = Form(default=None),
    clinical_note: str | None = Form(default=None),
    _: dict = Depends(_get_current_user),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    prediction_class = int(predicted.item())
    confidence = float(conf.item())
    prediction_label = _class_label(prediction_class)
    top_predictions = _to_top_predictions(probs)
    risk_label, risk_score, recommendation = _risk_from_confidence(confidence)
    triage_level, requires_urgent_review, alert_message = _triage_from_confidence(
        confidence
    )
    queue_status = "open" if triage_level == "critical" else "none"

    response = AnalyzeResponse(
        analysis_id=str(uuid.uuid4()),
        prediction_class=prediction_class,
        prediction_label=prediction_label,
        confidence=confidence,
        risk_label=risk_label,
        risk_score=risk_score,
        recommendation=recommendation,
        triage_level=triage_level,
        requires_urgent_review=requires_urgent_review,
        alert_message=alert_message,
        queue_status=queue_status,
        patient_id=patient_id,
        exam_type=exam_type,
        clinical_note=clinical_note,
        top_predictions=top_predictions,
    )

    heatmap_png_bytes: bytes | None = None
    if heatmap:
        cam = _grad_cam(model, image_tensor)
        heatmap_base64, heatmap_png_bytes = _tensor_to_heatmap(cam, image)
        response.heatmap_base64_png = heatmap_base64 or None

    _save_analysis(response, image_bytes=image_bytes, heatmap_png_bytes=heatmap_png_bytes)
    return response


@app.get("/history")
async def get_history(
    limit: int = 20,
    patient_id: str | None = None,
    risk_label: str | None = None,
    critical_only: bool = False,
    _: dict = Depends(_get_current_user),
):
    limit = max(1, min(limit, 200))
    query = """
        SELECT analysis_id, timestamp, prediction_class, prediction_label,
               confidence, risk_label, risk_score, recommendation,
               patient_id, exam_type, clinical_note, top_predictions_json,
               queue_status, resolved_by, resolved_at
        FROM analyses
        WHERE 1=1
    """
    params: list = []
    if patient_id:
        query += " AND patient_id = ?"
        params.append(patient_id)
    if risk_label:
        query += " AND risk_label = ?"
        params.append(risk_label)
    if critical_only:
        query += " AND confidence >= ?"
        params.append(CRITICAL_CONFIDENCE_THRESHOLD)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _get_connection() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_history_row_to_json(row) for row in rows]


@app.get("/stats")
async def get_stats(_: dict = Depends(_get_current_user)):
    with _get_connection() as conn:
        total_row = conn.execute(
            "SELECT COUNT(*) AS total, AVG(confidence) AS avg_confidence FROM analyses"
        ).fetchone()
        dist_rows = conn.execute(
            "SELECT risk_label, COUNT(*) AS count FROM analyses GROUP BY risk_label"
        ).fetchall()
        critical_row = conn.execute(
            "SELECT COUNT(*) AS c FROM analyses WHERE confidence >= ?",
            (CRITICAL_CONFIDENCE_THRESHOLD,),
        ).fetchone()
        open_critical_row = conn.execute(
            "SELECT COUNT(*) AS c FROM analyses WHERE queue_status = 'open'"
        ).fetchone()
        last_high = conn.execute(
            """
            SELECT analysis_id, timestamp, patient_id, exam_type, confidence, queue_status
            FROM analyses
            WHERE confidence >= ? AND queue_status = 'open'
            ORDER BY confidence DESC, timestamp ASC
            LIMIT 5
            """,
            (CRITICAL_CONFIDENCE_THRESHOLD,),
        ).fetchall()

    total = int(total_row["total"] or 0)
    avg_conf = float(total_row["avg_confidence"] or 0.0)
    dist = {row["risk_label"]: int(row["count"]) for row in dist_rows}
    critical_count = int(critical_row["c"] or 0)
    open_critical_count = int(open_critical_row["c"] or 0)
    recent_alerts = [
        {
            "analysis_id": row["analysis_id"],
            "timestamp": row["timestamp"],
            "patient_id": row["patient_id"],
            "exam_type": row["exam_type"],
            "confidence": row["confidence"],
            "queue_status": row["queue_status"],
            "alert_message": _triage_from_confidence(float(row["confidence"]))[2],
        }
        for row in last_high
    ]
    return {
        "total": total,
        "risk_distribution": dist,
        "avg_confidence": avg_conf,
        "critical_count": critical_count,
        "open_critical_count": open_critical_count,
        "critical_rate": (critical_count / total) if total else 0.0,
        "recent_alerts": recent_alerts,
        "critical_threshold": CRITICAL_CONFIDENCE_THRESHOLD,
    }


@app.get("/report/{analysis_id}")
async def download_report(analysis_id: str, _: dict = Depends(_get_current_user)):
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT analysis_id, timestamp, prediction_class, prediction_label,
                   confidence, risk_label, risk_score, recommendation,
                   patient_id, exam_type, clinical_note, image_bytes, heatmap_png_bytes
            FROM analyses
            WHERE analysis_id = ?
            """,
            (analysis_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    pdf_bytes = _build_report_pdf(row)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="medscan_report_{analysis_id}.pdf"'
        },
    )


@app.get("/queue/critical")
async def get_critical_queue(limit: int = 30, _: dict = Depends(_get_current_user)):
    limit = max(1, min(limit, 200))
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT analysis_id, timestamp, prediction_class, prediction_label,
                   confidence, risk_label, risk_score, recommendation,
                   patient_id, exam_type, clinical_note, top_predictions_json,
                   queue_status, resolved_by, resolved_at
            FROM analyses
            WHERE queue_status = 'open' AND confidence >= ?
            ORDER BY confidence DESC, timestamp ASC
            LIMIT ?
            """,
            (CRITICAL_CONFIDENCE_THRESHOLD, limit),
        ).fetchall()

    return {
        "updated_at": time.time(),
        "count": len(rows),
        "items": [_history_row_to_json(row) for row in rows],
    }


@app.post("/queue/critical/{analysis_id}/resolve")
async def resolve_critical_case(
    analysis_id: str, user: dict = Depends(_get_current_user)
):
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT analysis_id, queue_status FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if row["queue_status"] == "resolved":
            return {"ok": True, "already_resolved": True}

        conn.execute(
            """
            UPDATE analyses
            SET queue_status = 'resolved', resolved_by = ?, resolved_at = ?
            WHERE analysis_id = ?
            """,
            (user["username"], time.time(), analysis_id),
        )
        conn.commit()
    return {"ok": True, "already_resolved": False}


_init_db()
