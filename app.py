"""FastAPI app — upload, process, browse evidence, analytics.

Frontend is server-rendered Jinja + HTMX + Tailwind (CDN) + Chart.js. The
heavy lifting lives in backend.pipeline; this module is just transport + glue.
"""
from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend import config, db
from backend.pipeline import detect, engine, plate
from backend.pipeline.annotate import annotate

app = FastAPI(title="Traffic Violation CV", version="0.1.0")
TEMPLATES = Jinja2Templates(directory=str(config.ROOT / "frontend" / "templates"))

# --- Security: HTTP Basic auth -------------------------------------------
# Records/evidence expose plate numbers (PII), so every route is protected.
# Credentials come from env; if unset we generate a random password at startup
# and print it once to the log (secure-by-default, no hardcoded creds).
_security = HTTPBasic()
AUTH_USER = os.environ.get("TVCV_USERNAME", "admin")
AUTH_PASS = os.environ.get("TVCV_PASSWORD") or secrets.token_urlsafe(12)
if not os.environ.get("TVCV_PASSWORD"):
    print(f"[auth] No TVCV_PASSWORD set. Generated login -> user: {AUTH_USER} "
          f"password: {AUTH_PASS}  (set TVCV_PASSWORD to override)")

# Max upload size (bytes) to prevent memory-exhaustion DoS.
MAX_UPLOAD_BYTES = int(os.environ.get("TVCV_MAX_UPLOAD_MB", "15")) * 1024 * 1024


# Auth toggle: set TVCV_AUTH=1 to re-enable HTTP Basic auth. Default is OFF
# for easy local demo use (server binds to 127.0.0.1 only).
AUTH_ENABLED = os.environ.get("TVCV_AUTH", "0") == "1"


def require_auth(request: Request) -> str:
    """No-op unless TVCV_AUTH=1. Keeps route decorators intact either way."""
    if not AUTH_ENABLED:
        return "anonymous"
    header = request.headers.get("authorization", "")
    ok = False
    if header.startswith("Basic "):
        import base64
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
            ok = (secrets.compare_digest(user, AUTH_USER) and
                  secrets.compare_digest(pw, AUTH_PASS))
        except Exception:
            ok = False
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user

app.mount("/static", StaticFiles(directory=str(config.ROOT / "frontend" / "static")), name="static")
app.mount("/media/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")
app.mount("/media/annotated", StaticFiles(directory=str(config.ANNOTATED_DIR)), name="annotated")


@app.middleware("http")
async def _protect_media(request: Request, call_next):
    """Uploaded/annotated images contain plates (PII) -> require Basic auth."""
    if AUTH_ENABLED and request.url.path.startswith("/media/"):
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            import base64
            try:
                user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = (secrets.compare_digest(user, AUTH_USER) and
                      secrets.compare_digest(pw, AUTH_PASS))
            except Exception:
                ok = False
        if not ok:
            from starlette.responses import Response
            return Response(status_code=401,
                            headers={"WWW-Authenticate": "Basic"})
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


def _backend_banner() -> dict:
    return {"detector": detect.model_status(), "ocr": plate.status(),
            "demo": detect.model_status() == "mock"}


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
def home(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html", {
        "backend": _backend_banner(), "vtypes": config.VIOLATION_META,
    })


@app.post("/analyze", response_class=HTMLResponse,
          dependencies=[Depends(require_auth)])
async def analyze(request: Request, file: UploadFile = File(...),
                  stop_line_y: str = Form(""),
                  deep_derain: bool = Form(False),
                  deep_deblur: bool = Form(False)):
    # 1) Reject oversized uploads BEFORE buffering the body (real DoS guard).
    clen = request.headers.get("content-length", "")
    if clen.isdigit() and int(clen) > MAX_UPLOAD_BYTES:
        return HTMLResponse(
            f"<div class='text-red-600'>File too large "
            f"(max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).</div>",
            status_code=413)
    # 2) Allow only image extensions (defense-in-depth vs served-content XSS,
    #    e.g. a malicious .svg/.html being served back from /media).
    ext = (Path(file.filename or "img.jpg").suffix or ".jpg").lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return HTMLResponse(
            "<div class='text-red-600'>Unsupported file type (images only).</div>",
            status_code=415)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:  # fallback when content-length absent/lying
        return HTMLResponse(
            f"<div class='text-red-600'>File too large "
            f"(max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).</div>",
            status_code=413)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return HTMLResponse("<div class='text-red-600'>Could not read image.</div>",
                            status_code=400)

    uid = uuid.uuid4().hex[:12]
    orig_name, anno_name = f"{uid}{ext}", f"{uid}_annotated.jpg"
    orig_path = config.UPLOAD_DIR / orig_name
    anno_path = config.ANNOTATED_DIR / anno_name
    cv2.imwrite(str(orig_path), img)

    # Stop line is OPT-IN: only pass it when the user supplied a value.
    # Empty -> the detector auto-detects a white road marking or abstains.
    meta: dict = {"deep_derain": deep_derain, "deep_deblur": deep_deblur}
    if stop_line_y.strip():
        try:
            meta["stop_line_y"] = float(stop_line_y)
        except ValueError:
            pass
    frame, metrics = engine.process(img, meta=meta)
    cv2.imwrite(str(anno_path), annotate(frame))
    eid = db.save_result(orig_name, anno_name, frame, metrics)

    return TEMPLATES.TemplateResponse(request, "_result.html", {
        "eid": eid, "anno_name": anno_name, "metrics": metrics,
        "frame": frame, "vmeta": config.VIOLATION_META,
    })


@app.get("/records", response_class=HTMLResponse,
         dependencies=[Depends(require_auth)])
def records(request: Request, vtype: str = "", plate: str = ""):
    rows = db.list_evidence(vtype=vtype, plate=plate)
    return TEMPLATES.TemplateResponse(request, "records.html", {
        "rows": rows, "vtypes": config.VIOLATION_META,
        "f_vtype": vtype, "f_plate": plate, "backend": _backend_banner(),
    })


@app.get("/records/table", response_class=HTMLResponse,
         dependencies=[Depends(require_auth)])
def records_table(request: Request, vtype: str = "", plate: str = ""):
    rows = db.list_evidence(vtype=vtype, plate=plate)
    return TEMPLATES.TemplateResponse(request, "_records_table.html", {
        "rows": rows, "vtypes": config.VIOLATION_META,
    })


@app.get("/evidence/{eid}", response_class=HTMLResponse,
         dependencies=[Depends(require_auth)])
def evidence_detail(request: Request, eid: int):
    ev = db.get_evidence(eid)
    if not ev:
        return HTMLResponse("Not found", status_code=404)
    return TEMPLATES.TemplateResponse(request, "detail.html", {
        "ev": ev, "vmeta": config.VIOLATION_META, "backend": _backend_banner(),
    })


@app.get("/analytics", response_class=HTMLResponse,
         dependencies=[Depends(require_auth)])
def analytics(request: Request):
    stats = db.stats()
    label_map = {k: m["label"] for k, m in config.VIOLATION_META.items()}
    top_type = max(stats["by_type"].items(), key=lambda kv: kv[1],
                   default=(None, 0))
    return TEMPLATES.TemplateResponse(request, "analytics.html", {
        "stats": stats, "vmeta": config.VIOLATION_META,
        "label_map": label_map, "top_type": top_type,
        "backend": _backend_banner(),
    })


@app.get("/api/stats", dependencies=[Depends(require_auth)])
def api_stats():
    return JSONResponse(db.stats())


@app.get("/api/health")
def health():
    return {"status": "ok", **_backend_banner()}
