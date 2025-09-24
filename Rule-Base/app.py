# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import re

app = FastAPI(title="Urgency Rule-based API")

TH = ["ด่วน","เร่งด่วน","ภายในวันนี้","ภายในพรุ่งนี้","ด่วนมาก","ด่วนที่สุด"]
EN = ["urgent","asap","immediately","reply required","action needed","by eod"]

class Mail(BaseModel):
    subject: str = ""
    body: str = ""
    sender: str = ""             # e.g. "ajarn@ubu.ac.th"
    due_iso: str | None = None   # ISO8601 e.g. "2025-10-03T12:00:00"
    headers: dict | None = None  # optional: {"Importance":"high","X-Priority":"1"}

def has_keyword(text:str)->bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in TH+EN)

def hours_to_due(due_iso:str|None):
    if not due_iso: return None
    try:
        due = datetime.fromisoformat(due_iso)
        return (due - datetime.now()).total_seconds()/3600
    except Exception:
        return None

def sender_weight(sender:str)->int:
    s = (sender or "").lower()
    if s.endswith("@ubu.ac.th"): return 2
    if any(x in s for x in ["ajarn","teacher","advisor","boss"]): return 2
    return 0

def header_weight(headers:dict|None)->int:
    if not headers: return 0
    h = {k.lower():str(v).lower() for k,v in headers.items()}
    w = 0
    if h.get("importance") == "high": w += 2
    if h.get("x-priority") in ["1","high"]: w += 2
    return w

@app.post("/classify")
def classify(mail: Mail):
    text = f"{mail.subject}\n{mail.body}"
    score, reasons = 0, []

    if has_keyword(text):
        score += 3; reasons.append("keyword")

    h = hours_to_due(mail.due_iso)
    if h is not None:
        if h <= 48: score += 4; reasons.append("due<=48h")
        elif h <= 72: score += 2; reasons.append("due<=72h")

    sw = sender_weight(mail.sender)
    if sw: score += sw; reasons.append("important_sender")

    hw = header_weight(mail.headers)
    if hw: score += hw; reasons.append("priority_header")

    label = "normal"
    if score >= 6: label = "urgent"
    elif score >= 3: label = "medium"

    return {
        "label": label,
        "score": score,
        "reasons": reasons,
        "hours_to_due": h
    }
