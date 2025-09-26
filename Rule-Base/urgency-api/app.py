from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os, re

# Hugging Face (Zero-shot)
from transformers import pipeline

app = FastAPI(title="Urgency Classifier (Regex + Rules + Zero-shot)")

# -----------------------------
# Config
# -----------------------------
HF_MODEL = os.getenv("HF_MODEL", "valhalla/distilbart-mnli-12-3")  # เบา เร็ว
CANDIDATE_LABELS = ["urgent", "medium", "normal"]

# Weights / Thresholds (ปรับได้)
W_KEYWORD = 3
W_DUE_48H = 4
W_DUE_72H = 2
W_SENDER = 2
W_HEADER = 2
THRESH_URGENT = 6
THRESH_MEDIUM = 3

# หาก zero-shot ทำนายเป็น urgent ด้วย prob สูง ให้เสริมแต้ม
ZSL_URGENT_BONUS_AT = 0.70  # ถ้า prob urgent >= 0.70
W_ZSL_URGENT = 3

# -----------------------------
# Regex patterns (TH + EN)
# -----------------------------
THAI_MONTHS = {
    "ม.ค.":1,"ก.พ.":2,"มี.ค.":3,"เม.ย.":4,"พ.ค.":5,"มิ.ย.":6,
    "ก.ค.":7,"ส.ค.":8,"ก.ย.":9,"ต.ค.":10,"พ.ย.":11,"ธ.ค.":12,
    "มกราคม":1,"กุมภาพันธ์":2,"มีนาคม":3,"เมษายน":4,"พฤษภาคม":5,"มิถุนายน":6,
    "กรกฎาคม":7,"สิงหาคม":8,"กันยายน":9,"ตุลาคม":10,"พฤศจิกายน":11,"ธันวาคม":12,
}

RE_URGENT_KW = re.compile(
    r"\b(ด่วน|เร่งด่วน|ภายในวันนี้|ภายในพรุ่งนี้|urgent|asap|immediately|reply\s*required|by\s*eod)\b",
    re.IGNORECASE
)

RE_ABS_DUE = re.compile(
    r'ครบกำหนด\s*(?:วันที่\s*)?(\d{1,2})\s*'
    r'(ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.|'
    r'มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)',
    re.IGNORECASE
)

RE_REL_DUE_TH = re.compile(r'\b(วันนี้|พรุ่งนี้|มะรืนนี้)\b')
RE_REL_DUE_EN = re.compile(r'\b(today|tomorrow|day\s+after\s+tomorrow)\b', re.IGNORECASE)

RE_TIME_BEFORE_TH = re.compile(r'ก่อน\s*(\d{1,2})(?:[:\.](\d{2}))?\s*(?:น\.|นาฬิกา)?', re.IGNORECASE)
RE_TIME_BEFORE_EN = re.compile(r'before\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', re.IGNORECASE)

RE_TITLE_IN_QUOTES = re.compile(r'ครบกำหนด[^:\n]*[:：]\s*[\"“](.+?)[\"”]', re.IGNORECASE)

# -----------------------------
# Zero-shot pipeline (โหลดครั้งเดียว)
# -----------------------------
zsl = pipeline("zero-shot-classification", model=HF_MODEL)

# -----------------------------
# Schemas
# -----------------------------
class Mail(BaseModel):
    subject: str = ""
    body: str = ""
    sender: str = ""              # เช่น "ajarn@ubu.ac.th"
    due_iso: Optional[str] = None # ถ้ามีเวลาจากระบบอื่นส่งมา
    headers: Optional[Dict[str, Any]] = None  # {"Importance":"high","X-Priority":"1"}

class Result(BaseModel):
    label: str
    score: float
    reasons: List[str]
    hours_to_due: Optional[float] = None
    zsl_label: Optional[str] = None
    zsl_scores: Optional[Dict[str, float]] = None
    extracted: Optional[Dict[str, Any]] = None

# -----------------------------
# Helpers
# -----------------------------
def parse_thai_date(day: int, month_text: str, year: int=None) -> Optional[datetime]:
    m = THAI_MONTHS.get(month_text.strip(), None)
    if not m: return None
    if year is None:
        year = datetime.now().year
    try:
        return datetime(year, m, day)
    except ValueError:
        return None

def parse_due_from_text(text: str) -> Optional[datetime]:
    now = datetime.now()
    due_date = None

    # absolute (เช่น 21 ก.ย.)
    ma = RE_ABS_DUE.search(text)
    if ma:
        d = int(ma.group(1))
        mtxt = ma.group(2)
        due_date = parse_thai_date(d, mtxt, now.year)

    # relative (ไทย/อังกฤษ)
    if not due_date:
        mr_th = RE_REL_DUE_TH.search(text)
        if mr_th:
            word = mr_th.group(1)
            delta = 0 if word=="วันนี้" else 1 if word=="พรุ่งนี้" else 2
            due_date = (now + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            mr_en = RE_REL_DUE_EN.search(text)
            if mr_en:
                word = mr_en.group(1).lower()
                delta = 0 if "today" in word else 1 if "tomorrow" in word else 2
                due_date = (now + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)

    # time qualifiers
    if due_date:
        mt_th = RE_TIME_BEFORE_TH.search(text)
        mt_en = RE_TIME_BEFORE_EN.search(text)
        if mt_th:
            hh = int(mt_th.group(1)); mm = int(mt_th.group(2) or 0)
            due_date = due_date.replace(hour=hh, minute=mm, second=0, microsecond=0)
        elif mt_en:
            hh = int(mt_en.group(1)); mm = int(mt_en.group(2) or 0)
            ampm = (mt_en.group(3) or "").lower()
            if ampm == "pm" and hh < 12: hh += 12
            if ampm == "am" and hh == 12: hh = 0
            due_date = due_date.replace(hour=hh, minute=mm, second=0, microsecond=0)

    return due_date

def hours_to_due(due: Optional[datetime]) -> Optional[float]:
    if not due: return None
    return (due - datetime.now()).total_seconds()/3600.0

def sender_weight(sender: str) -> int:
    s = (sender or "").lower()
    if s.endswith("@ubu.ac.th"): return W_SENDER
    if any(x in s for x in ["ajarn","teacher","advisor","boss","หัวหน้า","อาจารย์"]):
        return W_SENDER
    return 0

def header_weight(headers: Optional[Dict[str, Any]]) -> int:
    if not headers: return 0
    h = {str(k).lower(): str(v).lower() for k,v in headers.items()}
    w = 0
    if h.get("importance") == "high": w += W_HEADER
    if h.get("x-priority") in ["1","high"]: w += W_HEADER
    return w

def extract_fields(text: str) -> Dict[str, Any]:
    title = None
    mt = RE_TITLE_IN_QUOTES.search(text)
    if mt: title = mt.group(1).strip()
    due_dt = parse_due_from_text(text)
    kws = [m.group(1) for m in RE_URGENT_KW.finditer(text)]
    return {
        "title": title,
        "due_iso_from_text": due_dt.isoformat() if due_dt else None,
        "urgent_keywords": kws,
    }

def zero_shot_predict(text: str):
    res = zsl(text, CANDIDATE_LABELS)
    labels, scores = res["labels"], res["scores"]
    mapping = {lab: float(sc) for lab, sc in zip(labels, scores)}
    top = labels[0]
    return top, mapping

# -----------------------------
# Core classification
# -----------------------------
def classify_core(mail: Mail) -> Result:
    text = (mail.subject or "") + "\n" + (mail.body or "")
    extracted = extract_fields(text)

    # 1) due time: from provided due_iso OR parsed from text
    due_dt = None
    if mail.due_iso:
        try:
            due_dt = datetime.fromisoformat(mail.due_iso)
        except Exception:
            pass
    if not due_dt and extracted["due_iso_from_text"]:
        try:
            due_dt = datetime.fromisoformat(extracted["due_iso_from_text"])
        except Exception:
            pass

    h = hours_to_due(due_dt)

    # 2) rule-based score
    score = 0
    reasons: List[str] = []

    # keyword
    if extracted["urgent_keywords"]:
        score += W_KEYWORD
        reasons.append("keyword")

    # due
    if h is not None:
        if h <= 48:
            score += W_DUE_48H; reasons.append("due<=48h")
        elif h <= 72:
            score += W_DUE_72H; reasons.append("due<=72h")

    # sender & headers
    sw = sender_weight(mail.sender)
    if sw: score += sw; reasons.append("important_sender")
    hw = header_weight(mail.headers)
    if hw: score += hw; reasons.append("priority_header")

    # base label จากกฎ
    label = "normal"
    if score >= THRESH_URGENT: label = "urgent"
    elif score >= THRESH_MEDIUM: label = "medium"

    # 3) zero-shot เสริม
    z_label, z_scores = zero_shot_predict(text)

    # กรณี regex จับตรงมาก ๆ: ถ้า keyword + due<=48h → ยึด urgent ทันที (เพื่อ precision สูง)
    if ("keyword" in reasons) and ("due<=48h" in reasons):
        final_label = "urgent"
        final_score = 1.0
    else:
        # ensemble logic:
        final_label = label
        final_score = min(score / 10.0, 1.0)  # scale 0..1 สำหรับแสดงผล

        # ถ้ากฎยังไม่แรง → ให้ zero-shot ช่วยดันขึ้น
        if label == "normal":
            # ถ้าโมเดลบอก urgent ชัดเจน ก็เชื่อโมเดล
            if z_scores.get("urgent", 0.0) >= ZSL_URGENT_BONUS_AT:
                final_label = "urgent"
                final_score = max(final_score, z_scores["urgent"])
            else:
                # ไม่ถึงเกณฑ์ → อาจเป็น medium ตามโมเดล
                if z_label == "medium":
                    final_label = "medium"
                    final_score = max(final_score, z_scores["medium"])
        elif label == "medium":
            # ถ้าโมเดลมั่นใจว่า urgent → ดันเป็น urgent
            if z_scores.get("urgent", 0.0) >= ZSL_URGENT_BONUS_AT:
                final_label = "urgent"
                final_score = max(final_score, z_scores["urgent"])
        # ถ้า label จากกฎเป็น urgent อยู่แล้วก็ถือว่ามั่นใจ

    return Result(
        label=final_label,
        score=round(float(final_score), 4),
        reasons=reasons,
        hours_to_due=h,
        zsl_label=z_label,
        zsl_scores={k: round(v,4) for k,v in z_scores.items()},
        extracted=extracted
    )

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": HF_MODEL}

@app.post("/classify", response_model=Result)
def classify(mail: Mail):
    return classify_core(mail)
