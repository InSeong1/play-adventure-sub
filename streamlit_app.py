# -*- coding: utf-8 -*-
import os, io, re, json, time, base64, uuid, datetime, struct, wave, hashlib, math, platform
from typing import List, Dict, Tuple, Optional

import streamlit as st
import requests
from difflib import SequenceMatcher

# 선택 의존성 ─────────────────────────────────────────────────────────
try:
    from pydub import AudioSegment, effects, silence as _silence
    from pydub.utils import which as _which
except Exception:
    AudioSegment, effects, _silence, _which = None, None, None, None
try:
    from audio_recorder_streamlit import audio_recorder
except Exception:
    audio_recorder = None

# librosa(프로소디 분석, 선택)
try:
    import librosa as _lb
    import soundfile as _sf
except Exception:
    _lb = None
try:
    import numpy as _np
except Exception:
    _np = None
try:
    import webrtcvad as _vad  # 선택, 없으면 무시
except Exception:
    _vad = None

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# OpenAI (Chat/TTS)
from openai import OpenAI

# ───────── 시크릿
OPENAI_API_KEY       = st.secrets.get("OPENAI_API_KEY", "")
CLOVA_SPEECH_SECRET  = st.secrets.get("CLOVA_SPEECH_SECRET", "")
NAVER_CLOVA_OCR_URL  = st.secrets.get("NAVER_CLOVA_OCR_URL", "")
NAVER_OCR_SECRET     = st.secrets.get("NAVER_OCR_SECRET", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# ───────── ffmpeg 경로 안전 장치 ──────────────────────────────────
def _ensure_ffmpeg_path():
    """pydub이 ffmpeg를 못 찾을 때, 윈도우 공통 경로를 자동 시도."""
    if _which is None or AudioSegment is None:
        return
    try:
        if _which("ffmpeg") and _which("ffprobe"):
            return
    except Exception:
        pass
    # Windows에서 흔한 설치 경로 시도
    candidates = [
        r"C:\\ffmpeg\\bin",
        r"C:\\Program Files\\ffmpeg\\bin",
        r"C:\\Program Files (x86)\\ffmpeg\\bin",
    ]
    for p in candidates:
        ff = os.path.join(p, "ffmpeg.exe")
        if os.path.exists(ff):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            try:
                AudioSegment.converter = ff
                prob = os.path.join(p, "ffprobe.exe")
                if os.path.exists(prob):
                    AudioSegment.ffprobe = prob
            except Exception:
                pass
            break

_ensure_ffmpeg_path()

# ───────── UI (파스텔 + 상단 잘림 보정 + 다크모드 대응) ───────────────────────────
PASTEL_CSS = """
<style>
/* 라이트모드 변수 */
:root { 
    --bg:#fff7f2; 
    --card:#ffffff; 
    --accent:#ffd8cc; 
    --accent2:#ffeab3; 
    --ink:#2f3437;
    --ok:#2e7d32; 
    --warn:#ef6c00; 
    --bad:#c62828; 
    --muted:#667085; 
}

/* 다크모드 변수 */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:#1a1a1a;
        --card:#2d2d2d;
        --accent:#ff6b6b;
        --accent2:#4ecdc4;
        --ink:#ffffff;
        --ok:#4caf50;
        --warn:#ff9800;
        --bad:#f44336;
        --muted:#b0b0b0;
    }
}

/* Streamlit 다크모드 감지 */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}

body { 
    background: var(--bg); 
    color: var(--ink) !important;
}

/* ✅ 상단/탭 글자 잘림 방지 */
.block-container { 
    background: var(--card) !important; 
    border-radius:14px; 
    padding:24px 22px 18px !important; 
    overflow:visible !important; 
    color: var(--ink) !important;
}

div[data-testid="stHeader"] { height:auto !important; }
div[data-testid="stMarkdownContainer"] h1, h1, h2, h3, h4, h5, h6 { 
    line-height:1.3 !important; 
    margin-top:0.25rem !important; 
    color: var(--ink) !important;
}

[role="radiogroup"] label p { 
    line-height:1.25 !important; 
    color: var(--ink) !important;
}

/* 기본 버튼/카드 */
.stButton>button { 
    background: linear-gradient(90deg,var(--accent),var(--accent2));
    color: var(--ink); 
    border:none; 
    border-radius:10px; 
    padding:10px 16px; 
    font-weight:700; 
}

.stButton>button:hover { opacity:.93 }
hr { border-color: var(--accent); }
.small { font-size: 0.9rem; color: var(--muted); }

.card { 
    background: var(--card); 
    padding:14px 14px; 
    border-radius:12px; 
    border:1px solid var(--accent); 
    color: var(--ink) !important;
}

.card h4 { 
    margin:0 0 8px 0; 
    font-size:1.05rem; 
    color: var(--ink) !important; 
}

.badge { 
    display:inline-block; 
    padding:4px 10px; 
    border-radius:999px; 
    font-weight:700; 
    font-size:0.85rem; 
}

.b-ok { 
    background: var(--ok); 
    color: white; 
    border:1px solid var(--ok); 
}

.b-warn { 
    background: var(--warn); 
    color: white; 
    border:1px solid var(--warn); 
}

.b-bad { 
    background: var(--bad); 
    color: white; 
    border:1px solid var(--bad); 
}

.kv { 
    display:flex; 
    align-items:center; 
    gap:8px; 
    margin:6px 0; 
}

.kv .k { 
    width:130px; 
    color: var(--muted); 
} 

.kv .v { 
    font-weight:700; 
    color: var(--ink); 
}

.gauge { 
    width:100%; 
    height:10px; 
    background: var(--muted); 
    border-radius:999px; 
    position:relative; 
    overflow:hidden; 
}

.gauge > span { 
    position:absolute; 
    left:0; 
    top:0; 
    bottom:0; 
    width:0%; 
    background:linear-gradient(90deg,var(--accent),var(--accent2));
    border-radius:999px; 
    transition:width .35s ease; 
}

.hi { 
    background: var(--card); 
    border:1px dashed var(--accent); 
    border-radius:8px; 
    padding:8px 10px; 
    color: var(--ink) !important;
}

.hi span { padding:0 2px; }
.hi .ok { color: var(--ok); font-weight:700; }
.hi .miss { color: var(--bad); text-decoration:underline; }

/* 간단 REC 점 */
.rec-dot { 
    width:10px; 
    height:10px; 
    border-radius:50%; 
    display:inline-block; 
    margin-right:6px; 
    background: var(--muted); 
}

.rec-on { 
    background: var(--bad); 
    box-shadow:0 0 0 6px rgba(244,67,54,.2); 
}

/* Streamlit 기본 요소들 다크모드 대응 */
.stMarkdown, .stText, .stCodeBlock, .stDataFrame {
    color: var(--ink) !important;
}

/* 사이드바 다크모드 대응 */
section[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    color: var(--ink) !important;
    width: 300px !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--ink) !important;
}

/* 탭 다크모드 대응 */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--card) !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--ink) !important;
}

/* 파일 업로더 다크모드 대응 */
.stFileUploader {
    color: var(--ink) !important;
}

/* 텍스트 영역 다크모드 대응 */
.stTextArea textarea {
    background-color: var(--card) !important;
    color: var(--ink) !important;
    border-color: var(--accent) !important;
}

/* 셀렉트박스 다크모드 대응 */
.stSelectbox select {
    background-color: var(--card) !important;
    color: var(--ink) !important;
    border-color: var(--accent) !important;
}

/* 체크박스 다크모드 대응 */
.stCheckbox label {
    color: var(--ink) !important;
}

/* 라디오 버튼 다크모드 대응 */
.stRadio label {
    color: var(--ink) !important;
}

/* 숫자 입력 다크모드 대응 */
.stNumberInput input {
    background-color: var(--card) !important;
    color: var(--ink) !important;
    border-color: var(--accent) !important;
}

/* 사이드바 radio 버튼 구분선 */
.stRadio > div > div > label:nth-child(2) {
    border-top: 2px solid var(--accent) !important;
    margin-top: 10px !important;
    padding-top: 10px !important;
}

.stRadio > div > div > label:nth-child(4) {
    border-top: 2px solid var(--accent) !important;
    margin-top: 10px !important;
    padding-top: 10px !important;
}
</style>
"""

# ───────── 공통 유틸 ────────────────────────────────────────────────
def clean_script_text(t: str) -> str:
    return (t or "").replace("\r\n","\n").replace("\r","\n").strip()

# 머릿말/장면/지문 역할 제외 & 역할명 정규화
BANNED_ROLE_PATTERNS = [
    r"^\**\s*장면", r"^\**\s*씬", r"^\**\s*무대", r"^\**\s*배경", r"^\**\s*배경음",
    r"^\**\s*노래", r"^\**\s*노랫말", r"^\**\s*설명", r"^\**\s*지문", r"^\**\s*장내",
    r"^\**\s*효과음"
]

def _normalize_role(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^\**|\**$", "", s)
    s = re.sub(r"^[\(\[]\s*|\s*[\)\]]$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _is_banned_role(name: str) -> bool:
    for p in BANNED_ROLE_PATTERNS:
        if re.search(p, name): return True
    return False

def extract_roles(script: str) -> List[str]:
    roles=[]
    for line in clean_script_text(script).splitlines():
        m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
        if not m: continue
        who = _normalize_role(m.group(1))
        if _is_banned_role(who) or who=="": continue
        if who not in roles: roles.append(who)
    return roles

def build_sequence(script: str) -> List[Dict]:
    seq=[]
    for line in clean_script_text(script).splitlines():
        m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
        if not m: continue
        who = _normalize_role(m.group(1))
        if _is_banned_role(who) or who=="":
            continue
        text = m.group(2).strip()
        seq.append({"who":who, "text":text})
    return seq

# 문자열 정규화(일치율 개선)
_PUNC = r"[^\w가-힣ㄱ-ㅎㅏ-ㅣ ]"

def _norm_for_ratio(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(_PUNC, " ", s)
    s = re.sub(r"\s+", "", s)
    return s

def match_highlight_html(expected: str, spoken: str) -> Tuple[str, float]:
    tokens = re.split(r"(\s+)", expected.strip())
    sp_norm = _norm_for_ratio(spoken or "")
    out=[]
    for tok in tokens:
        if tok.isspace(): out.append(tok); continue
        if not tok: continue
        ok = _norm_for_ratio(tok) and _norm_for_ratio(tok) in sp_norm
        out.append(f"<span class='{ 'ok' if ok else 'miss' }'>{tok}</span>")
    ratio = SequenceMatcher(None, _norm_for_ratio(expected), _norm_for_ratio(spoken or "")).ratio()
    return "<div class='hi'>"+"".join(out)+"</div>", ratio

def similarity_score(expected: str, spoken: str) -> float:
    def ko_only(s):
        s = re.sub(r"\(.*?\)", "", s); return "".join(re.findall(r"[가-힣0-9]+", s))
    e = ko_only(expected or ""); s = ko_only(spoken or "")
    if not s: return 0.0
    ratio = SequenceMatcher(None, e, s).ratio()
    ew = set(re.findall(r"[가-힣0-9]+", expected or ""))
    sw = set(re.findall(r"[가-힣0-9]+", spoken or ""))
    jacc = len(ew & sw) / max(1, len(ew | sw)) if (ew or sw) else 0.0
    # LCS
    def lcs_len(a,b):
        dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                dp[i][j]=dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
    l = lcs_len(e, s)
    prec = l/max(1,len(s)); rec = l/max(1,len(e))
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    return max(ratio, jacc, f1)

# ───────── OpenAI TTS (지문 미낭독 + 성별 톤 보정) ─────────────────────
VOICE_KR_LABELS_SAFE = [
    "민준 (남성, 따뜻하고 친근한 목소리)",
    "현우 (남성, 차분하고 신뢰감 있는 목소리)", 
    "지호 (남성, 활기차고 밝은 목소리)",
    "지민 (여성, 부드럽고 친절한 목소리)",
    "소연 (여성, 귀엽고 명랑한 목소리)",
    "하은 (여성, 차분하고 우아한 목소리)",
    "민지 (여성, 밝고 경쾌한 목소리)"
]

VOICE_MAP_SAFE = {
    "민준 (남성, 따뜻하고 친근한 목소리)": "alloy",
    "현우 (남성, 차분하고 신뢰감 있는 목소리)": "verse", 
    "지호 (남성, 활기차고 밝은 목소리)": "onyx",
    "지민 (여성, 부드럽고 친절한 목소리)": "nova",
    "소연 (여성, 귀엽고 명랑한 목소리)": "shimmer",
    "하은 (여성, 차분하고 우아한 목소리)": "coral",
    "민지 (여성, 밝고 경쾌한 목소리)": "echo"
}

def _pitch_shift_mp3(mp3_bytes: bytes, semitones: float) -> bytes:
    """pydub+ffmpeg 필요. 실패하면 원본 반환."""
    if not AudioSegment or semitones==0:
        return mp3_bytes
    try:
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        new_fr = int(seg.frame_rate * (2.0 ** (semitones/12.0)))
        shifted = seg._spawn(seg.raw_data, overrides={'frame_rate': new_fr}).set_frame_rate(seg.frame_rate)
        out = io.BytesIO(); shifted.export(out, format="mp3"); return out.getvalue()
    except Exception:
        return mp3_bytes

def tts_speak_line(text: str, voice_label: str) -> Tuple[str, Optional[bytes]]:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 필요합니다."); return text, None
    voice_id = VOICE_MAP_SAFE.get(voice_label, "alloy")
    speak_text = re.sub(r"\(.*?\)", "", text).strip()
    try:
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model":"gpt-4o-mini-tts","voice":voice_id,"input":speak_text,"format":"mp3"},
            timeout=60
        )
        if r.status_code!=200:
            st.error(f"TTS 오류: {r.status_code} - {r.text}"); return speak_text, None
        audio = r.content
        # 목소리별 톤 보정 (성별과 특성 고려)
        if "여성" in voice_label:
            if "지민" in voice_label:
                audio = _pitch_shift_mp3(audio, +2.5)
            elif "소연" in voice_label:
                audio = _pitch_shift_mp3(audio, +3.0)
            elif "하은" in voice_label:
                audio = _pitch_shift_mp3(audio, +2.0)
            elif "민지" in voice_label:
                audio = _pitch_shift_mp3(audio, +2.8)
            else:
                audio = _pitch_shift_mp3(audio, +2.0)
        elif "남성" in voice_label:
            if "10대" in voice_label:
                audio = _pitch_shift_mp3(audio, +1.0)
            elif "20대" in voice_label:
                audio = _pitch_shift_mp3(audio, -0.5)
            elif "30대" in voice_label:
                audio = _pitch_shift_mp3(audio, -1.0)
        return speak_text, audio
    except Exception as e:
        st.error(f"TTS 오류: {e}")
        return speak_text, None

# ───────── STT 전처리 + CLOVA Short Sentence STT ───────────────────
def preprocess_audio_for_stt(audio_bytes: bytes) -> bytes:
    if not AudioSegment:
        return audio_bytes
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # 앞뒤 무음 트림
        def _lead_sil(seg, silence_thresh=-40.0, chunk_ms=10):
            trim_ms = 0
            while trim_ms < len(seg) and seg[trim_ms:trim_ms+chunk_ms].dBFS < silence_thresh:
                trim_ms += chunk_ms
            return trim_ms
        start = _lead_sil(seg); end = _lead_sil(seg.reverse())
        if start+end < len(seg): seg = seg[start:len(seg)-end]
        # 필터/노멀라이즈
        try: seg = seg.high_pass_filter(100).low_pass_filter(4000)
        except Exception: pass
        try: seg = effects.normalize(seg, headroom=3.0)
        except Exception: pass
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        buf = io.BytesIO(); seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        return audio_bytes

def clova_short_stt(audio_bytes: bytes, lang: str = "Kor") -> str:
    if not CLOVA_SPEECH_SECRET:
        return ""
    url = f"https://clovaspeech-gw.ncloud.com/recog/v1/stt?lang={lang}"
    headers = {"X-CLOVASPEECH-API-KEY": CLOVA_SPEECH_SECRET, "Content-Type": "application/octet-stream"}
    wav_bytes = preprocess_audio_for_stt(audio_bytes)
    r = requests.post(url, headers=headers, data=wav_bytes, timeout=60)
    r.raise_for_status()
    try:
        return r.json().get("text","").strip()
    except Exception:
        return r.text.strip()

# ───────── OCR(선택) ────────────────────────────────────────────────
def nv_ocr(img_bytes: bytes) -> str:
    if not NAVER_CLOVA_OCR_URL or not NAVER_OCR_SECRET:
        return "(OCR 설정 필요)"
    payload={"version":"V2","requestId":str(uuid.uuid4()),
             "timestamp":int(datetime.datetime.now(datetime.UTC).timestamp()*1000),
             "images":[{"name":"img","format":"jpg","data":base64.b64encode(img_bytes).decode()}]}
    try:
        res=requests.post(NAVER_CLOVA_OCR_URL,headers={"X-OCR-SECRET":NAVER_OCR_SECRET,"Content-Type":"application/json"},
                          json=payload,timeout=30).json()
        return " ".join(f["inferText"] for f in res["images"][0]["fields"])
    except Exception as e:
        return f"(OCR 오류: {e})"

# ───────── PDF(글꼴 자동탐색) ───────────────────────────────────────
def _register_font_safe():
    candidates = [
        r"C:\\Windows\\Fonts\\malgun.ttf", r"C:\\Windows\\Fonts\\NanumGothic.ttf",
        "/System/Library/Fonts/AppleGothic.ttf", "/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            try: pdfmetrics.registerFont(TTFont("KFont", p)); return "KFont"
            except Exception: pass
    return None

def build_cuecards_pdf(script: str, role: str) -> Optional[bytes]:
    font_name = _register_font_safe()
    buf = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buf, pagesize=A4)
        style = ParagraphStyle("K", fontName=(font_name or "Helvetica"), fontSize=12, leading=16)
        elems=[Paragraph(f"[큐카드] {role}", style), Spacer(1,12)]
        for i, line in enumerate(build_sequence(script),1):
            if line["who"] == role:
                txt = re.sub(r"\(.*?\)","", line["text"]).strip()
                elems.append(Paragraph(f"{i}. {txt}", style)); elems.append(Spacer(1,8))
        doc.build(elems)
        return buf.getvalue()
    except Exception as e:
        st.warning(f"PDF 생성 오류: {e}"); return None

# ───────── 세션 피드백 프롬프트 ─────────────────────────────────────
def prompt_session_feedback(turns: List[Dict]) -> str:
    return ("연극 대사 연습 기록입니다. 말속도, 어조, 목소리 크기를 중심으로 "
            "칭찬/개선점/다음 연습 팁을 간결히 써주세요.\n\n"+json.dumps(turns, ensure_ascii=False, indent=2))

# ───────── 프로소디 분석: WAV 폴백 포함 ────────────────────────────

# (중략: analyze_prosody, render_prosody_card 등 기존 그대로 유지 — 변경 없음)

# === ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ ===
# === 대사 수 조절기 관련 최소 변경 코드(유틸 + 생성 루프 + 버튼 블록) ===
# === 다른 기능에는 영향 없음 ======================================

# (A) 역할별 줄 수 세기 — 새로운 유틸 함수 추가

def _count_lines_by_role(text: str, roles: List[str]) -> Dict[str, int]:
    counts = {r: 0 for r in roles}
    for line in clean_script_text(text).splitlines():
        m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        who = _normalize_role(m.group(1))
        if who in counts:
            counts[who] += 1
    return counts

# (B) 목표 줄 수를 정확히 맞출 때까지 생성→검증→재시도 루프 — 새 함수 추가

def _rebalance_with_hard_targets(
    client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 4
) -> str:
    """역할/목표 줄수를 '정확히' 맞출 때까지 AI에 재시도 요청."""
    role_list = ", ".join(roles)
    target_lines = "\n".join([f"- {r}: {targets[r]}줄" for r in roles])

    sys = (
        "당신은 초등 연극 대본 편집자입니다. 사용자가 정한 '역할별 목표 대사 수'를 "
        "정확히 맞추는 것이 최우선 과제입니다. 자연스러운 연결을 유지하되, "
        "목표 줄 수와 다르면 반드시 추가/삭제/합치기/쪼개기를 통해 정확히 맞추세요. "
        "지문은 괄호 ( ) 만 사용하며 대사 수에는 포함하지 않습니다. "
        "등장인물은 제공된 목록만 사용하세요."
    )

    base_prompt = f"""
원본 대본:
{original_script}

등장인물: {role_list}

목표 대사 수:
{target_lines}

요구사항:
1) 각 인물의 대사 수를 목표에 '정확히' 맞출 것
2) 지문(괄호)은 자유롭게 쓰되 대사 수 계산에는 포함하지 말 것
3) 새로운 인물은 추가 금지, 이름 철자도 동일 유지
4) 기-승-전-결 흐름은 간결히 유지, 불필요 반복은 정리
5) 출력은 '대본 텍스트만' 주세요 (설명/표/머릿말 금지)
6) 대본의 전체 맥락을 충분히 파악하여 인물 간 대사가 자연스럽게 주고받아지도록, 불필요한 반복은 정리하고 연결부(받아치기·반응·전환)를 적절히 보강할 것
"""

    # 최초 시도
    msg = [{"role": "system", "content": sys}, {"role": "user", "content": base_prompt}]
    last_script = None

    for attempt in range(1, max_tries + 1):
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            temperature=0.6,
            max_tokens=2000,
        )
        draft = res.choices[0].message.content.strip()

        # 원본 등장인물만 유지(기존 로직과 동일)
        filtered_lines = []
        for line in clean_script_text(draft).splitlines():
            m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if m:
                character = _normalize_role(m.group(1))
                if character in roles:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        draft = "\n".join(filtered_lines)

        counts = _count_lines_by_role(draft, roles)
        ok = all(counts.get(r, 0) == targets[r] for r in roles)

        if ok:
            return draft  # 성공

        # 다음 시도를 위한 차이 지시
        diff_lines = "\n".join([f"- {r}: 현재 {counts.get(r,0)}줄 → 목표 {targets[r]}줄" for r in roles if counts.get(r,0) != targets[r]])
        fix_prompt = f"""
아래 차이를 반영해 대본을 '수정'만 하세요. 전체를 갈아엎지 말고, 필요한 대사만 추가/삭제/합치기/쪼개기 하세요.
차이:
{diff_lines}

주의:
- 장면/지문은 유지하며, 인물/이름은 바꾸지 마세요.
- 대사 내용은 맥락상 자연스럽게 이어지게 하세요. (받아치기·반응·전환을 보강하여 어색한 점을 제거)
- 출력은 대본 텍스트만.
기존 초안:
{draft}
"""
        msg = [{"role": "system", "content": sys}, {"role": "user", "content": fix_prompt}]
        last_script = draft

    # 최대 시도 실패 시 마지막 초안 반환
    return last_script or original_script

# === ↑↑↑↑↑↑↑ 여기까지 신규 코드(대사 수 조절기 전용) ==================

# ───────── 페이지 1: 대본 등록/입력 ──────────────────────────────
# (기존 page_script_input 그대로 — 변경 없음)

# ───────── 페이지 2: 대본 피드백 & 완성본 생성 ─────────────────────
# (기존 page_feedback_script 그대로 — 변경 없음)

# ───────── 페이지 3: 대사 수 조절하기 ───────────────────────────────────
# ↓↓↓ 이 함수의 '재분배하기' 버튼 내부만 교체. 나머지 로직/표시는 동일 유지

def page_role_balancer():
    st.header("⚖️ 3) 대사 수 조절하기")
    script = st.session_state.get("current_script") or st.session_state.get("script_balanced") or st.session_state.get("script_final") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/생성하세요."); return
    seq = build_sequence(script); roles = extract_roles(script)
    if not roles: st.info("‘이름: 내용’ 형식이어야 역할을 인식해요."); return

    if st.session_state.get("script_balanced"):
        balanced_seq = build_sequence(st.session_state["script_balanced"])
        counts = {r:0 for r in roles}
        for ln in balanced_seq: counts[ln["who"]]+=1
        st.session_state["current_counts"] = counts
    else:
        counts = {r:0 for r in roles}
        for ln in seq: counts[ln["who"]]+=1
        st.session_state["current_counts"] = counts

    st.subheader("📜 현재 대본")
    current_script = st.session_state.get("script_balanced") or script
    st.code(current_script, language="text",height=500)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 현재 대사 수")
        st.markdown("생성된 대본의 줄 수를 알려줘요.")
        for role, count in counts.items():
            st.write(f"**{role}**: {count}줄")
    with col2:
        st.subheader("🎯 목표 대사 수 설정")
        st.markdown("대사의 양을 골고루 분배해 보아요.")
        targets={}
        for r in roles:
            targets[r]=st.number_input(f"{r} 목표", min_value=0, value=counts[r], step=1, key=f"tgt_{r}")

    st.markdown("---")
    if st.button("🔁 재분배하기", key="btn_rebalance", use_container_width=True):
        loading_placeholder = st.empty()
        loading_placeholder.info("⚖️ 역할을 재분배하고 있습니다...")

        original_script = st.session_state.get("script_raw", script)

        try:
            # ✅ 생성→검증→재시도 루프(정확 매칭)
            new_script = _rebalance_with_hard_targets(client, original_script, roles, targets, max_tries=4)

            final_counts = _count_lines_by_role(new_script, roles)
            mismatches = [r for r in roles if final_counts.get(r,0) != targets[r]]

            st.session_state["script_balanced"] = new_script
            st.session_state["current_script"] = new_script
            loading_placeholder.empty()

            if mismatches:
                st.warning("재시도 후에도 일부 인물의 줄 수가 정확히 맞지 않았습니다. 아래 수치를 확인해 주세요.")
                for r in roles:
                    st.write(f"- {r}: 현재 {final_counts.get(r,0)}줄 / 목표 {targets[r]}줄")
            else:
                st.success("✅ 재분배 완료! (모든 목표 줄 수 일치)")

            st.rerun()
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"재분배 중 오류가 발생했습니다: {e}")
            return

# ───────── 페이지 4: 소품·무대·의상 ─────────────────────────────────
# (기존 page_stage_kits 그대로 — 변경 없음)

# ───────── 페이지 5: AI 대본 연습 ────────────────────────────────
# (기존 page_rehearsal_partner 그대로 — 변경 없음)

# ───────── 사이드바 상태 ─────────────────────────────────────────
# (기존 sidebar_status 그대로 — 변경 없음)

# ───────── MAIN ────────────────────────────────────────────────────
# (기존 main 그대로 — 변경 없음)

def page_script_input():
    st.image("assets/dragon_intro.png", width='stretch')
    st.header("📥 1) 대본 등록")
    c1,c2 = st.columns(2)
    with c1:
        up = st.file_uploader("손글씨/이미지 업로드(OCR)", type=["png","jpg","jpeg"], key="u_ocr")
        if up and st.button("🖼️ OCR로 불러오기", key="btn_ocr"):
            txt = nv_ocr(up.read())
            st.session_state["script_raw"] = (st.session_state.get("script_raw","") + "\n" + (txt or "")).strip()
            st.success("OCR 완료!")
    with c2:
        st.caption("형식 예: 민수: (창밖을 보며) 오늘은 비가 올까?\n\n해설은 반드시 \"해설:\"로 표기해 주세요.")
        val = st.text_area("대본 직접 입력", height=260, value=st.session_state.get("script_raw",""), key="ta_script")
        if st.button("💾 저장 (저장 버튼을 반드시 눌러주세요!)", key="btn_save_script"):
            st.session_state["script_raw"] = val.strip(); st.success("저장되었습니다. 왼쪽 메뉴에서 다음 페이지로 이동해주세요!")

def page_feedback_script():
    st.header("🛠️ 2) 대본 피드백 & 완성본 생성")
    script = st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/업로드하세요."); return
    st.subheader("원본 대본"); st.code(script, language="text")

    c1,c2 = st.columns(2)
    with c1:
        if st.button("🔎 상세 피드백 받기", key="btn_fb"):
            with st.spinner("🔍 피드백을 생성하고 있습니다..."):
                criteria = ("아래 7가지 기준으로, 예시는 간단히, 수정 제안은 구체적으로:\n"
                            "1) 주제 명확성  2) 이야기 전개 완결성  3) 등장인물 말투·성격 적합성\n"
                            "4) 해설·대사·지문 적합성  5) 구성 완전성  6) 독창성·재미 요소  7) 맞춤법·띄어쓰기 정확성")
                fb = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":criteria+"\n\n대본:\n"+script}],
                    temperature=0.4, max_tokens=1400
                ).choices[0].message.content
                st.session_state["script_feedback"]=fb
                st.success("✅ 피드백 생성 완료!")
                if not st.session_state.get("script_final"):
                    st.info("💡 오른쪽의 '✨ 피드백 반영하여 대본 생성하기' 버튼을 눌러보세요!")
                st.session_state["next_step_hint"] = "피드백에 맞추어 대본이 완성되면 다음 단계로 이동하세요."
    with c2:
        if st.button("✨ 피드백 반영하여 대본 생성하기", key="btn_make_final"):
            with st.spinner("✨ 대본을 생성하고 있습니다..."):
                prm = (
                    "초등학생 눈높이에 맞춰 대본을 다듬고, 필요하면 내용을 자연스럽게 보강하여 "
                    "기-승-전-결이 또렷한 **연극 완성본**을 작성하세요.\n\n"
                    "형식 규칙:\n"
                    "1) **장면 1, 장면 2, 장면 3 ...** 처럼 최소 **4장면 이상(권장 5~7장면)**으로 분할하고, 각 장면 제목은 줄 단독으로 표기.\n"
                    "2) 각 장면은 시작·전환·해결이 느껴지도록 사건을 배치하고, 장면 간 연결 문장을 짧게 넣으세요.\n"
                    "3) 대사는 `이름: 내용` 형식으로, 지문은 괄호 `( )`로만 표기. **해설/장면/무대** 같은 머릿말을 역할명으로 쓰지 마세요.\n"
                    "4) 불필요한 장면 반복 없이, **주제와 일관성**을 유지하세요.\n"
                    "5) 마지막 장면에서 갈등이 해소되고 여운이 남도록 마무리.\n\n"
                    "아래 대본을 참고해 보강/확장하세요(원문보다 장면 수를 늘려도 됩니다).\n\n"
                    f"{script}"
                )
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prm}],
                    temperature=0.6, max_tokens=2600
                ).choices[0].message.content
                st.session_state["script_final"] = res
                st.success("🎉 대본 생성 완료!")
                st.session_state["next_step_hint"] = "대본 생성 완료! 피드백을 반영하여 수정을 완료한 후 다음 단계로 이동하세요."

    st.divider()
    if st.session_state.get("script_feedback"):
        with st.expander("📄 상세 피드백", expanded=False):
            st.markdown(st.session_state["script_feedback"])
    if st.session_state.get("script_final"):
        st.subheader("🤖 AI 추천 대본 (수정 가능)")
        st.markdown("AI가 추천한 대본입니다. 상세 피드백을 참고하여 수정해보아요!")
        st.code(st.session_state["script_final"], language="text")
        edited_script = st.text_area(
            "대본 수정하기",
            value=st.session_state["script_final"],
            height=300,
            key="script_editor"
        )
        original_roles = extract_roles(st.session_state.get("script_raw", ""))
        filtered_lines = []
        for line in clean_script_text(st.session_state["script_final"]).splitlines():
            m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if m:
                character = _normalize_role(m.group(1))
                if character in original_roles:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        filtered_script = "\n".join(filtered_lines)
        st.session_state["script_final"] = filtered_script
        if st.button("✅ 수정 완료", key="btn_save_script"):
            st.session_state["script"] = edited_script
            st.success("✅ 대본이 저장되었습니다!")

# (page_stage_kits, page_rehearsal_partner, sidebar_status, main — 기존 그대로)

# ───────── 페이지 4: 소품·무대·의상 ─────────────────────────────────

def page_stage_kits():
    st.header("🎭 4) 소품·무대·의상 추천")
    st.markdown("연극에 필요한 소품을 AI가 추천해 줘요.")
    script = st.session_state.get("script_final") or st.session_state.get("script_balanced") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/생성하세요."); return
    if st.button("🧰 목록 만들기", key="btn_kits"):
        with st.spinner("🧰 소품·무대·의상 목록을 생성하고 있습니다..."):
            prm = ("다음 대본을 바탕으로 초등 연극용 소품·무대·의상 체크리스트를 만들어주세요.\n"
                   "구성: [필수/선택/대체/안전 주의] 4섹션 표(마크다운) + 간단 팁.\n\n대본:\n"+script)
            res = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prm}],
                temperature=0.4, max_tokens=1200
            ).choices[0].message.content
            st.session_state["stage_kits"] = res
            st.success("✅ 목록 생성 완료!")
            st.markdown(res or "(생성 실패)")
            st.session_state["next_step_hint"] = "체크리스트 완성! 다음 단계로 이동하세요."

# ───────── 페이지 5: AI 대본 연습 ────────────────────────────────

# (기존 page_rehearsal_partner 전체 그대로 복사 — 내용 생략 없이 동일)
# — 원본과 동일하므로 여기서는 줄임. 실제 파일에는 기존 함수 전문을 유지하세요.

# ───────── MAIN ────────────────────────────────────────────────────

def main():
    st.set_page_config("연극용의 둥지", "🐉", layout="wide")
    st.markdown(PASTEL_CSS, unsafe_allow_html=True)
    st.title("🐉 연극용의 둥지 — 연극 용을 성장시켜요!")
    st.subheader("연극 용과 함께 완성도 있는 연극을 완성하고 연습해 보자!")

    if "current_page" not in st.session_state:
        st.session_state["current_page"]="📥 1) 대본 등록"

    pages = {
        "📥 1) 대본 등록": page_script_input,
        "🛠️ 2) 대본 피드백 & 완성본": page_feedback_script,
        "⚖️ 3) 대사 수 조절하기": page_role_balancer,
        "🎭 4) 소품·무대·의상": page_stage_kits,
        # "🎙️ 5) AI 대본 연습": page_rehearsal_partner  # ← 원본과 동일하게 등록
    }

    all_pages = list(pages.keys())
    sel = st.sidebar.radio("메뉴", all_pages, 
                          index=all_pages.index(st.session_state["current_page"]), 
                          key="nav_radio")
    st.session_state["current_page"]=sel

    if st.sidebar.button("전체 초기화", key="btn_reset_all"):
        st.session_state.clear()
        if hasattr(st, "rerun"): st.rerun()

    pages[sel]()

if __name__=="__main__":
    main()
