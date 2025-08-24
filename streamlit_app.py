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
    "지민 (여성, 부드럽고 친절한 목소리)": "coral",
    "소연 (여성, 귀엽고 명랑한 목소리)": "nova",
    "하은 (여성, 차분하고 우아한 목소리)": "echo",
    "민지 (여성, 밝고 경쾌한 목소리)": "shimmer"
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
        # 목소리별 톤 보정 (나이대와 성별 고려)
        if "10대" in voice_label:
            if "여성" in voice_label:
                audio = _pitch_shift_mp3(audio, +2.0)  # 10대 여성: 약간 높게
            else:
                audio = _pitch_shift_mp3(audio, +1.0)  # 10대 남성: 살짝 높게
        elif "20대" in voice_label:
            if "여성" in voice_label:
                audio = _pitch_shift_mp3(audio, +1.5)  # 20대 여성: 적당히 높게
            else:
                audio = _pitch_shift_mp3(audio, -0.5)  # 20대 남성: 살짝 낮게
        elif "30대" in voice_label:
            if "여성" in voice_label:
                audio = _pitch_shift_mp3(audio, +1.0)  # 30대 여성: 약간 높게
            else:
                audio = _pitch_shift_mp3(audio, -1.0)  # 30대 남성: 낮게
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

def _analyze_wav_pure(audio_bytes: bytes, stt_text: str) -> dict:
    """ffmpeg/librosa 없이도 동작하는 초간단 WAV 분석(16-bit 위주)."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            ch = wf.getnchannels(); sw = wf.getsampwidth(); sr = wf.getframerate(); n = wf.getnframes()
            raw = wf.readframes(n)
        if sw not in (1,2):
            return {"speed_label":"데이터 부족","volume_label":"데이터 부족","tone_label":"데이터 부족","spacing_label":"데이터 부족",
                    "syllables_per_sec":None,"wps":None,"rms_db":None,"f0_hz":None,"f0_var":None,"pause_ratio":None}
        if sw == 1:
            fmt = f"{len(raw)}b"; maxv = 127.0; arr = struct.unpack(fmt, raw)
        else:
            fmt = f"{len(raw)//2}h"; maxv = 32767.0; arr = struct.unpack(fmt, raw)
        if ch > 1:
            arr = [(arr[i] + arr[i+1]) / 2.0 for i in range(0, len(arr), ch)]
        else:
            arr = list(arr)
        dur = len(arr)/sr
        if dur <= 0.0:
            raise RuntimeError("empty audio")
        # RMS(dBFS)
        mean_sq = sum((x/maxv)*(x/maxv) for x in arr)/len(arr)
        rms = math.sqrt(max(mean_sq, 1e-12))
        rms_db = 20.0*math.log10(rms)
        # 20ms 창 에너지 → 무음 비율
        win = int(sr*0.02) or 1
        energies = []
        for i in range(0, len(arr), win):
            w = arr[i:i+win]
            if not w: break
            e = math.sqrt(sum((x/maxv)*(x/maxv) for x in w)/len(w))
            energies.append(e)
        if energies:
            hi = sorted(energies)[int(max(0, len(energies)*0.9)-1)]
            thr = max(hi*0.1, 1e-6)
            unvoiced = sum(1 for e in energies if e < thr) * 0.02
        else:
            unvoiced = 0.0
        pause_ratio = min(1.0, max(0.0, unvoiced/max(dur,1e-6)))
        # 속도(음절/초) & 단어/초
        syllables = len([c for c in (stt_text or "") if ('가' <= c <= '힣') or c.isdigit()])
        voiced = max(dur - unvoiced, 1e-6)
        syl_rate = (syllables/voiced) if syllables>0 else None
        wps = (len((stt_text or "").split())/voiced) if stt_text else None
        def lab_speed(s):
            if s is None: return "데이터 부족"
            if s>=7.3: return "너무 빠름"
            if s>=6.6: return "빠름"
            if s>=5.2: return "적당함"
            if s>=3.8: return "느림"
            return "너무 느림"
        def lab_volume(db):
            if db is None: return "데이터 부족"
            if db>=-13: return "너무 큼"
            if db>=-23: return "큼"
            if db>=-37: return "적당함"
            if db>=-47: return "작음"
            return "너무 작음"
        spacing = ("잘 띄어 읽음" if 0.08<=pause_ratio<=0.28 else
                   "보통" if 0.04<=pause_ratio<0.08 or 0.28<pause_ratio<=0.40 else
                   "잘 띄어 읽는 것이 되지 않음")
        # 톤(간이)
        if energies:
            rng = (max(energies)-min(energies))
            if rng>0.25 and rms_db>-20: tone="화내는 어조"
            elif rng>0.18 and pause_ratio>=0.2: tone="즐거운 어조"
            elif rms_db<-35: tone="슬픈 어조"
            else: tone="담담한 어조"
        else:
            tone="담담한 어조"
        return {"speed_label":lab_speed(syl_rate),"volume_label":lab_volume(rms_db),
                "tone_label":tone,"spacing_label":spacing,
                "syllables_per_sec":syl_rate,"wps":wps,"rms_db":rms_db,
                "f0_hz":None,"f0_var":None,"pause_ratio":pause_ratio}
    except Exception:
        return {"speed_label":"데이터 부족","volume_label":"데이터 부족",
                "tone_label":"데이터 부족","spacing_label":"데이터 부족",
                "syllables_per_sec":None,"wps":None,"rms_db":None,"f0_hz":None,
                "f0_var":None,"pause_ratio":None}


def analyze_prosody(audio_bytes: bytes, stt_text: str) -> dict:
    """
    1) librosa(+numpy) 사용 시 정밀 분석
    2) 실패하면 pydub 경량 분석
    3) 그래도 실패하면 순수 WAV 폴백(_analyze_wav_pure)
    """
    # 1) librosa 경로
    if _lb is not None and _np is not None:
        try:
            y, sr = _lb.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            if y is None or (hasattr(y, "size") and y.size == 0):
                raise RuntimeError("empty audio")
            # 발성 길이
            if _vad:
                int16 = (y * 32767).astype("int16").tobytes()
                v = _vad.Vad(2); frame_ms = 20
                step = int(sr * frame_ms / 1000)
                frames = [int16[i:i+2*step] for i in range(0, len(int16), 2*step)]
                voiced = []; cur=None; t=0.0
                for f in frames:
                    isv = v.is_speech(f, sr)
                    if isv and cur is None: cur=[t,None]
                    if (not isv) and cur is not None: cur[1]=t; voiced.append(cur); cur=None
                    t += frame_ms/1000.0
                if cur is not None: cur[1]=t; voiced.append(cur)
                voiced_total = sum([e-s for s,e in voiced])
            else:
                intervals = _lb.effects.split(y, top_db=35)
                voiced_total = sum([(e - s)/sr for s, e in intervals]) if intervals else len(y)/sr
            total = len(y)/sr
            voiced_total = voiced_total if voiced_total > 0 else total
            syllables = len(re.findall(r"[가-힣]", stt_text or ""))
            syl_rate = syllables/voiced_total if voiced_total>0 else None
            words = len((stt_text or "").split()); wps = words/voiced_total if voiced_total>0 else None
            if syl_rate is None: speed = "데이터 부족"
            else:
                speed = ("너무 빠름" if syl_rate>=7.3 else
                         "빠름"      if syl_rate>=6.6 else
                         "적당함"    if syl_rate>=5.2 else
                         "느림"      if syl_rate>=3.8 else "너무 느림")
            rms = float((_np.sqrt(_np.mean(y*y))) + 1e-12)
            rms_db = 20.0 * math.log10(rms)
            volume = ("너무 큼" if rms_db>=-13 else
                      "큼"     if rms_db>=-23 else
                      "적당함" if rms_db>=-37 else
                      "작음"   if rms_db>=-47 else "너무 작음")
            try:
                f0, _, _ = _lb.pyin(y, fmin=75, fmax=500, sr=sr, frame_length=2048, hop_length=256)
                if f0 is not None:
                    f0_valid = f0[_np.isfinite(f0)]
                    if f0_valid.size>0:
                        f0_med = float(_np.nanmedian(f0_valid))
                        f0_std = float(_np.nanstd(f0_valid))
                        pitch_desc = ("낮음" if f0_med<140 else "중간" if f0_med<200 else "높음")
                        var_desc   = ("변화 적음" if f0_std<15 else "변화 적당" if f0_std<35 else "변화 큼")
                        if pitch_desc=="높음" and var_desc!="변화 적음": tone="활기찬/즐거운 어조"
                        elif pitch_desc=="낮음" and var_desc=="변화 적음": tone="담담·낮은 톤"
                        elif var_desc=="변화 큼": tone="감정 기복 큰 어조"
                        else: tone="담담한 어조"
                    else:
                        f0_med, f0_std, tone = None, None, "담담한 어조"
                else:
                    f0_med, f0_std, tone = None, None, "담담한 어조"
            except Exception:
                f0_med, f0_std, tone = None, None, "담담한 어조"
            unvoiced = max(0.0, total - voiced_total)
            pause_ratio = unvoiced/total if total>0 else 0.0
            spacing = ("잘 띄어 읽음" if 0.08<=pause_ratio<=0.28 else
                       "보통" if 0.04<=pause_ratio<0.08 or 0.28<pause_ratio<=0.40 else
                       "잘 띄어 읽는 것이 되지 않음")
            return {"speed_label":speed,"volume_label":volume,"tone_label":tone,"spacing_label":spacing,
                    "syllables_per_sec":syl_rate,"wps":wps,"rms_db":rms_db,
                    "f0_hz":f0_med,"f0_var":f0_std,"pause_ratio":pause_ratio}
        except Exception:
            pass
    # 2) pydub 경로
    if AudioSegment is not None:
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            dur = max(0.001, seg.duration_seconds)
            rms_dbfs = seg.dBFS if seg.dBFS != float("-inf") else -60.0
            volume = ("너무 큼" if rms_dbfs>-13 else
                      "큼"     if rms_dbfs>-23 else
                      "적당함" if rms_dbfs>-37 else
                      "작음"   if rms_dbfs>-47 else "너무 작음")
            if _silence:
                non = _silence.detect_nonsilent(seg, min_silence_len=120,
                                                silence_thresh=max(-60, int(seg.dBFS)-10))
                voiced_total = sum((b-a) for a,b in non)/1000.0 if non else dur
            else:
                voiced_total = dur
            unvoiced = max(0.0, dur - voiced_total)
            pause_ratio = unvoiced/dur if dur>0 else 0.0
            spacing = ("잘 띄어 읽음" if 0.08<=pause_ratio<=0.28 else
                       "보통" if 0.04<=pause_ratio<0.08 or 0.28<pause_ratio<=0.40 else
                       "잘 띄어 읽는 것이 되지 않음")
            syllables = len(re.findall(r"[가-힣]", stt_text or ""))
            syl_rate = (syllables/voiced_total) if (voiced_total>0 and syllables>0) else None
            if syl_rate is None: speed = "데이터 부족"
            else:
                speed = ("너무 빠름" if syl_rate>=7.3 else
                         "빠름"      if syl_rate>=6.6 else
                         "적당함"    if syl_rate>=5.2 else
                         "느림"      if syl_rate>=3.8 else "너무 느림")
            wps = (len((stt_text or '').split())/voiced_total) if (voiced_total>0 and stt_text) else None
            step=50; vals=[]
            for i in range(0, len(seg), step):
                v = seg[i:i+step].dBFS
                vals.append(-60.0 if v==float("-inf") else v)
            rng = (max(vals)-min(vals)) if vals else 0.0
            if rng>20 and rms_dbfs>-20: tone="화내는 어조"
            elif rng>15 and pause_ratio>=0.2: tone="즐거운 어조"
            elif rms_dbfs<-35: tone="슬픈 어조"
            else: tone="담담한 어조"
            return {"speed_label":speed,"volume_label":volume,"tone_label":tone,"spacing_label":spacing,
                    "syllables_per_sec":syl_rate,"wps":wps,"rms_db":rms_dbfs,
                    "f0_hz":None,"f0_var":None,"pause_ratio":pause_ratio}
        except Exception:
            pass
    # 3) 순수 WAV 폴백
    return _analyze_wav_pure(audio_bytes, stt_text)


def _badge(label: str) -> str:
    if label in ("적당함","잘 띄어 읽음") or "활기찬" in label:
        cls="b-ok"
    elif label in ("빠름","느림","보통"):
        cls="b-warn"
    else:
        cls="b-bad"
    return f"<span class='badge {cls}'>{label}</span>"

def _gauge_html(pct: float) -> str:
    pct = max(0,min(100,int(pct)))
    return f"<div class='gauge'><span style='width:{pct}%'></span></div>"

def _score_speed(sps: Optional[float]) -> int:
    if sps is None or sps<=0: return 0
    center=3.3; spread=2.2
    dist=abs(sps-center)/spread
    return max(0,int(100*(1-dist)))

def _score_volume(rms_db: Optional[float]) -> int:
    if rms_db is None: return 0
    center=-24.0; spread=12.0
    dist=abs(rms_db-center)/spread
    return max(0,int(100*(1-dist)))

def _score_spacing(pause_ratio: Optional[float]) -> int:
    if pause_ratio is None: return 0
    center=0.18; spread=0.18
    dist=abs(pause_ratio-center)/spread
    return max(0,int(100*(1-dist)))

def render_prosody_card(pros: dict):
    sp = pros.get("speed_label","데이터 부족")
    vo = pros.get("volume_label","데이터 부족")
    to = pros.get("tone_label","데이터 부족")
    spc= pros.get("spacing_label","데이터 부족")
    sps = pros.get("syllables_per_sec")
    voldb = pros.get("rms_db")
    pr   = pros.get("pause_ratio")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><h4>🗣️ 말속도</h4>"+_badge(sp)+
                    f"<div class='kv'><div class='k'>음절/초</div><div class='v'>{(sps or 0):.2f}</div></div>"+
                    _gauge_html(_score_speed(sps))+"</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h4>🔊 목소리 크기</h4>"+_badge(vo)+
                    f"<div class='kv'><div class='k'>RMS(dBFS)</div><div class='v'>{(voldb if voldb is not None else 0):.1f}</div></div>"+
                    _gauge_html(_score_volume(voldb))+"</div>", unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        st.markdown("<div class='card'><h4>🎭 어조(피치)</h4>"+_badge(to)+
                    f"<div class='kv'><div class='k'>F0(Hz)</div><div class='v'>{(int(pros.get('f0_hz')) if pros.get('f0_hz') else '—')}</div></div>"+
                    "</div>", unsafe_allow_html=True)

# ───────── 페이지 1: 대본 업로드/입력 ──────────────────────────────
def page_script_input():
    # 용 소개 이미지 추가
    st.image("assets/dragon_intro.png", width=400, use_container_width =True)
    st.header("📥 1) 대본 업로드/입력")
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
        if st.button("💾 저장", key="btn_save_script"):
            st.session_state["script_raw"] = val.strip(); st.success("저장되었습니다. 왼쪽 메뉴에서 다음 페이지로 이동해주세요!")

# ───────── 페이지 2: 대본 피드백 & 완성본 생성 ─────────────────────
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
                            "1) 주제 일관성  2) 기승전결 자연스러움  3) 인물 말투/성격 적합성\n"
                            "4) 대사·지문 적합성  5) 구성 완전성  6) 독창성/재미  7) 맞춤법·띄어쓰기")
                fb = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":criteria+"\n\n대본:\n"+script}],
                    temperature=0.4, max_tokens=1400
                ).choices[0].message.content
                st.session_state["script_feedback"]=fb
                st.success("✅ 피드백 생성 완료!")
                
                # 오른쪽으로 이동 안내
                if not st.session_state.get("script_final"):
                    st.info("💡 오른쪽의 '✨ 피드백 반영하여 대본 생성하기' 버튼을 눌러보세요!")
                    
                # 왼쪽 메뉴 다음 단계 안내
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
                
                # 왼쪽 메뉴 다음 단계 안내
                st.session_state["next_step_hint"] = "대본 생성 완료! 피드백을 반영하여 수정을 완료한 후 다음 단계로 이동하세요."

    st.divider()
    if st.session_state.get("script_feedback"):
        with st.expander("📄 상세 피드백", expanded=False):
            st.markdown(st.session_state["script_feedback"])
    if st.session_state.get("script_final"):
        st.subheader("🤖 AI 추천 대본 (수정 가능)")
        st.markdown("AI가 추천한 대본입니다. 상세 피드백을 참고하여 수정해보아요!")
        
        # 수정 가능한 텍스트박스
        edited_script = st.text_area(
            "대본 수정",
            value=st.session_state["script_final"],
            height=300,
            key="script_editor"
        )
        
        # 수정 완료 버튼
        if st.button("✅ 수정 완료", key="btn_save_script"):
            st.session_state["script"] = edited_script
            st.success("✅ 대본이 저장되었습니다!")

# ───────── 페이지 3: 역할 밸런서 ───────────────────────────────────
# ==== 역할 대사 생성 헬퍼 ==========================================
def _strip_role_prefix(text: str, role: str) -> str:
    if not text: return ""
    m = re.match(rf"^\s*{re.escape(role)}\s*[:：]\s*(.+)$", text.strip())
    return (m.group(1).strip() if m else text.strip())

def _gen_contextual_line(role: str, seq: List[Dict], anchor_idx: int) -> str:
    """
    새로 끼워 넣을 위치(anchor_idx) 주변 문맥을 보고, role에 맞는 한 줄 대사를 생성.
    실패하면 짧은 기본 문장으로 폴백.
    """
    try:
        ctx_lo = max(0, anchor_idx - 6)
        ctx_hi = min(len(seq), anchor_idx + 6)
        ctx_text = "\n".join(f"{ln['who']}: {ln['text']}" for ln in seq[ctx_lo:ctx_hi])

        # OpenAI 프롬프트 (초등 수준, 1줄, 괄호 지문 허용)
        prm = (
            "아래 연극 대본의 문맥을 자연스럽게 이어서, 지정된 인물의 대사를 한국어 한 줄로 만들어줘.\n"
            "제약:\n"
            f"- 인물 이름: '{role}'\n"
            "- 출력은 반드시 한 줄, 형식은 '이름: 대사'.\n"
            "- 초등학생 눈높이(친절/간결/품위). 과도한 갈등/폭력/비속어 금지.\n"
            "- 필요할 경우 ( ) 안에 아주 짧은 지문 1개까지 허용.\n"
            "- 이미 바로 앞에서 같은 말 반복하지 말고, 대화가 앞으로 나아가게.\n\n"
            "문맥:\n" + ctx_text
        )

        if not OPENAI_API_KEY:
            raise RuntimeError("no api key")

        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prm}],
            temperature=0.6,
            max_tokens=90,
        ).choices[0].message.content.strip()

        # '이름: 내용' 보장/정리
        line = out.splitlines()[0].strip()
        # 혹시 모델이 이름 없이 준 경우 보정
        if not re.match(rf"^\s*{re.escape(role)}\s*[:：]", line):
            line = f"{role}: " + _strip_role_prefix(line, role)
        # 본문만 리턴 (페이지 로직은 who/text로 나눠 쓰니까)
        return _strip_role_prefix(line, role) or "(미소 지으며) 좋아, 한 번 해보자!"
    except Exception:
        # 폴백 문장(문맥 실패/오프라인/키 누락 시)
        return "(고개를 끄덕이며) 좋아, 그 방법이 괜찮겠어!"
    
def page_role_balancer():
    st.header("⚖️ 3) 역할 밸런서(대사 수 조절)")
    # 스크립트 우선순위: 현재 스크립트 > 재분배된 것 > 최종 스크립트 > 원본 스크립트
    script = st.session_state.get("current_script") or st.session_state.get("script_balanced") or st.session_state.get("script_final") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/생성하세요."); return
    seq = build_sequence(script); roles = extract_roles(script)
    if not roles: st.info("‘이름: 내용’ 형식이어야 역할을 인식해요."); return

    # 현재 대사 수 계산 (재분배된 스크립트가 있으면 그것을 우선 사용)
    if st.session_state.get("script_balanced"):
        # 재분배된 스크립트로 계산
        balanced_seq = build_sequence(st.session_state["script_balanced"])
        counts = {r:0 for r in roles}
        for ln in balanced_seq: counts[ln["who"]]+=1
        st.subheader("현재 대사 수 (재분배 후)")
        # counts를 session_state에 저장하여 다음 재분배 시 사용
        st.session_state["current_counts"] = counts
    else:
        # 원본 스크립트로 계산
        counts = {r:0 for r in roles}
        for ln in seq: counts[ln["who"]]+=1
        st.subheader("현재 대사 수")
        # counts를 session_state에 저장
        st.session_state["current_counts"] = counts
    
    st.write(counts)
    st.markdown("생성된 대본의 줄 수를 알려줘요.")

    st.subheader("목표 대사 수 설정")
    st.markdown("대사의 양을 골고루 분배해 보아요.")
    targets={}
    cols = st.columns(min(4, max(1,len(roles))))
    for i,r in enumerate(roles):
        with cols[i % len(cols)]:
            # 재분배 후에는 업데이트된 counts를 기본값으로 사용
            targets[r]=st.number_input(f"{r} 목표", min_value=0, value=counts[r], step=1, key=f"tgt_{r}")

    if st.button("🔁 재분배하기 (더블 클릭)", key="btn_rebalance"):
        with st.spinner("⚖️ 역할을 재분배하고 있습니다..."):

        # ── 준비: 현재 카운트/목표/과잉·부족 계산(기존코드 그대로)
         current_counts = st.session_state.get("current_counts", counts)
        need   = {r: max(0, targets[r] - current_counts[r]) for r in roles}
        excess = {r: max(0, current_counts[r] - targets[r]) for r in roles}
        skip_step = {r: (current_counts[r] // excess[r] if excess[r] > 0 else None) for r in roles}
        seen = {r: 0 for r in roles}

        # ── 0) 재분배할 원본 시퀀스 확보
        script_to_rebalance = st.session_state.get("script_balanced") or script
        seq_to_rebalance = build_sequence(script_to_rebalance)

        # ── 1) 과잉 대사 드롭하여 base 시퀀스 만들기 → new_seq **여기서 생성**
        new_seq = []
        for ln in seq_to_rebalance:
            r = ln["who"]
            seen[r] += 1
            if excess[r] > 0 and skip_step[r] and (seen[r] % max(1, skip_step[r]) == 0) and current_counts[r] > targets[r]:
                current_counts[r] -= 1  # 이 줄은 버림
                continue
            new_seq.append(ln)

        # ── 2) 부족한 역할 대사 채우기(문맥 기반 1줄 생성 + 균등 분산 삽입)
        for r in roles:
            k = int(need[r])
            if k <= 0:
                continue

            # new_seq가 비어있으면 안전 가드: 그냥 k줄 append
            if len(new_seq) == 0:
                for _ in range(k):
                    content = _gen_contextual_line(r, new_seq, 0)
                    new_seq.append({"who": r, "text": content})
                continue

            # (a) 삽입 지점 계산: 전체를 k+1로 등분해서 각 구간 끝쪽에 한 줄
            step = max(1, len(new_seq) // (k + 1))
            anchors = []
            for i in range(k):
                pos = step * (i + 1) - 1
                anchors.append(min(len(new_seq) - 1, max(0, pos)))

            # (b) 실제 삽입 (앞에서부터 넣으면 인덱스 밀려서 shift 보정)
            shift = 0
            for anchor in anchors:
                anchor_idx = min(len(new_seq) - 1, max(0, anchor + shift))
                content = _gen_contextual_line(r, new_seq, anchor_idx)  # 문맥 기반 한 줄 생성
                insert_at = min(len(new_seq), anchor_idx + 1)
                new_seq.insert(insert_at, {"who": r, "text": content})
                shift += 1

        # ── 3) 저장/미리보기 반영
        st.session_state["script_balanced"] = "\n".join([f"{x['who']}: {x['text']}" for x in new_seq])
        st.session_state["current_script"] = st.session_state["script_balanced"]
        st.success("✅ 재분배 완료!")

            
            # 재분배된 스크립트를 현재 script로 설정하여 즉시 반영
        st.session_state["current_script"] = st.session_state["script_balanced"]
            
            # 왼쪽 메뉴 다음 단계 안내
        st.session_state["next_step_hint"] = "역할 재분배 완료! 다음 단계로 이동하세요."

    if st.session_state.get("script_balanced"):
        st.subheader("재분배 결과 미리보기")
        st.code(st.session_state["script_balanced"], language="text")

# ───────── 페이지 4: 소품·무대·의상 ─────────────────────────────────
def page_stage_kits():
    st.header("🎭 4) 소품·무대·의상 추천")
    st.markdown("연극에 필요한 소품을 AI가 추천해줘요.")
    script = st.session_state.get("script_final") or st.session_state.get("script_balanced") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/생성하세요."); return
    if st.button("🧰 체크리스트 만들기", key="btn_kits"):
        with st.spinner("🧰 소품·무대·의상 체크리스트를 생성하고 있습니다..."):
            prm = ("다음 대본을 바탕으로 초등 연극용 소품·무대·의상 체크리스트를 만들어주세요.\n"
                   "구성: [필수/선택/대체/안전 주의] 4섹션 표(마크다운) + 간단 팁.\n\n대본:\n"+script)
            res = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prm}],
                temperature=0.4, max_tokens=1200
            ).choices[0].message.content
            st.session_state["stage_kits"] = res
            st.success("✅ 체크리스트 생성 완료!")
            st.markdown(res or "(생성 실패)")
            
            # 왼쪽 메뉴 다음 단계 안내
            st.session_state["next_step_hint"] = "체크리스트 완성! 다음 단계로 이동하세요."

# ───────── 페이지 5: 리허설 파트너 ────────────────────────────────
def page_rehearsal_partner():
    st.header("🎙️ 5) 리허설 파트너 — 줄 단위 STT(REST, 한 번 클릭→자동 분석)")

    script = st.session_state.get("script_final") or st.session_state.get("script_balanced") or st.session_state.get("script_raw","")
    if not script:
        st.warning("먼저 대본을 등록/생성하세요."); return

    seq = build_sequence(script); roles = extract_roles(script)
    if not seq or not roles:
        st.info("‘이름: 내용’ 형식이어야 리허설 가능해요."); return

    # 상태키
    st.session_state.setdefault("duet_cursor", 0)
    st.session_state.setdefault("duet_turns", [])
    st.session_state.setdefault("auto_done_token", None)

    # 텍스트 분석 포함 여부(기본: 포함)
    want_metrics = st.checkbox("텍스트 분석 포함(말속도·크기·어조·띄어읽기)", value=True, key="ck_metrics")

    # 파트너 TTS 보이스
    st.markdown("**🎭 상대역 음성 선택**")
    st.markdown("연극에 적합한 목소리를 선택해보세요!")
    
    voice_label = st.selectbox(
        "상대역 음성 선택", 
        VOICE_KR_LABELS_SAFE, 
        index=0, 
        key="tts_voice_label_safe",
        help="각 목소리는 나이대와 특성이 표시되어 있어요. 역할에 맞는 목소리를 선택해보세요!"
    )
    
    # 선택된 목소리 정보 표시
    if voice_label:
        # 목소리별 특성 설명
        voice_descriptions = {
            "민준 (남성, 따뜻하고 친근한 목소리)": "🎭 **따뜻하고 친근한 목소리** - 선생님이나 부모님 역할에 적합해요!",
            "현우 (남성, 차분하고 신뢰감 있는 목소리)": "🎭 **차분하고 신뢰감 있는 목소리** - 의사나 경찰관 같은 전문직 역할에 좋아요!",
            "지호 (남성, 활기차고 밝은 목소리)": "🎭 **활기차고 밝은 목소리** - 친구나 동생 역할에 어울려요!",
            "지민 (여성, 부드럽고 친절한 목소리)": "🎭 **부드럽고 친절한 목소리** - 친절한 선생님이나 언니 역할에 어울려요!",
            "소연 (여성, 귀엽고 명랑한 목소리)": "🎭 **귀엽고 명랑한 목소리** - 귀여운 친구나 동생 역할에 최고예요!",
            "하은 (여성, 차분하고 우아한 목소리)": "🎭 **차분하고 우아한 목소리** - 우아한 공주나 여왕 역할에 어울려요!",
            "민지 (여성, 밝고 경쾌한 목소리)": "🎭 **밝고 경쾌한 목소리** - 활발한 친구나 운동선수 역할에 어울려요!"
        }
        
        st.success(f"✅ **선택된 목소리**: {voice_label}")
        st.info(voice_descriptions.get(voice_label, "멋진 목소리네요!"))

    # 이동 컨트롤
    # c1, c2, c3 = st.columns(3)
    # with c1:
    #     if st.button("⏮️ 처음부터 다시", key="restart_all"):
    #         st.session_state["duet_cursor"]=0
    #         st.session_state["duet_turns"]=[]; st.session_state["auto_done_token"]=None
    #         if hasattr(st, "rerun"): st.rerun()
    # with c2:
    #     if st.button("⬅️ 이전 줄로 이동", key="prev_line"):
    #         st.session_state["duet_cursor"]=max(0, st.session_state["duet_cursor"]-1)
    #         st.session_state["auto_done_token"]=None
    #         if hasattr(st, "rerun"): st.rerun()
    # with c3:
    #     if st.button("➡️ 다음 줄 이동", key="next_line"):
    #         st.session_state["duet_cursor"]=min(len(seq), st.session_state["duet_cursor"]+1)
    #         st.session_state["auto_done_token"]=None
    #         if hasattr(st, "rerun"): st.rerun()

    # 내 역할
    my_role = st.selectbox("내 역할(실시간)", roles, key="role_live")
    
    # 역할이 변경되면 자동으로 페이지 새로고침
    if "previous_role" not in st.session_state:
        st.session_state["previous_role"] = my_role
    elif st.session_state["previous_role"] != my_role:
        st.session_state["previous_role"] = my_role
        st.success(f"✅ 역할이 '{my_role}'로 변경되었습니다!")
        st.rerun()

    # 현재 줄
    cur_idx = st.session_state.get("duet_cursor", 0)
    if cur_idx >= len(seq):
        st.success("🎉 끝까지 진행했습니다. 이제 세션 종료 & 종합 피드백을 받아보세요!")
        st.info("💡 아래의 '🏁 세션 종료 & 종합 피드백' 버튼을 눌러 연습 결과를 확인해보세요!")
    else:
        cur_line = seq[cur_idx]
        st.markdown(f"#### 현재 줄 #{cur_idx+1}: **{cur_line['who']}** — {cur_line['text']}")

        # (내 차례)
        if cur_line["who"] == my_role:
            st.info("내 차례예요. 아래 **마이크 버튼을 한 번만** 눌러 말하고, 버튼이 다시 바뀌면 자동 분석이 시작됩니다.")
            audio_bytes = None
            if audio_recorder is not None:
                # 녹음 상태에 따른 텍스트 변경
                #recording_status = "🎙️ 녹음중" if st.session_state.get(f"recording_{cur_idx}", False) else "🎤 녹음 준비"
                #st.markdown(f"**{recording_status}**")
                st.markdown("💡 **마이크 아이콘을 클릭하여 녹음 시작/중지**")
                
                audio_bytes = audio_recorder(
                    text="🎤 말하고 인식(자동 분석)", sample_rate=16000,
                    pause_threshold=2.0, key=f"audrec_one_{cur_idx}"
                )
            else:
                st.warning("audio-recorder-streamlit 패키지가 필요합니다. `pip install audio-recorder-streamlit`")

            if audio_bytes:
                # 상태머신: 인식중 → 분석중 → 완료
                with st.status("🎧 인식 중...", expanded=True) as s:
                    token = hashlib.sha256(audio_bytes).hexdigest()[:16]
                    if st.session_state.get("auto_done_token") != (cur_idx, token):
                        st.session_state["auto_done_token"] = (cur_idx, token)

                        stt = clova_short_stt(audio_bytes, lang="Kor")
                        s.update(label="🧪 분석 중...", state="running")

                        st.markdown("**STT 인식 결과(원문)**")
                        st.text_area("인식된 문장", value=stt or "(빈 문자열)", height=90, key=f"saw_{cur_idx}")

                        expected_core = re.sub(r"\(.*?\)", "", cur_line["text"]).strip()
                        st.caption("비교 기준(지문 제거)")
                        st.code(expected_core, language="text")

                        html, _ = match_highlight_html(expected_core, stt or "")
                        score = similarity_score(expected_core, stt or "")
                        st.markdown("**일치 하이라이트(초록=일치, 빨강=누락)**", unsafe_allow_html=True)
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption(f"일치율(내부 지표) 약 {score*100:.0f}%")

                        if want_metrics:
                            pros = analyze_prosody(audio_bytes, stt or "")
                            render_prosody_card(pros)
                        else:
                            st.info("텍스트만 확인 모드입니다.")

                        turns = st.session_state.get("duet_turns", [])
                        turns.append({
                            "line_idx": cur_idx+1, "who": cur_line["who"],
                            "expected": expected_core, "spoken": stt, "score": score
                        })
                        st.session_state["duet_turns"] = turns
                        s.update(label="✅ 인식 완료", state="complete")

            cA, cB = st.columns(2)
            with cA:
                if st.button("⬅️ 이전 줄로 이동", key=f"prev_live_{cur_idx}"):
                    st.session_state["duet_cursor"] = max(0, cur_idx-1)
                    st.session_state["auto_done_token"]=None
                    if hasattr(st, "rerun"): st.rerun()
            with cB:
                if st.button("➡️ 다음 줄 이동", key=f"next_live_{cur_idx}"):
                    st.session_state["duet_cursor"] = min(len(seq), cur_idx+1)
                    st.session_state["auto_done_token"]=None
                    if hasattr(st, "rerun"): st.rerun()

    # (상대역 차례)
        else:
            st.info("지금은 상대역 차례예요. ‘🔊 파트너 말하기(수동)’으로 듣거나, 이전/다음 줄로 이동할 수 있어요.")
            if st.button("🔊 파트너 말하기(수동)", key=f"partner_say_live_cur_{cur_idx}"):
                with st.spinner("🔊 음성 합성 중…"):
                    speak_text, audio = tts_speak_line(cur_line["text"], voice_label)
                    st.success(f"파트너({cur_line['who']}): {speak_text}")
                    if audio: st.audio(audio, format="audio/mpeg")

            cA, cB = st.columns(2)
            with cA:
                if st.button("⬅️ 이전 줄로 이동", key=f"prev_live2_{cur_idx}"):
                    st.session_state["duet_cursor"] = max(0, cur_idx-1)
                    st.session_state["auto_done_token"]=None
                    if hasattr(st, "rerun"): st.rerun()
            with cB:
                if st.button("➡️ 다음 줄 이동", key=f"next_live2_{cur_idx}"):
                    st.session_state["duet_cursor"] = min(len(seq), cur_idx+1)
                    st.session_state["auto_done_token"]=None
                    if hasattr(st, "rerun"): st.rerun()

    # 세션 종합 피드백
    if st.button("🏁 세션 종료 & 종합 피드백", key="end_feedback"):
        with st.spinner("🏁 종합 피드백을 생성하고 있습니다..."):
            feed = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt_session_feedback(st.session_state.get('duet_turns',[]))}],
                temperature=0.3, max_tokens=1200
            ).choices[0].message.content
            st.session_state["session_feedback"] = feed
            st.success("✅ 종합 피드백 생성 완료!")
            st.markdown(feed or "(피드백 실패)")
            
            # 모험 마무리 안내
            st.balloons()
            st.markdown("""
            🎉 **축하합니다! 연극 연습을 성공적으로 마쳤습니다!**

            """)
            
            # 용 완성 이미지와 대사 추가
            st.image("assets/dragon_end.png", width=400, use_container_width =True)
            st.markdown("""
            🐉 **이제 연극 용이 모두 성장했어요!** 
            
            다시 돌아가서 연극 대모험을 완료해보세요! 🎭✨
            """)
            
            # 왼쪽 메뉴 초기화 안내
            st.session_state["next_step_hint"] = "🎉 연극 연습 완료! 새로운 모험을 시작해보세요!"

# ───────── 사이드바 상태 ─────────────────────────────────────────
def sidebar_status():
    st.sidebar.markdown("### 상태")
    def badge(ok: bool): return f"{'✅' if ok else '⚠️'}"
    has_script = bool(st.session_state.get("script_raw") or st.session_state.get("script_final") or st.session_state.get("script_balanced"))
    st.sidebar.markdown(f"- 대본 입력: {badge(has_script)}")
    st.sidebar.markdown(f"- OpenAI TTS: {badge(bool(OPENAI_API_KEY))}")
    st.sidebar.markdown(f"- CLOVA STT: {badge(bool(CLOVA_SPEECH_SECRET))}")
    st.sidebar.markdown(f"- OCR(선택): {badge(bool(NAVER_CLOVA_OCR_URL and NAVER_OCR_SECRET))}")
    
    # 다음 단계 안내
    if st.session_state.get("next_step_hint"):
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        st.sidebar.markdown("### 💡 다음 단계")
        st.sidebar.info(st.session_state["next_step_hint"])
        
        # 안내 메시지 초기화 버튼
        if st.sidebar.button("✅ 확인했어요", key="clear_hint"):
            del st.session_state["next_step_hint"]
            st.rerun()
    
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='small'>지문(괄호)은 TTS에서 읽지 않도록 처리됩니다.</div>", unsafe_allow_html=True)

# ───────── MAIN ────────────────────────────────────────────────────
def main():
    st.set_page_config("연극용의 둥지", "🐉", layout="wide")
    st.markdown(PASTEL_CSS, unsafe_allow_html=True)
    st.title("🐉 연극용의 둥지 — 연극 용을 성장시켜요!")
    st.subheader("연극 용과 함께 완성도 있는 연극을 완성하고 연습해보자!")

    if "current_page" not in st.session_state:
        st.session_state["current_page"]="📥 1) 대본 업로드/입력"

    pages = {
        "📥 1) 대본 업로드/입력": page_script_input,
        "🛠️ 2) 대본 피드백 & 완성본": page_feedback_script,
        "⚖️ 3) 역할 밸런서": page_role_balancer,
        "🎭 4) 소품·무대·의상": page_stage_kits,
        "🎙️ 5) 리허설 파트너": page_rehearsal_partner
    }

    #sidebar_status()
    sel = st.sidebar.radio("메뉴", list(pages.keys()), index=list(pages).index(st.session_state["current_page"]), key="nav_radio")
    st.session_state["current_page"]=sel

    if st.sidebar.button("전체 초기화", key="btn_reset_all"):
        st.session_state.clear()
        if hasattr(st, "rerun"): st.rerun()

    pages[sel]()

if __name__=="__main__":
    main()

