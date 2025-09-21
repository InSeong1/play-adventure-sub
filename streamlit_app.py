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
        if _is_banned_role(who) or who=="": continue
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
        "/System/Library/Fonts\\AppleGothic.ttf", "/Library/Fonts/AppleGothic.ttf",
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
        # 톤(개선된 판단 기준)
        if energies:
            rng = (max(energies)-min(energies))
            # 더 정확한 어조 판단을 위한 다중 기준
            if rng>0.25 and rms_db>-20 and pause_ratio<0.15: tone="화내는 어조"
            elif rng>0.18 and pause_ratio>=0.2 and rms_db>-30: tone="즐거운 어조"
            elif rms_db<-35 and pause_ratio>0.25: tone="슬픈 어조"
            elif rng<0.1 and pause_ratio<0.1: tone="담담한 어조"
            else: tone="보통 어조"  # 애매한 경우를 위한 중간 카테고리
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
                speed = ("너무 빠름" if syl_rate>=5.0 else
                         "빠름"      if syl_rate>=4.0 else
                         "적당함"    if syl_rate>=2.0 else
                         "느림"      if syl_rate>=1.2 else "너무 느림")
            rms = float((_np.sqrt(_np.mean(y*y))) + 1e-12)
            rms_db = 20.0 * math.log10(rms)
            volume = ("너무 큼" if rms_db>=-9 else
                      "큼"     if rms_db>=-15 else
                      "적당함" if rms_db>=-25 else
                      "작음"   if rms_db>=-35 else "너무 작음")
            try:
                f0, _, _ = _lb.pyin(y, fmin=75, fmax=500, sr=sr, frame_length=2048, hop_length=256)
                if f0 is not None:
                    f0_valid = f0[_np.isfinite(f0)]
                    if f0_valid.size>0:
                        f0_med = float(_np.nanmedian(f0_valid))
                        f0_std = float(_np.nanstd(f0_valid))
                        pitch_desc = ("낮음" if f0_med<140 else "중간" if f0_med<200 else "높음")
                        var_desc   = ("변화 적음" if f0_std<15 else "변화 적당" if f0_std<35 else "변화 큼")
                        # 더 정확한 어조 판단을 위한 다중 기준
                        if pitch_desc=="높음" and var_desc!="변화 적음" and pause_ratio>=0.15: tone="활기찬/즐거운 어조"
                        elif pitch_desc=="낮음" and var_desc=="변화 적음" and pause_ratio<0.1: tone="담담·낮은 톤"
                        elif var_desc=="변화 큼" and rms_db>-25: tone="감정 기복 큰 어조"
                        elif pitch_desc=="중간" and var_desc=="변화 적당": tone="보통 어조"
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
            volume = ("너무 큼" if rms_dbfs>-9 else
                      "큼"     if rms_dbfs>-15 else
                      "적당함" if rms_dbfs>-25 else
                      "작음"   if rms_dbfs>-35 else "너무 작음")
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
                speed = ("너무 빠름" if syl_rate>=5.0 else
                         "빠름"      if syl_rate>=4.0 else
                         "적당함"    if syl_rate>=2.0 else
                         "느림"      if syl_rate>=1.2 else "너무 느림")
            wps = (len((stt_text or '').split())/voiced_total) if (voiced_total>0 and stt_text) else None
            step=50; vals=[]
            for i in range(0, len(seg), step):
                v = seg[i:i+step].dBFS
                vals.append(-60.0 if v==float("-inf") else v)
            rng = (max(vals)-min(vals)) if vals else 0.0
            # 더 정확한 어조 판단을 위한 다중 기준
            if rng>20 and rms_dbfs>-20 and pause_ratio<0.15: tone="화내는 어조"
            elif rng>15 and pause_ratio>=0.2 and rms_dbfs>-30: tone="즐거운 어조"
            elif rms_dbfs<-35 and pause_ratio>0.25: tone="슬픈 어조"
            elif rng<10 and pause_ratio<0.1: tone="담담한 어조"
            else: tone="보통 어조"  # 애매한 경우를 위한 중간 카테고리
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
    center=3.0; spread=2.0
    dist=abs(sps-center)/spread
    return max(0,int(100*(1-dist)))

def _score_volume(rms_db: Optional[float]) -> int:
    if rms_db is None: return 0
    center=-20.0; spread=10.0
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
                    "<div style='font-size: 0.8rem; color: #666; margin-top: 8px;'>" +
                    "💡 <strong>참고:</strong> 어조는 목소리의 높낮이와 변화로 판단해요. " +
                    "하지만 실제 감정과 다를 수 있으니 참고만 해주세요! 😊" +
                    "</div>" +
                    "</div>", unsafe_allow_html=True)

# ───────── [추가] 줄 수/내용 검증 & 비상 후처리 헬퍼 ─────────────────
def count_dialogue_lines(script: str, roles: List[str]) -> Dict[str, int]:
    """역할별 대사 줄 수를 센다(지문 제외)."""
    counts = {r: 0 for r in roles}
    for line in clean_script_text(script).splitlines():
        m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        who = _normalize_role(m.group(1))
        if who in roles and not _is_banned_role(who):
            counts[who] += 1
    return counts

# 의미 없는 채우기 대사 패턴
_FILLER_PAT = re.compile(r"^\s*(?:[.]{2,}|…+|음+\.{0,3}$|음+$|아+|어+|흠+|하아+|에에+|허허+|헉+|끙+|으으+|[. ]+)$")

def is_meaningful_utterance(text: str) -> bool:
    """
    내용 품질 검증: '음...', '...', '아...' 같은 채우기 대사를 걸러낸다.
    - 최소 글자수
    - 한글 포함 + 단어 2개 이상 또는 종결부호/어미
    """
    t = (text or "").strip()
    if len(t) < 4:
        return False
    if _FILLER_PAT.match(t):
        return False
    has_hangul = bool(re.search(r"[가-힣]", t))
    has_sentence_end = bool(re.search(r"[.?!]$|다$|요$|죠$|네$|까$", t))
    has_space = (" " in t)
    return has_hangul and (has_space or has_sentence_end)

def fallback_hard_adjust(script: str, roles: List[str], targets: Dict[str, int]) -> str:
    """
    최후 수단: 초과 대사는 지문으로 강등, 부족 대사는 긴 대사를 분할/보충.
    '개수' 정합을 보장하기 위한 안전 장치.
    """
    lines = clean_script_text(script).splitlines()

    def _counts():
        return count_dialogue_lines("\n".join(lines), roles)

    # 1) 초과 → 지문 전환 (우선적으로 의미없는 대사를 지문 처리)
    cur = _counts()
    for r in roles:
        while cur[r] > targets[r]:
            # 뒤에서부터 스캔
            target_i = -1
            for i in range(len(lines)-1, -1, -1):
                m = re.match(rf"\s*{re.escape(r)}\s*[:：]\s*(.+)$", lines[i])
                if m:
                    text = m.group(1).strip()
                    target_i = i
                    # 의미 없는 대사면 그 라인을 우선적으로 지문화
                    if not is_meaningful_utterance(text):
                        break
            if target_i >= 0:
                text = re.sub(rf"^\s*{re.escape(r)}\s*[:：]\s*", "", lines[target_i]).strip()
                lines[target_i] = f"({text})"
            cur = _counts()
            if cur[r] <= targets[r]:
                break

    # 2) 부족 → 분할 보충/짧은 보충
    cur = _counts()
    for r in roles:
        while cur[r] < targets[r]:
            # 가장 긴 '의미 있는' 대사를 찾아 분할
            longest_i = -1
            longest_len = -1
            for i, ln in enumerate(lines):
                m = re.match(rf"\s*{re.escape(r)}\s*[:：]\s*(.+)$", ln)
                if m:
                    t = m.group(1).strip()
                    if is_meaningful_utterance(t) and len(t) > longest_len:
                        longest_len = len(t)
                        longest_i = i
            if longest_i == -1:
                # 없으면 보충 한 줄 추가
                lines.append(f"{r}: (잠시 생각하더니) 내 생각엔, 조금 더 구체적으로 계획을 세우는 게 좋겠어.")
                cur = _counts()
                continue

            # 문장 분할
            m = re.match(rf"\s*{re.escape(r)}\s*[:：]\s*(.+)$", lines[longest_i])
            t = m.group(1).strip()
            parts = re.split(r"([\.!\?])", t)
            sentences = []
            buf = ""
            for p in parts:
                buf += p
                if p in (".", "!", "?"):
                    sentences.append(buf.strip()); buf = ""
            if buf.strip():
                sentences.append(buf.strip())

            if len(sentences) >= 2:
                first = sentences[0]
                second = " ".join(sentences[1:]).strip() or "그리고, 우리 역할을 나눠서 해 보자."
                lines[longest_i] = f"{r}: {first}"
                lines.insert(longest_i + 1, f"{r}: {second}")
            else:
                # 자연스러운 짧은 보충
                lines.insert(longest_i + 1, f"{r}: 내 말은, 구체적으로 어떻게 할지 지금 정해보자는 거야.")

            cur = _counts()
            if cur[r] >= targets[r]:
                break

    return "\n".join(lines)

# ───────── 페이지 1: 대본 등록/입력 ──────────────────────────────
def page_script_input():
    # 경고 해결: use_container_width
    st.image("assets/dragon_intro.png", use_container_width=True)
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

        # AI 생성 대본을 자동으로 원본 등장인물만 필터링
        original_roles = extract_roles(st.session_state.get("script_raw", ""))
        filtered_lines = []
        for line in clean_script_text(st.session_state["script_final"]).splitlines():
            m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if m:
                character = _normalize_role(m.group(1))
                if character in original_roles:
                    filtered_lines.append(line)
            else:
                # 지문이나 장면 설명은 그대로 유지
                filtered_lines.append(line)
        filtered_script = "\n".join(filtered_lines)
        st.session_state["script_final"] = filtered_script

        if st.button("✅ 수정 완료", key="btn_save_script"):
            st.session_state["script"] = edited_script
            st.success("✅ 대본이 저장되었습니다!")

# ───────── 페이지 3: 대사 수 조절하기 ───────────────────────────────────
def page_role_balancer():
    st.header("⚖️ 3) 대사 수 조절하기")
    # 스크립트 우선순위: 현재 스크립트 > 재분배된 것 > 최종 스크립트 > 원본 스크립트
    script = st.session_state.get("current_script") or st.session_state.get("script_balanced") or st.session_state.get("script_final") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/생성하세요."); return
    seq = build_sequence(script); roles = extract_roles(script)
    if not roles: st.info("‘이름: 내용’ 형식이어야 역할을 인식해요."); return

    # 현재 대사 수 계산
    if st.session_state.get("script_balanced"):
        balanced_seq = build_sequence(st.session_state["script_balanced"])
        counts = {r:0 for r in roles}
        for ln in balanced_seq: counts[ln["who"]]+=1
        st.session_state["current_counts"] = counts
    else:
        counts = {r:0 for r in roles}
        for ln in seq: counts[ln["who"]]+=1
        st.session_state["current_counts"] = counts
    
    # 현재 대본 표시
    st.subheader("📜 현재 대본")
    current_script = st.session_state.get("script_balanced") or script
    st.code(current_script, language="text",height=500)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 현재 대사 수")
        for role, count in counts.items():
            st.write(f"**{role}**: {count}줄")
    
    with col2:
        st.subheader("🎯 목표 대사 수 설정")
        st.markdown("대사의 양을 골고루 분배해 보아요.")
        targets={}
        for r in roles:
            targets[r]=st.number_input(f"{r} 목표", min_value=0, value=counts[r], step=1, key=f"tgt_{r}")

    # ───────────────── 재분배 버튼
    st.markdown("---")
    if st.button("🔁 재분배하기", key="btn_rebalance", use_container_width=True):
        loading_placeholder = st.empty()
        loading_placeholder.info("⚖️ 역할을 재분배하고 있습니다...")

        original_script = st.session_state.get("script_raw", script)

        # 1) 프롬프트: 형식 강제 + 내용품질 강제
        targets_dict = {r: int(st.session_state.get(f"tgt_{r}", counts[r])) for r in roles}
        roles_csv = ", ".join(roles)
        hard_constraints = json.dumps(targets_dict, ensure_ascii=False)

        system_rules = (
            "너는 초등 연극 대본 편집자다. 반드시 정확한 형식과 줄 수를 지켜라.\n"
            "출력 규칙:\n"
            "1) 첫 줄에 <COUNTS>{...}</COUNTS> 형태로 '최종 대사 수(JSON)'를 출력한다. 이 JSON은 요청 타깃과 정확히 일치해야 한다.\n"
            "2) 두 번째 줄부터는 대본만 출력한다. 형식은 오직 `역할명: 내용` 또는 지문(괄호) 라인만.\n"
            "3) 등장인물은 요청된 목록만 사용. 새 인물 금지.\n"
            "4) 지문(괄호)은 대사 수 계산에서 제외된다.\n"
            "5) 대사는 반드시 인물의 감정/상황/관계를 드러내는 **구체적이고 자연스러운 문장**이어야 한다.\n"
            "6) 같은 말이나 의미 없는 반복(예: '음...', '...', '아...')으로 줄 수를 채우지 말라.\n"
            "7) 기승전결을 유지·개선하고, 장면 전환은 짧은 연결 문장으로 자연스럽게 처리하라."
        )

        user_prompt = f"""
원본 대본:
{original_script}

등장인물(고정): {roles_csv}

목표 대사 수(JSON):
{hard_constraints}

요구:
- 각 등장인물이 말하는 '대사 줄'의 개수가 목표와 정확히 일치해야 한다.
- 내용은 상황과 주제에 맞게 자연스럽고 구체적으로 채워라. 같은 말 반복·의미 없는 말(음..., ...) 금지.
- 불가피하면 초과 대사는 '지문(괄호)'으로 전환해 개수를 맞춘다.
- 절대 새 인물을 추가하지 말라.
"""

        def ask_llm(req_messages):
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=req_messages,
                temperature=0.4,
                max_tokens=2400,
            ).choices[0].message.content.strip()

        # 2) 생성 → 검증(줄수+내용) → 수정 루프
        msgs = [
            {"role": "system", "content": system_rules},
            {"role": "user", "content": user_prompt},
        ]

        MAX_TRIES = 4
        final_script = None
        body = ""
        for attempt in range(1, MAX_TRIES + 1):
            txt = ask_llm(msgs)

            # 헤더 파싱
            m = re.search(r"<COUNTS>\s*(\{.*?\})\s*</COUNTS>", txt, flags=re.S)
            body = re.sub(r"^.*?</COUNTS>\s*\n?", "", txt, flags=re.S) if m else txt
            try:
                _ = json.loads(m.group(1)) if m else {}
            except Exception:
                pass

            # 원본에 없는 역할은 지문으로 강등하여 줄수에 영향 없게 함
            filtered_lines = []
            for line in clean_script_text(body).splitlines():
                mm = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
                if mm:
                    character = _normalize_role(mm.group(1))
                    text = mm.group(2).strip()
                    if character in roles:
                        filtered_lines.append(f"{character}: {text}")
                    else:
                        filtered_lines.append(f"({text})")
                else:
                    filtered_lines.append(line)
            body = "\n".join(filtered_lines)

            # 실제 줄 수/내용 검증
            actual = count_dialogue_lines(body, roles)

            bad_lines = []  # (idx, role, text)
            for i, line in enumerate(body.splitlines()):
                mm = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
                if not mm:
                    continue
                role = _normalize_role(mm.group(1))
                text = mm.group(2).strip()
                if role in roles and not is_meaningful_utterance(text):
                    bad_lines.append((i+1, role, text))

            if actual == targets_dict and not bad_lines:
                final_script = body
                break

            # 불일치/품질 문제 보고하고 재요청
            diffs = "\n".join([f"- {r}: 실제 {actual.get(r,0)} vs 목표 {targets_dict[r]}" for r in roles if actual.get(r,0)!=targets_dict[r]])
            quality = "\n".join([f"- #{i}: {r} → '{t}'" for (i,r,t) in bad_lines])

            reason = []
            if diffs: reason.append(f"[줄 수 불일치]\n{diffs}")
            if bad_lines: reason.append(f"[의미 없는 대사 감지]\n{quality}")
            reason_txt = "\n\n".join(reason) if reason else "(규칙 위반 없음)"

            msgs.append({"role": "assistant", "content": txt})
            msgs.append({
                "role": "user",
                "content": f"""아래 문제를 모두 수정해서 다시 출력하세요.

{reason_txt}

반드시 지킬 것:
- 첫 줄: <COUNTS>{{...}}</COUNTS> (목표와 정확히 일치)
- 이후: 대본만 (역할명: 내용 또는 지문)
- 의미 없는 채우기 대사 금지. 각 대사는 구체적이고 자연스러워야 함.
- 새 인물 금지. 필요하면 지문으로 처리.
"""
            })

        # 3) 그래도 안 맞으면 비상 후처리
        if final_script is None:
            loading_placeholder.warning("LLM이 목표 줄 수/내용을 100% 맞추지 못해, 안전 후처리를 적용합니다.")
            body_after = fallback_hard_adjust(body, roles, targets_dict)
            final_script = body_after

        # 최종 저장
        st.session_state["script_balanced"] = final_script
        st.session_state["current_script"] = final_script

        loading_placeholder.empty()
        st.success("✅ 재분배 완료! (줄 수 정확 + 내용 품질 검증 적용)")
        st.rerun()

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
def page_rehearsal_partner():
    st.header("🎙️ 5) AI 대본 연습 — 줄 단위 (한 번 클릭→자동 분석)")

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
        voice_descriptions = {
            "민준 (남성, 따뜻하고 친근한 목소리)": "🎭 **따뜻하고 친근한 목소리** - 선생님이나 부모님 역할에 적합해요!",
            "현우 (남성, 차분하고 신뢰감 있는 목소리)": "🎭 **차분하고 신뢰감 있는 목소리** - 의사나 경찰관 같은 전문직 역할에 좋아요!",
            "지호 (남성, 활기차고 밝은 목소리)": "🎭 **활기차고 밝은 목소리** - 친구나 동생 역할에 어울려요!",
            "지민 (여성, 부드럽고 친절한 목소리)": "🎭 **부드럽고 친절한 여성 목소리** - 친절한 선생님이나 언니 역할에 어울려요! ✨",
            "소연 (여성, 귀엽고 명랑한 목소리)": "🎭 **귀엽고 명랑한 여성 목소리** - 귀여운 친구나 동생 역할에 최고예요! ✨",
            "하은 (여성, 차분하고 우아한 목소리)": "🎭 **차분하고 우아한 여성 목소리** - 우아한 공주나 여왕 역할에 어울려요! ✨",
            "민지 (여성, 밝고 경쾌한 목소리)": "🎭 **밝고 경쾌한 여성 목소리** - 활발한 친구나 운동선수 역할에 어울려요! ✨"
        }
        
        st.success(f"✅ **선택된 목소리**: {voice_label}")
        st.info(voice_descriptions.get(voice_label, "멋진 목소리네요!"))

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
        st.success("🎉 끝까지 진행했습니다. 이제 연습 종료 & 종합 피드백을 받아보세요!")
        st.info("💡 아래의 '🏁 연습 종료 & 종합 피드백' 버튼을 눌러 연습 결과를 확인해보세요!")
    else:
        cur_line = seq[cur_idx]
        st.markdown(f"#### 현재 줄 #{cur_idx+1}: **{cur_line['who']}** — {cur_line['text']}")

        # (내 차례)
        if cur_line["who"] == my_role:
            st.info("내 차례예요. 아래 **마이크 버튼을 한 번만** 눌러 말하고, 버튼이 다시 바뀌면 자동 분석이 시작됩니다.")
            audio_bytes = None
            if audio_recorder is not None:
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
            st.info("지금은 상대역 차례예요. ‘🔊 파트너 음성 듣기’로 듣거나, 이전/다음 줄로 이동할 수 있어요.")
            if st.button("🔊 파트너 음성 듣기", key=f"partner_say_live_cur_{cur_idx}"):
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
    if st.button("🏁 연습 종료 & 종합 피드백", key="end_feedback"):
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
            # 경고 해결: use_container_width
            st.image("assets/dragon_end.png", use_container_width=True)
            st.markdown("""
            🐉 **이제 연극 용이 모두 성장했어요!** 
            
            다시 돌아가서 연극 대모험을 완료해보세요! 🎭✨
            """)
            
            st.session_state["next_step_hint"] = "🎉 연극 연습 완료! 새로운 모험을 시작해보세요!"

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
        "🎙️ 5) AI 대본 연습": page_rehearsal_partner
    }

    #sidebar_status()  # (요청: 다른 기능은 그대로 두기 – 주석 상태 유지)

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
