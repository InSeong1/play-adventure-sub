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
@media (prefers-color-scheme: dark) {
    :root {
        --bg:#1a1a1a; --card:#2d2d2d; --accent:#ff6b6b; --accent2:#4ecdc4;
        --ink:#ffffff; --ok:#4caf50; --warn:#ff9800; --bad:#f44336; --muted:#b0b0b0;
    }
}
[data-testid="stAppViewContainer"] { background-color: var(--bg) !important; }
body { background: var(--bg); color: var(--ink) !important; }
.block-container { background: var(--card) !important; border-radius:14px; padding:24px 22px 18px !important; overflow:visible !important; color: var(--ink) !important; }
div[data-testid="stHeader"] { height:auto !important; }
div[data-testid="stMarkdownContainer"] h1, h1, h2, h3, h4, h5, h6 { line-height:1.3 !important; margin-top:0.25rem !important; color: var(--ink) !important; }
[role="radiogroup"] label p { line-height:1.25 !important; color: var(--ink) !important; }
.stButton>button { background: linear-gradient(90deg,var(--accent),var(--accent2)); color: var(--ink); border:none; border-radius:10px; padding:10px 16px; font-weight:700; }
.stButton>button:hover { opacity:.93 }
hr { border-color: var(--accent); }
.small { font-size: 0.9rem; color: var(--muted); }
.card { background: var(--card); padding:14px 14px; border-radius:12px; border:1px solid var(--accent); color: var(--ink) !important; }
.card h4 { margin:0 0 8px 0; font-size:1.05rem; color: var(--ink) !important; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:0.85rem; }
.b-ok { background: var(--ok); color: white; border:1px solid var(--ok); }
.b-warn { background: var(--warn); color: white; border:1px solid var(--warn); }
.b-bad { background: var(--bad); color: white; border:1px solid var(--bad); }
.kv { display:flex; align-items:center; gap:8px; margin:6px 0; }
.kv .k { width:130px; color: var(--muted); } 
.kv .v { font-weight:700; color: var(--ink); }
.gauge { width:100%; height:10px; background: var(--muted); border-radius:999px; position:relative; overflow:hidden; }
.gauge > span { position:absolute; left:0; top:0; bottom:0; width:0%; background:linear-gradient(90deg,var(--accent),var(--accent2)); border-radius:999px; transition:width .35s ease; }
.hi { background: var(--card); border:1px dashed var(--accent); border-radius:8px; padding:8px 10px; color: var(--ink) !important; }
.hi span { padding:0 2px; }
.hi .ok { color: var(--ok); font-weight:700; }
.hi .miss { color: var(--bad); text-decoration:underline; }
.rec-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; background: var(--muted); }
.rec-on { background: var(--bad); box-shadow:0 0 0 6px rgba(244,67,54,.2); }
.stMarkdown, .stText, .stCodeBlock, .stDataFrame { color: var(--ink) !important; }
section[data-testid="stSidebar"] { background-color: var(--card) !important; color: var(--ink) !important; width: 300px !important; }
section[data-testid="stSidebar"] .stMarkdown { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-list"] { background-color: var(--card) !important; }
.stTabs [data-baseweb="tab"] { color: var(--ink) !important; }
.stFileUploader { color: var(--ink) !important; }
.stTextArea textarea { background-color: var(--card) !important; color: var(--ink) !important; border-color: var(--accent) !important; }
.stSelectbox select { background-color: var(--card) !important; color: var(--ink) !important; border-color: var(--accent) !important; }
.stCheckbox label { color: var(--ink) !important; }
.stRadio label { color: var(--ink) !important; }
.stNumberInput input { background-color: var(--card) !important; color: var(--ink) !important; border-color: var(--accent) !important; }
.stRadio > div > div > label:nth-child(2),
.stRadio > div > div > label:nth-child(4) { border-top: 2px solid var(--accent) !important; margin-top: 10px !important; padding-top: 10px !important; }
</style>
"""

# ───────── 공통 유틸 ────────────────────────────────────────────────
def clean_script_text(t: str) -> str:
    return (t or "").replace("\r\n","\n").replace("\r","\n").strip()

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

# 추가: 인물별 줄 수 집계
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
    "현우 (남성, 활기차고 밝은 목소리)", 
    "지호 (남성, 차분하고 신뢰감 있는 목소리)",
    "지민 (여성, 부드럽고 친절한 목소리)",
    "소연 (여성, 귀엽고 명랑한 목소리)",
    "하은 (여성, 차분하고 우아한 목소리)",
    "민지 (여성, 밝고 경쾌한 목소리)"
]

VOICE_MAP_SAFE = {
    "민준 (남성, 따뜻하고 친근한 목소리)": "alloy",
    "현우 (남성, 활기차고 밝은 목소리)": "verse", 
    "지호 (남성, 차분하고 신뢰감 있는 목소리)": "onyx",
    "지민 (여성, 부드럽고 친절한 목소리)": "nova",
    "소연 (여성, 귀엽고 명랑한 목소리)": "shimmer",
    "하은 (여성, 차분하고 우아한 목소리)": "coral",
    "민지 (여성, 밝고 경쾌한 목소리)": "echo"
}

def _pitch_shift_mp3(mp3_bytes: bytes, semitones: float) -> bytes:
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
        if "여성" in voice_label:
            if "지민" in voice_label:  audio = _pitch_shift_mp3(audio, +2.5)
            elif "소연" in voice_label: audio = _pitch_shift_mp3(audio, +3.0)
            elif "하은" in voice_label: audio = _pitch_shift_mp3(audio, +2.0)
            elif "민지" in voice_label: audio = _pitch_shift_mp3(audio, +2.8)
            else: audio = _pitch_shift_mp3(audio, +2.0)
        elif "남성" in voice_label:
            if "10대" in voice_label: audio = _pitch_shift_mp3(audio, -1.0)
            elif "20대" in voice_label: audio = _pitch_shift_mp3(audio, -0.5)
            elif "30대" in voice_label: audio = _pitch_shift_mp3(audio, +1.0)
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
        def _lead_sil(seg, silence_thresh=-40.0, chunk_ms=10):
            trim_ms = 0
            while trim_ms < len(seg) and seg[trim_ms:trim_ms+chunk_ms].dBFS < silence_thresh:
                trim_ms += chunk_ms
            return trim_ms
        start = _lead_sil(seg); end = _lead_sil(seg.reverse())
        if start+end < len(seg): seg = seg[start:len(seg)-end]
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
        mean_sq = sum((x/maxv)*(x/maxv) for x in arr)/len(arr)
        rms = math.sqrt(max(mean_sq, 1e-12))
        rms_db = 20.0*math.log10(rms)
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
        if energies:
            rng = (max(energies)-min(energies))
            if rng>0.25 and rms_db>-20 and pause_ratio<0.15: tone="화내는 어조"
            elif rng>0.18 and pause_ratio>=0.2 and rms_db>-30: tone="즐거운 어조"
            elif rms_db<-35 and pause_ratio>0.25: tone="슬픈 어조"
            elif rng<0.1 and pause_ratio<0.1: tone="담담한 어조"
            else: tone="보통 어조"
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
    if _lb is not None and _np is not None:
        try:
            y, sr = _lb.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            if y is None or (hasattr(y, "size") and y.size == 0):
                raise RuntimeError("empty audio")
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
            if rng>20 and rms_dbfs>-20 and pause_ratio<0.15: tone="화내는 어조"
            elif rng>15 and pause_ratio>=0.2 and rms_dbfs>-30: tone="즐거운 어조"
            elif rms_dbfs<-35 and pause_ratio>0.25: tone="슬픈 어조"
            elif rng<10 and pause_ratio<0.1: tone="담담한 어조"
            else: tone="보통 어조"
            return {"speed_label":speed,"volume_label":volume,"tone_label":tone,"spacing_label":spacing,
                    "syllables_per_sec":syl_rate,"wps":wps,"rms_db":rms_dbfs,
                    "f0_hz":None,"f0_var":None,"pause_ratio":pause_ratio}
        except Exception:
            pass
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
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><h4>🗣️ 말속도</h4>"+_badge(sp)+
                    f"<div class='kv'><div class='k'>음절/초</div><div class='v'>{(sps or 0):.2f}</div></div>"+
                    _gauge_html(_score_speed(sps))+"</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h4>🔊 목소리 크기</h4>"+_badge(vo)+
                    f"<div class='kv'><div class='k'>RMS(dBFS)</div><div class='v'>{(voldb if voldb is not None else 0):.1f}</div></div>"+
                    _gauge_html(_score_volume(voldb))+"</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>🎭 어조(피치)</h4>"+_badge(to)+
                "<div style='font-size: 0.8rem; color: #666; margin-top: 8px;'>💡 <strong>참고:</strong> 어조는 목소리의 높낮이와 변화로 판단해요. 실제 감정과 다를 수 있으니 참고만 해주세요! 😊</div>"+
                "</div>", unsafe_allow_html=True)

# ───────── 페이지 1: 대본 등록/입력 ──────────────────────────────
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
                    "1) **장면 1, 장면 2, 장면 3 ...** 최소 4장면 이상.\n"
                    "2) 장면 간 자연스러운 전환과 사건 배치.\n"
                    "3) 대사는 `이름: 내용`, 지문은 ( ) 만 사용. 머릿말을 역할명으로 쓰지 않기.\n"
                    "4) 주제와 일관성 유지, 마지막 장면에서 갈등 해결.\n\n"
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
        edited_script = st.text_area("대본 수정하기", value=st.session_state["script_final"], height=300, key="script_editor")
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

# ───────── 하이브리드 재분배(추가 전용/삭제 전용) ─────────────────────
def _augment_with_additions_only(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 3) -> str:
    current = _count_lines_by_role(original_script, roles)
    deficits = {r: max(0, targets.get(r, current.get(r, 0)) - current.get(r, 0)) for r in roles}
    if all(v == 0 for v in deficits.values()):
        return original_script

    numbered_lines = [f"{i:04d}: {ln}" for i, ln in enumerate(clean_script_text(original_script).splitlines(), 1)]
    numbered_script = "\n".join(numbered_lines)

    sys = (
        "당신은 초등 연극 대본 편집자입니다. 기존 대사는 절대 수정/삭제하지 말고, "
        "사용자가 지정한 '부족한 줄 수'만큼 새 대사를 끼워 넣어 주세요. "
        "등장인물은 지정 목록만 사용하고, 지문은 괄호( )만 사용합니다. "
        "원래 대본의 기·승·전·결 및 갈등–해결 흐름을 유지하며, 인물 간 자연스러운 주고받기를 만드세요. "
        "출력은 지시문만."
    )
    deficit_list = "\n".join([f"- {r}: +{deficits[r]}줄" for r in roles if deficits[r] > 0])
    format_rule = (
        "형식(각 줄 별도):\n"
        "INSERT AFTER LINE <번호>: <인물명>: <대사내용>\n"
        "- 자연스러운 위치가 없으면 INSERT AFTER LINE END 사용\n"
        "- 설명/표/머릿말 없이 지시문만"
    )
    user = f"[원본 대본(줄 번호 포함)]\n{numbered_script}\n\n[부족한 줄 수]\n{deficit_list}\n\n{format_rule}"
    msg = [{"role":"system","content":sys},{"role":"user","content":user}]

    insert_after_map: Dict[object, List[str]] = {}
    ok_count = {r: 0 for r in roles}

    for _ in range(max_tries):
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=msg, temperature=0.5, max_tokens=1200
        )
        text = (res.choices[0].message.content or "").strip()
        pattern = r'^INSERT AFTER LINE (END|\d+):\s*([^:：]+)\s*[:：]\s*(.+)$'
        for raw in text.splitlines():
            m = re.match(pattern, raw.strip(), re.IGNORECASE)
            if not m: 
                continue
            where, who, content = m.groups()
            who = _normalize_role(who)
            if who not in roles:
                continue
            if ok_count[who] >= deficits.get(who, 0):
                continue
            key = ("END" if where.upper()=="END" else int(where))
            insert_after_map.setdefault(key, []).append(f"{who}: {content}")
            ok_count[who] += 1
        if all(ok_count[r] == deficits[r] for r in roles):
            break

    base_lines = clean_script_text(original_script).splitlines()
    out = []
    for idx, line in enumerate(base_lines, 1):
        out.append(line)
        if idx in insert_after_map:
            out.extend(insert_after_map[idx])
    if "END" in insert_after_map:
        out.extend(insert_after_map["END"])
    return "\n".join(out)

def _prune_with_deletions_only(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 3) -> str:
    current = _count_lines_by_role(original_script, roles)
    over = {r: max(0, current.get(r, 0) - targets.get(r, current.get(r, 0))) for r in roles}
    if all(v == 0 for v in over.values()):
        return original_script

    base_lines = clean_script_text(original_script).splitlines()
    numbered = [f"{i:04d}: {ln}" for i, ln in enumerate(base_lines, 1)]
    numbered_script = "\n".join(numbered)

    sys = (
        "당신은 초등 연극 대본 편집자입니다. '불필요·중복·주제와 무관'한 대사를 우선 삭제하여 "
        "목표 줄 수로 줄이되, 원래 대본의 기·승·전·결과 갈등–해결 흐름을 유지하세요. "
        "기존 대사 문구는 바꾸지 말고 '삭제'만 하세요. 필요한 경우 매우 짧은 연결 지문(괄호)만 추가할 수 있습니다. "
        "출력은 지시문만."
    )
    over_list = "\n".join([f"- {r}: -{over[r]}줄" for r in roles if over[r] > 0])
    format_rule = (
        "형식(각 줄 별도):\n"
        "DELETE LINE <번호>\n"
        "또는 흐름 보강 지문: INSERT AFTER LINE <번호>: (짧은 연결 지문)"
    )
    user = f"[원본 대본(줄 번호 포함)]\n{numbered_script}\n\n[줄여야 할 개수]\n{over_list}\n\n{format_rule}"
    msg = [{"role":"system","content":sys},{"role":"user","content":user}]

    deletions: set = set()
    insert_map: Dict[object, List[str]] = {}

    for _ in range(max_tries):
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=msg, temperature=0.4, max_tokens=1200
        )
        text = (res.choices[0].message.content or "").strip()
        del_pat = r'^DELETE LINE (\d+)\s*$'
        ins_pat = r'^INSERT AFTER LINE (END|\d+):\s*\((.+)\)\s*$'
        for raw in text.splitlines():
            s = raw.strip()
            m1 = re.match(del_pat, s, re.IGNORECASE)
            if m1:
                deletions.add(int(m1.group(1))); continue
            m2 = re.match(ins_pat, s, re.IGNORECASE)
            if m2:
                where, content = m2.groups()
                key = ("END" if where.upper()=="END" else int(where))
                insert_map.setdefault(key, []).append(f"({content.strip()})"); continue
        break

    out = []
    for i, ln in enumerate(base_lines, 1):
        if i in deletions:
            continue
        out.append(ln)
        if i in insert_map:
            out.extend(insert_map[i])
    if "END" in insert_map:
        out.extend(insert_map["END"])
    return "\n".join(out)

# ───────── 페이지 3: 대사 수 조절하기 ───────────────────────────────────
def page_role_balancer():
    st.header("⚖️ 3) 대사 수 조절하기")
    script = (st.session_state.get("current_script") 
              or st.session_state.get("script_balanced") 
              or st.session_state.get("script_final") 
              or st.session_state.get("script_raw",""))
    if not script: 
        st.warning("먼저 대본을 입력/생성하세요."); return
    roles = extract_roles(script)
    if not roles: 
        st.info("‘이름: 내용’ 형식이어야 역할을 인식해요."); return

    counts = _count_lines_by_role(script, roles)
    st.subheader("📜 현재 대본")
    st.code(script, language="text", height=480)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 현재 대사 수")
        for r in roles:
            st.write(f"**{r}**: {counts.get(r,0)}줄")
    with col2:
        st.subheader("🎯 목표 대사 수")
        targets: Dict[str,int] = {}
        for r in roles:
            targets[r] = st.number_input(f"{r} 목표", min_value=0, value=counts[r], step=1, key=f"tgt_{r}")

    st.markdown("---")
    if st.button("🔁 재분배하기", key="btn_rebalance", use_container_width=True):
        with st.spinner("자연스러운 흐름을 유지하며 삭제/추가 반영 중..."):
            try:
                new_script = script
                # 1) 감소: 삭제 먼저
                if any(targets[r] < counts.get(r,0) for r in roles):
                    new_script = _prune_with_deletions_only(client, new_script, roles, targets, max_tries=3)
                # 2) 증가: 부족분 추가
                after_counts = _count_lines_by_role(new_script, roles)
                if any(targets[r] > after_counts.get(r,0) for r in roles):
                    new_script = _augment_with_additions_only(client, new_script, roles, targets, max_tries=3)

                st.session_state["script_balanced"] = new_script
                st.session_state["current_script"] = new_script
                final_counts = _count_lines_by_role(new_script, roles)

                st.success("✅ 재분배 완료! 아래 결과를 확인하세요.")
                st.code(new_script, language="text", height=480)
                st.info("최종 줄 수: " + ", ".join([f"{r} {final_counts.get(r,0)}줄" for r in roles]))
            except Exception as e:
                st.error(f"재분배 중 오류: {e}")

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
    st.header("🎙️ 5) AI 대본 연습 — 줄 단위(한 번 클릭→자동 분석)")

    script = st.session_state.get("script_final") or st.session_state.get("script_balanced") or st.session_state.get("script_raw","")
    if not script:
        st.warning("먼저 대본을 등록/생성하세요."); return

    seq = build_sequence(script); roles = extract_roles(script)
    if not seq or not roles:
        st.info("‘이름: 내용’ 형식이어야 리허설 가능해요."); return

    st.session_state.setdefault("duet_cursor", 0)
    st.session_state.setdefault("duet_turns", [])
    st.session_state.setdefault("auto_done_token", None)

    want_metrics = st.checkbox("텍스트 분석 포함(말속도·크기·어조·띄어읽기)", value=True, key="ck_metrics")

    st.markdown("**🎭 상대역 음성 선택**")
    st.markdown("연극에 적합한 목소리를 선택해보세요!")
    voice_label = st.selectbox("상대역 음성 선택", VOICE_KR_LABELS_SAFE, index=0,
                               key="tts_voice_label_safe",
                               help="각 목소리는 나이대와 특성이 표시되어 있어요. 역할에 맞는 목소리를 선택해보세요!")
    if voice_label:
        st.success(f"✅ **선택된 목소리**: {voice_label}")

    my_role = st.selectbox("내 역할(실시간)", roles, key="role_live")
    if "previous_role" not in st.session_state:
        st.session_state["previous_role"] = my_role
    elif st.session_state["previous_role"] != my_role:
        st.session_state["previous_role"] = my_role
        st.success(f"✅ 역할이 '{my_role}'로 변경되었습니다!")
        st.rerun()

    cur_idx = st.session_state.get("duet_cursor", 0)
    if cur_idx >= len(seq):
        st.success("🎉 끝까지 진행했습니다. 이제 연습 종료 & 종합 피드백을 받아보세요!")
        st.info("💡 아래의 '🏁 연습 종료 & 종합 피드백' 버튼을 눌러 연습 결과를 확인해보세요!")
    else:
        cur_line = seq[cur_idx]
        st.markdown(f"#### 현재 줄 #{cur_idx+1}: **{cur_line['who']}** — {cur_line['text']}")
        if cur_line["who"] == my_role:
            st.info("내 차례예요. 아래 **마이크 버튼을 한 번만** 눌러 말하고, 버튼이 다시 바뀌면 자동 분석이 시작됩니다.")
            audio_bytes = None
            if audio_recorder is not None:
                st.markdown("💡 **마이크 아이콘을 클릭하여 녹음 시작/중지**")
                audio_bytes = audio_recorder(text="🎤 말하고 인식(자동 분석)", sample_rate=16000,
                                             pause_threshold=2.0, key=f"audrec_one_{cur_idx}")
            else:
                st.warning("audio-recorder-streamlit 패키지가 필요합니다. `pip install audio-recorder-streamlit`")

            if audio_bytes:
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
                        turns.append({"line_idx": cur_idx+1, "who": cur_line["who"],
                                      "expected": expected_core, "spoken": stt, "score": score})
                        st.session_state["duet_turns"] = turns
                        s.update(label="✅ 인식 완료", state="complete")

            cA, cB = st.columns(2)
            with cA:
                if st.button("⬅️ 이전 줄로 이동", key=f"prev_live_{cur_idx}"):
                    st.session_state["duet_cursor"] = max(0, cur_idx-1)
                    st.session_state["auto_done_token"]=None
                    st.rerun()
            with cB:
                if st.button("➡️ 다음 줄 이동", key=f"next_live_{cur_idx}"):
                    st.session_state["duet_cursor"] = min(len(seq), cur_idx+1)
                    st.session_state["auto_done_token"]=None
                    st.rerun()
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
                    st.rerun()
            with cB:
                if st.button("➡️ 다음 줄 이동", key=f"next_live2_{cur_idx}"):
                    st.session_state["duet_cursor"] = min(len(seq), cur_idx+1)
                    st.session_state["auto_done_token"]=None
                    st.rerun()

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
            st.balloons()
            st.image("assets/dragon_end.png", width='stretch')
            st.markdown("🐉 **이제 연극 용이 모두 성장했어요!** 다시 돌아가서 연극 대모험을 완료해보세요! 🎭✨")
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
    if st.session_state.get("next_step_hint"):
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        st.sidebar.markdown("### 💡 다음 단계")
        st.sidebar.info(st.session_state["next_step_hint"])
        if st.sidebar.button("✅ 확인했어요", key="clear_hint"):
            del st.session_state["next_step_hint"]; st.rerun()
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='small'>지문(괄호)은 TTS에서 읽지 않도록 처리됩니다.</div>", unsafe_allow_html=True)

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

    #sidebar_status()
    all_pages = list(pages.keys())
    sel = st.sidebar.radio("메뉴", all_pages, 
                          index=all_pages.index(st.session_state["current_page"]), 
                          key="nav_radio")
    st.session_state["current_page"]=sel

    if st.sidebar.button("전체 초기화", key="btn_reset_all"):
        st.session_state.clear()
        st.rerun()

    pages[sel]()

if __name__=="__main__":
    main()
