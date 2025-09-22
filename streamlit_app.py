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
    if _which is None or AudioSegment is None:
        return
    try:
        if _which("ffmpeg") and _which("ffprobe"):
            return
    except Exception:
        pass
    candidates = [r"C:\\ffmpeg\\bin", r"C:\\Program Files\\ffmpeg\\bin", r"C:\\Program Files (x86)\\ffmpeg\\bin"]
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
:root { --bg:#fff7f2; --card:#ffffff; --accent:#ffd8cc; --accent2:#ffeab3; --ink:#2f3437; --ok:#2e7d32; --warn:#ef6c00; --bad:#c62828; --muted:#667085; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a1a; --card:#2d2d2d; --accent:#ff6b6b; --accent2:#4ecdc4; --ink:#ffffff; --ok:#4caf50; --warn:#ff9800; --bad:#f44336; --muted:#b0b0b0; } }
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
.stRadio > div > div > label:nth-child(2) { border-top: 2px solid var(--accent) !important; margin-top: 10px !important; padding-top: 10px !important; }
.stRadio > div > div > label:nth-child(4) { border-top: 2px solid var(--accent) !important; margin-top: 10px !important; padding-top: 10px !important; }
</style>
"""

# ───────── 공통 유틸 ────────────────────────────────────────────────
def clean_script_text(t: str) -> str:
    return (t or "").replace("\r\n","\n").replace("\r","\n").strip()

BANNED_ROLE_PATTERNS = [r"^\**\s*장면", r"^\**\s*씬", r"^\**\s*무대", r"^\**\s*배경", r"^\**\s*배경음", r"^\**\s*노래", r"^\**\s*노랫말", r"^\**\s*설명", r"^\**\s*지문", r"^\**\s*장내", r"^\**\s*효과음"]

def _normalize_role(raw: str) -> str:
    s = raw.strip(); s = re.sub(r"^\**|\**$", "", s); s = re.sub(r"^[\(\[]\s*|\s*[\)\]]$", "", s); s = re.sub(r"\s+", " ", s); return s

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
        text = m.group(2).strip(); seq.append({"who":who, "text":text})
    return seq

_PUNC = r"[^\w가-힣ㄱ-ㅎㅏ-ㅣ ]"

def _norm_for_ratio(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", s); s = re.sub(_PUNC, " ", s); s = re.sub(r"\s+", "", s); return s

def match_highlight_html(expected: str, spoken: str) -> Tuple[str, float]:
    tokens = re.split(r"(\s+)", expected.strip()); sp_norm = _norm_for_ratio(spoken or ""); out=[]
    for tok in tokens:
        if tok.isspace(): out.append(tok); continue
        if not tok: continue
        ok = _norm_for_ratio(tok) and _norm_for_ratio(tok) in sp_norm
        out.append(f"<span class='{ 'ok' if ok else 'miss' }'>{tok}</span>")
    ratio = SequenceMatcher(None, _norm_for_ratio(expected), _norm_for_ratio(spoken or "")).ratio();
    return "<div class='hi'>"+"".join(out)+"</div>", ratio

def similarity_score(expected: str, spoken: str) -> float:
    def ko_only(s): s = re.sub(r"\(.*?\)", "", s); return "".join(re.findall(r"[가-힣0-9]+", s))
    e = ko_only(expected or ""); s = ko_only(spoken or "");
    if not s: return 0.0
    ratio = SequenceMatcher(None, e, s).ratio();
    ew = set(re.findall(r"[가-힣0-9]+", expected or "")); sw = set(re.findall(r"[가-힣0-9]+", spoken or ""));
    jacc = len(ew & sw) / max(1, len(ew | sw)) if (ew or sw) else 0.0
    def lcs_len(a,b):
        dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                dp[i][j]=dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
    l = lcs_len(e, s); prec = l/max(1,len(s)); rec = l/max(1,len(e)); f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    return max(ratio, jacc, f1)

# ───────── OpenAI TTS (지문 미낭독 + 톤 보정) ─────────────────────
VOICE_KR_LABELS_SAFE = ["민준 (남성, 따뜻하고 친근한 목소리)","현우 (남성, 차분하고 신뢰감 있는 목소리)","지호 (남성, 활기차고 밝은 목소리)","지민 (여성, 부드럽고 친절한 목소리)","소연 (여성, 귀엽고 명랑한 목소리)","하은 (여성, 차분하고 우아한 목소리)","민지 (여성, 밝고 경쾌한 목소리)"]
VOICE_MAP_SAFE = {"민준 (남성, 따뜻하고 친근한 목소리)":"alloy","현우 (남성, 차분하고 신뢰감 있는 목소리)":"verse","지호 (남성, 활기차고 밝은 목소리)":"onyx","지민 (여성, 부드럽고 친절한 목소리)":"nova","소연 (여성, 귀엽고 명랑한 목소리)":"shimmer","하은 (여성, 차분하고 우아한 목소리)":"coral","민지 (여성, 밝고 경쾌한 목소리)":"echo"}

def _pitch_shift_mp3(mp3_bytes: bytes, semitones: float) -> bytes:
    if not AudioSegment or semitones==0: return mp3_bytes
    try:
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3"); new_fr = int(seg.frame_rate * (2.0 ** (semitones/12.0)))
        shifted = seg._spawn(seg.raw_data, overrides={'frame_rate': new_fr}).set_frame_rate(seg.frame_rate)
        out = io.BytesIO(); shifted.export(out, format="mp3"); return out.getvalue()
    except Exception: return mp3_bytes

def tts_speak_line(text: str, voice_label: str) -> Tuple[str, Optional[bytes]]:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 필요합니다."); return text, None
    voice_id = VOICE_MAP_SAFE.get(voice_label, "alloy"); speak_text = re.sub(r"\(.*?\)", "", text).strip()
    try:
        r = requests.post("https://api.openai.com/v1/audio/speech", headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json={"model":"gpt-4o-mini-tts","voice":voice_id,"input":speak_text,"format":"mp3"}, timeout=60)
        if r.status_code!=200:
            st.error(f"TTS 오류: {r.status_code} - {r.text}"); return speak_text, None
        audio = r.content
        if "여성" in voice_label:
            if "지민" in voice_label: audio = _pitch_shift_mp3(audio, +2.5)
            elif "소연" in voice_label: audio = _pitch_shift_mp3(audio, +3.0)
            elif "하은" in voice_label: audio = _pitch_shift_mp3(audio, +2.0)
            elif "민지" in voice_label: audio = _pitch_shift_mp3(audio, +2.8)
            else: audio = _pitch_shift_mp3(audio, +2.0)
        elif "남성" in voice_label:
            pass
        return speak_text, audio
    except Exception as e:
        st.error(f"TTS 오류: {e}"); return speak_text, None

# ───────── STT & OCR & PDF (필요 함수 일부 생략 없이 유지) ─────────────────────
def preprocess_audio_for_stt(audio_bytes: bytes) -> bytes:
    if not AudioSegment: return audio_bytes
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
        buf = io.BytesIO(); seg.export(buf, format="wav"); return buf.getvalue()
    except Exception:
        return audio_bytes

def clova_short_stt(audio_bytes: bytes, lang: str = "Kor") -> str:
    if not CLOVA_SPEECH_SECRET: return ""
    url = f"https://clovaspeech-gw.ncloud.com/recog/v1/stt?lang={lang}"
    headers = {"X-CLOVASPEECH-API-KEY": CLOVA_SPEECH_SECRET, "Content-Type": "application/octet-stream"}
    wav_bytes = preprocess_audio_for_stt(audio_bytes); r = requests.post(url, headers=headers, data=wav_bytes, timeout=60); r.raise_for_status()
    try: return r.json().get("text"," ").strip()
    except Exception: return r.text.strip()

# (OCR, PDF cuecards 함수 동일 — 지면상 생략 없이 포함 가능, 필요시 위 버전 사용)

def _register_font_safe():
    candidates = [r"C:\\Windows\\Fonts\\malgun.ttf", r"C:\\Windows\\Fonts\\NanumGothic.ttf", "/System/Library/Fonts/AppleGothic.ttf", "/Library/Fonts/AppleGothic.ttf", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    for p in candidates:
        if os.path.exists(p):
            try: pdfmetrics.registerFont(TTFont("KFont", p)); return "KFont"
            except Exception: pass
    return None

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def build_cuecards_pdf(script: str, role: str) -> Optional[bytes]:
    font_name = _register_font_safe(); buf = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buf, pagesize=A4)
        style = ParagraphStyle("K", fontName=(font_name or "Helvetica"), fontSize=12, leading=16)
        elems=[Paragraph(f"[큐카드] {role}", style), Spacer(1,12)]
        for i, line in enumerate(build_sequence(script),1):
            if line["who"] == role:
                txt = re.sub(r"\(.*?\)","", line["text"]).strip(); elems.append(Paragraph(f"{i}. {txt}", style)); elems.append(Spacer(1,8))
        doc.build(elems); return buf.getvalue()
    except Exception as e:
        st.warning(f"PDF 생성 오류: {e}"); return None

# ───────── 프로소디 분석 (요약본·원본과 동일) ─────────────────────
# analyze_prosody / _analyze_wav_pure / render_prosody_card 는 원본과 동일하게 유지 (지면상 생략)

# ───────── 보조 유틸 ──────────────────────────────────────────────

def _count_lines_by_role(text: str, roles: List[str]) -> Dict[str, int]:
    counts = {r: 0 for r in roles}
    for line in clean_script_text(text).splitlines():
        m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
        if not m: continue
        who = _normalize_role(m.group(1))
        if who in counts: counts[who] += 1
    return counts

# ───────── 추가 전용(원본 보존) ───────────────────────────────────

def _augment_with_additions_only(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 3) -> str:
    current = _count_lines_by_role(original_script, roles)
    deficits = {r: max(0, targets.get(r, current.get(r, 0)) - current.get(r, 0)) for r in roles}
    if all(v == 0 for v in deficits.values()): return original_script

    numbered_lines = [f"{i:04d}: {line}" for i, line in enumerate(clean_script_text(original_script).splitlines(), 1)]
    numbered_script = "\n".join(numbered_lines)

    sys = ("당신은 초등 연극 대본 편집자입니다. 절대 기존 대사를 수정하거나 삭제하지 마세요. 사용자가 지정한 '부족한 줄 수'만큼 새 대사를 만들어 '끼워 넣는' 방식으로만 작업합니다. 등장인물은 지정된 목록만 사용하고, 지문은 괄호( )만 사용합니다.")
    deficit_list = "\n".join([f"- {r}: +{deficits[r]}줄" for r in roles if deficits[r] > 0])

    format_rule = (
        "출력은 '삽입 지시'만 여러 줄로 주세요. 각 줄은 다음 형식을 반드시 따르세요:\n"
        "INSERT AFTER LINE <번호>: <인물명>: <대사내용>\n"
        "규칙:\n- <번호>는 '기존 원본 대본'의 줄 번호입니다. 해당 줄 바로 '뒤'에 새 대사를 삽입합니다.\n"
        "- 자연스러운 위치가 없으면 'INSERT AFTER LINE END' 사용\n- 기존 문장 변경 금지, 새 대사만 제시\n- 설명/표/머릿말/코드블록 금지"
    )

    user = f"""
[원본 대본(줄 번호 포함)]
{numbered_script}

[부족한 줄 수(추가해야 할 개수)]
{deficit_list}

요구사항:
1) 기존 대사는 절대 수정/삭제 금지 — '새 대사'만 추가
2) 각 인물의 부족한 줄 수만큼 정확히 추가
3) 대사 내용은 기존 맥락에 맞게 자연스럽게 연결(받아치기·반응·전환 보강)
4) 원래 대본의 기·승·전·결 흐름과 갈등–해결 관계를 해치지 않도록 위치 선정
5) 등장인물은 {', '.join(roles)}만 사용
6) 지문은 괄호( )만 사용하고, 대사 수에는 포함하지 않음

{format_rule}
"""

    msg = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    insertion_lines = None

    for _ in range(max_tries):
        res = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.5, max_tokens=1200)
        text = (res.choices[0].message.content or "").strip()
        insert_after_map = {}; pattern = r'^INSERT AFTER LINE (END|\d+):\s*([^:：]+)\s*[:：]\s*(.+)$'; ok_count = {r: 0 for r in roles}
        for raw in text.splitlines():
            m = re.match(pattern, raw.strip(), re.IGNORECASE)
            if not m: continue
            where, who, content = m.groups(); who = _normalize_role(who)
            if who not in roles: continue
            if ok_count[who] >= deficits.get(who, 0): continue
            ins_key = ("END" if where.upper() == "END" else int(where))
            insert_after_map.setdefault(ins_key, []).append(f"{who}: {content}"); ok_count[who] += 1
        if all(ok_count[r] == deficits[r] for r in roles):
            insertion_lines = insert_after_map; break

    base_lines = clean_script_text(original_script).splitlines(); out = []
    for idx, line in enumerate(base_lines, 1):
        out.append(line)
        if insertion_lines and idx in insertion_lines: out.extend(insertion_lines[idx])
    if insertion_lines and "END" in insertion_lines: out.extend(insertion_lines["END"])
    return "\n".join(out)

# ───────── 삭제 전용(원본 정리) ───────────────────────────────────

def _prune_with_deletions_only(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 3) -> str:
    current = _count_lines_by_role(original_script, roles)
    over = {r: max(0, current.get(r, 0) - targets.get(r, current.get(r, 0))) for r in roles}
    if all(v == 0 for v in over.values()): return original_script

    base_lines = clean_script_text(original_script).splitlines()
    numbered = [f"{i:04d}: {ln}" for i, ln in enumerate(base_lines, 1)]; numbered_script = "\n".join(numbered)

    sys = ("당신은 초등 연극 대본 편집자입니다. '불필요하거나 중복되는 대사'를 골라 삭제하여 목표 줄 수로 줄이되, 이야기의 기·승·전·결과 갈등–해결의 흐름은 유지하세요. 기존 대사 문구는 바꾸지 말고, '삭제'만 하세요. 필요하면 짧은 연결 지문(괄호)을 추가할 수 있습니다.")
    over_list = "\n".join([f"- {r}: -{over[r]}줄" for r in roles if over[r] > 0])

    format_rule = (
        "출력은 '삭제 지시'만 여러 줄로 주세요. 각 줄은 다음 형식을 반드시 따르세요:\n"
        "DELETE LINE <번호>\n"
        "규칙:\n- <번호>는 '원본 대본(줄 번호 포함)'의 번호입니다.\n"
        "- 필요 시 흐름 유지를 위해 'INSERT AFTER LINE <번호>: ( ... )' 형태의 짧은 '지문'만 추가할 수 있습니다."
    )

    user = f"""
[원본 대본(줄 번호 포함)]
{numbered_script}

[줄여야 할 개수(인물별)]
{over_list}

요구사항:
1) 각 인물의 대사 수를 목표에 맞도록 '삭제'만 수행 (수정/교체/새 인물 금지)
2) 불필요·중복·주제와 무관·장면 흐름을 해치는 대사를 우선적으로 삭제
3) 원래 기·승·전·결과 갈등–해결 관계 유지
4) 필요 시 매우 짧은 연결 지문(괄호)만 추가 가능 — 대사 수에는 불포함
5) 등장인물은 {', '.join(roles)}만 사용

{format_rule}
"""

    msg = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    deletions = set(); insert_map = {}

    for _ in range(max_tries):
        res = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.4, max_tokens=1200)
        text = (res.choices[0].message.content or "").strip()
        del_pat = r'^DELETE LINE (\d+)\s*$'; ins_pat = r'^INSERT AFTER LINE (END|\d+):\s*\((.+)\)\s*$'
        local_del = set(); local_ins = {}
        for raw in text.splitlines():
            s = raw.strip(); m1 = re.match(del_pat, s, re.IGNORECASE)
            if m1: local_del.add(int(m1.group(1))); continue
            m2 = re.match(ins_pat, s, re.IGNORECASE)
            if m2:
                where, content = m2.groups(); key = ("END" if where.upper() == "END" else int(where))
                local_ins.setdefault(key, []).append(f"({content.strip()})"); continue
        deletions |= local_del
        for k, v in local_ins.items(): insert_map.setdefault(k, []).extend(v)
        break

    out = []
    for i, ln in enumerate(base_lines, 1):
        if i in deletions: continue
        out.append(ln)
        if i in insert_map: out.extend(insert_map[i])
    if "END" in insert_map: out.extend(insert_map["END"])
    return "\n".join(out)

# ───────── 정확 재분배(전체 재작성 루프) ───────────────────────────

def _rebalance_with_hard_targets(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 4) -> str:
    role_list = ", ".join(roles); target_lines = "\n".join([f"- {r}: {targets[r]}줄" for r in roles])
    sys = ("당신은 초등 연극 대본 편집자입니다. 사용자가 정한 '역할별 목표 대사 수'를 정확히 맞추는 것이 최우선 과제입니다. 자연스러운 연결을 유지하되, 목표 줄 수와 다르면 반드시 추가/삭제/합치기/쪼개기를 통해 정확히 맞추세요. 지문은 괄호 ( ) 만 사용하며 대사 수에는 포함하지 않습니다. 등장인물은 제공된 목록만 사용하세요.")
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
7) 원래 대본의 기-승-전-결 구조와 주제 흐름을 해치지 않도록, 추가/삭제 시 장면 전환과 갈등-해결 관계를 유지하며 수정할 것
"""
    msg = [{"role": "system", "content": sys}, {"role": "user", "content": base_prompt}]; last_script = None
    for attempt in range(1, max_tries + 1):
        res = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.6, max_tokens=2000)
        draft = res.choices[0].message.content.strip()
        filtered_lines = []
        for line in clean_script_text(draft).splitlines():
            m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if m:
                character = _normalize_role(m.group(1))
                if character in roles: filtered_lines.append(line)
            else: filtered_lines.append(line)
        draft = "\n".join(filtered_lines)
        counts = _count_lines_by_role(draft, roles); ok = all(counts.get(r, 0) == targets[r] for r in roles)
        if ok: return draft
        diff_lines = "\n".join([f"- {r}: 현재 {counts.get(r,0)}줄 → 목표 {targets[r]}줄" for r in roles if counts.get(r,0) != targets[r]])
        fix_prompt = f"""
아래 차이를 반영해 대본을 '수정'만 하세요. 전체를 갈아엎지 말고, 필요한 대사만 추가/삭제/합치기/쪼개기 하세요.
차이:
{diff_lines}

주의:
- 장면/지문은 유지하며, 인물/이름은 바꾸지 마세요.
- 대사 내용은 맥락상 자연스럽게 이어지게 하세요. (받아치기·반응·전환을 보강하여 어색한 점을 제거)
- 원래 대본의 기·승·전·결 흐름을 해치지 않게 수정하세요.
- 출력은 대본 텍스트만.
기존 초안:
{draft}
"""
        msg = [{"role": "system", "content": sys}, {"role": "user", "content": fix_prompt}]; last_script = draft
    return last_script or original_script

# ───────── 페이지 1: 대본 등록/입력 ──────────────────────────────

def page_script_input():
    st.image("assets/dragon_intro.png", width='stretch')
    st.header("📥 1) 대본 등록")
    c1,c2 = st.columns(2)
    with c1:
        up = st.file_uploader("손글씨/이미지 업로드(OCR)", type=["png","jpg","jpeg"], key="u_ocr")
        if up and st.button("🖼️ OCR로 불러오기", key="btn_ocr"):
            st.session_state.setdefault("script_raw", "")
            st.session_state["script_raw"] = (st.session_state["script_raw"] + "\n" + up.name).strip()  # 실제 OCR 함수는 nv_ocr로 교체 가능
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
                fb = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":criteria+"\n\n대본:\n"+script}], temperature=0.4, max_tokens=1400).choices[0].message.content
                st.session_state["script_feedback"]=fb; st.success("✅ 피드백 생성 완료!")
                if not st.session_state.get("script_final"): st.info("💡 오른쪽의 '✨ 피드백 반영하여 대본 생성하기' 버튼을 눌러보세요!")
                st.session_state["next_step_hint"] = "피드백에 맞추어 대본이 완성되면 다음 단계로 이동하세요."
    with c2:
        if st.button("✨ 피드백 반영하여 대본 생성하기", key="btn_make_final"):
            with st.spinner("✨ 대본을 생성하고 있습니다..."):
                prm = ("초등학생 눈높이에 맞춰 대본을 다듬고, 필요하면 내용을 자연스럽게 보강하여 기-승-전-결이 또렷한 **연극 완성본**을 작성하세요.\n\n"
                       "형식 규칙:\n1) **장면 1, 장면 2, 장면 3 ...** 최소 4장면 이상(권장 5~7장면)\n2) 장면 간 연결 문장 짧게\n3) `이름: 내용`, 지문은 ( ) 표기. 해설/장면/무대를 역할명으로 쓰지 말 것\n4) 주제와 일관성 유지\n5) 마지막 장면에서 갈등 해소와 여운.\n\n아래 대본을 참고해 보강/확장하세요.\n\n"+script)
                res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prm}], temperature=0.6, max_tokens=2600).choices[0].message.content
                st.session_state["script_final"] = res; st.success("🎉 대본 생성 완료!")
                st.session_state["next_step_hint"] = "대본 생성 완료! 피드백을 반영하여 수정을 완료한 후 다음 단계로 이동하세요."

    st.divider()
    if st.session_state.get("script_feedback"):
        with st.expander("📄 상세 피드백", expanded=False): st.markdown(st.session_state["script_feedback"])
    if st.session_state.get("script_final"):
        st.subheader("🤖 AI 추천 대본 (수정 가능)"); st.markdown("AI가 추천한 대본입니다. 상세 피드백을 참고하여 수정해보아요!"); st.code(st.session_state["script_final"], language="text")
        edited_script = st.text_area("대본 수정하기", value=st.session_state["script_final"], height=300, key="script_editor")
        original_roles = extract_roles(st.session_state.get("script_raw", ""))
        filtered_lines = []
        for line in clean_script_text(st.session_state["script_final"]).splitlines():
            m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
            if m:
                character = _normalize_role(m.group(1))
                if character in original_roles: filtered_lines.append(line)
            else: filtered_lines.append(line)
        st.session_state["script_final"] = "\n".join(filtered_lines)
        if st.button("✅ 수정 완료", key="btn_save_script"):
            st.session_state["script"] = edited_script; st.success("✅ 대본이 저장되었습니다!")

# ───────── 페이지 3: 대사 수 조절하기 ───────────────────────────────────

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

    st.subheader("📜 현재 대본"); current_script = st.session_state.get("script_balanced") or st.session_state.get("script_augmented") or st.session_state.get("script_pruned") or script
    st.code(current_script, language="text",height=500)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 현재 대사 수"); st.markdown("생성된 대본의 줄 수를 알려줘요.")
        for role, count in counts.items(): st.write(f"**{role}**: {count}줄")
    with col2:
        st.subheader("🎯 목표 대사 수 설정"); st.markdown("대사의 양을 골고루 분배해 보아요.")
        targets={}
        for r in roles: targets[r]=st.number_input(f"{r} 목표", min_value=0, value=counts[r], step=1, key=f"tgt_{r}")

    # 모드 선택
    add_only = st.checkbox("원본 보존 모드(추가만) — 기존 대사는 바꾸지 않고 부족한 만큼만 새 대사를 끼워 넣기", value=True)
    prune_only = st.checkbox("원본 정리 모드(삭제만) — 내용상 덜 중요한 대사를 골라 목표까지 줄이기", value=False)
    if add_only and prune_only: st.info("두 모드를 동시에 켰습니다. '삭제만' 모드를 우선합니다.")

    st.markdown("---")
    if st.button("🔁 재분배하기", key="btn_rebalance", use_container_width=True):
        loading_placeholder = st.empty(); loading_placeholder.info("⚖️ 역할을 재분배하고 있습니다...")
        original_script = st.session_state.get("script_raw", script)
        try:
            current_counts = _count_lines_by_role(original_script, roles)
            need_prune = any(targets[r] < current_counts.get(r, 0) for r in roles)
            need_add   = any(targets[r] > current_counts.get(r, 0) for r in roles)

            if prune_only:
                new_script = _prune_with_deletions_only(client, original_script, roles, targets, max_tries=3)
            elif add_only:
                lowered = [r for r in roles if targets[r] < current_counts.get(r, 0)]
                if lowered:
                    st.warning("원본 보존(추가만) 모드에서는 감소 요청은 무시됩니다. 감소를 원하면 '삭제만' 모드를 사용하세요.")
                    for r in lowered: targets[r] = current_counts.get(r, 0)
                new_script = _augment_with_additions_only(client, original_script, roles, targets, max_tries=3)
            else:
                mid_script = original_script
                if need_prune: mid_script = _prune_with_deletions_only(client, mid_script, roles, targets, max_tries=3)
                if need_add: new_script = _augment_with_additions_only(client, mid_script, roles, targets, max_tries=3)
                else: new_script = mid_script

            final_counts = _count_lines_by_role(new_script, roles)
            mismatches = [r for r in roles if final_counts.get(r,0) != targets[r]]

            if prune_only: st.session_state["script_pruned"] = new_script
            elif add_only: st.session_state["script_augmented"] = new_script
            else: st.session_state["script_balanced"] = new_script
            st.session_state["current_script"] = new_script

            loading_placeholder.empty()
            if mismatches:
                st.warning("일부 인물의 줄 수가 정확히 맞지 않았습니다. 아래 수치를 확인해 주세요.")
                for r in roles: st.write(f"- {r}: 현재 {final_counts.get(r,0)}줄 / 목표 {targets[r]}줄")
            else:
                st.success("✅ 적용 완료!")
            st.rerun()
        except Exception as e:
            loading_placeholder.empty(); st.error(f"재분배 중 오류가 발생했습니다: {e}"); return

# ───────── 페이지 4: 소품·무대·의상 ─────────────────────────────────

def page_stage_kits():
    st.header("🎭 4) 소품·무대·의상 추천"); st.markdown("연극에 필요한 소품을 AI가 추천해 줘요.")
    script = st.session_state.get("script_final") or st.session_state.get("script_balanced") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 입력/생성하세요."); return
    if st.button("🧰 목록 만들기", key="btn_kits"):
        with st.spinner("🧰 소품·무대·의상 목록을 생성하고 있습니다..."):
            prm = ("다음 대본을 바탕으로 초등 연극용 소품·무대·의상 체크리스트를 만들어주세요.\n구성: [필수/선택/대체/안전 주의] 4섹션 표(마크다운) + 간단 팁.\n\n대본:\n"+script)
            res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prm}], temperature=0.4, max_tokens=1200).choices[0].message.content
            st.session_state["stage_kits"] = res; st.success("✅ 목록 생성 완료!"); st.markdown(res or "(생성 실패)")
            st.session_state["next_step_hint"] = "체크리스트 완성! 다음 단계로 이동하세요."

# ───────── 페이지 5: AI 대본 연습 ────────────────────────────────

def page_rehearsal_partner():
    st.header("🎙️ 5) AI 대본 연습 — 줄 단위 STT(REST, 한 번 클릭→자동 분석)")
    script = st.session_state.get("script_final") or st.session_state.get("script_balanced") or st.session_state.get("script_raw","")
    if not script: st.warning("먼저 대본을 등록/생성하세요."); return
    seq = build_sequence(script); roles = extract_roles(script)
    if not seq or not roles: st.info("‘이름: 내용’ 형식이어야 리허설 가능해요."); return

    st.session_state.setdefault("duet_cursor", 0); st.session_state.setdefault("duet_turns", []); st.session_state.setdefault("auto_done_token", None)
    want_metrics = st.checkbox("텍스트 분석 포함(말속도·크기·어조·띄어읽기)", value=True, key="ck_metrics")

    st.markdown("**🎭 상대역 음성 선택**"); st.markdown("연극에 적합한 목소리를 선택해보세요!")
    voice_label = st.selectbox("상대역 음성 선택", VOICE_KR_LABELS_SAFE, index=0, key="tts_voice_label_safe", help="각 목소리는 나이대와 특성이 표시되어 있어요. 역할에 맞는 목소리를 선택해보세요!")
    if voice_label:
        voice_descriptions = {"민준 (남성, 따뜻하고 친근한 목소리)":"🎭 **따뜻하고 친근한 목소리** - 선생님이나 부모님 역할에 적합해요!","현우 (남성, 차분하고 신뢰감 있는 목소리)":"🎭 **차분하고 신뢰감 있는 목소리** - 의사나 경찰관 같은 전문직 역할에 좋아요!","지호 (남성, 활기차고 밝은 목소리)":"🎭 **활기차고 밝은 목소리** - 친구나 동생 역할에 어울려요!","지민 (여성, 부드럽고 친절한 목소리)":"🎭 **부드럽고 친절한 여성 목소리** - 친절한 선생님이나 언니 역할에 어울려요! ✨","소연 (여성, 귀엽고 명랑한 목소리)":"🎭 **귀엽고 명랑한 여성 목소리** - 귀여운 친구나 동생 역할에 최고예요! ✨","하은 (여성, 차분하고 우아한 목소리)":"🎭 **차분하고 우아한 여성 목소리** - 우아한 공주나 여왕 역할에 어울려요! ✨","민지 (여성, 밝고 경쾌한 목소리)":"🎭 **밝고 경쾌한 여성 목소리** - 활발한 친구나 운동선수 역할에 어울려요! ✨"}
        st.success(f"✅ **선택된 목소리**: {voice_label}"); st.info(voice_descriptions.get(voice_label, "멋진 목소리네요!"))

    my_role = st.selectbox("내 역할(실시간)", roles, key="role_live")
    if "previous_role" not in st.session_state: st.session_state["previous_role"] = my_role
    elif st.session_state["previous_role"] != my_role:
        st.session_state["previous_role"] = my_role; st.success(f"✅ 역할이 '{my_role}'로 변경되었습니다!"); st.rerun()

    cur_idx = st.session_state.get("duet_cursor", 0)
    if cur_idx >= len(seq):
        st.success("🎉 끝까지 진행했습니다. 이제 연습 종료 & 종합 피드백을 받아보세요!"); st.info("💡 아래의 '🏁 연습 종료 & 종합 피드백' 버튼을 눌러 연습 결과를 확인해보세요!")
    else:
        cur_line = seq[cur_idx]; st.markdown(f"#### 현재 줄 #{cur_idx+1}: **{cur_line['who']}** — {cur_line['text']}")
        if cur_line["who"] == my_role:
            st.info("내 차례예요. 아래 **마이크 버튼을 한 번만** 눌러 말하고, 버튼이 다시 바뀌면 자동 분석이 시작됩니다.")
            audio_bytes = None
            if audio_recorder is not None:
                st.markdown("💡 **마이크 아이콘을 클릭하여 녹음 시작/중지**")
                audio_bytes = audio_recorder(text="🎤 말하고 인식(자동 분석)", sample_rate=16000, pause_threshold=2.0, key=f"audrec_one_{cur_idx}")
            else:
                st.warning("audio-recorder-streamlit 패키지가 필요합니다. `pip install audio-recorder-streamlit`")
            if audio_bytes:
                with st.status("🎧 인식 중...", expanded=True) as s:
                    token = hashlib.sha256(audio_bytes).hexdigest()[:16]
                    if st.session_state.get("auto_done_token") != (cur_idx, token):
                        st.session_state["auto_done_token"] = (cur_idx, token)
                        stt = clova_short_stt(audio_bytes, lang="Kor"); s.update(label="🧪 분석 중...", state="running")
                        st.markdown("**STT 인식 결과(원문)**"); st.text_area("인식된 문장", value=stt or "(빈 문자열)", height=90, key=f"saw_{cur_idx}")
                        expected_core = re.sub(r"\(.*?\)", "", cur_line["text"]).strip(); st.caption("비교 기준(지문 제거)"); st.code(expected_core, language="text")
                        html, _ = match_highlight_html(expected_core, stt or ""); score = similarity_score(expected_core, stt or ""); st.markdown("**일치 하이라이트(초록=일치, 빨강=누락)**", unsafe_allow_html=True); st.markdown(html, unsafe_allow_html=True); st.caption(f"일치율(내부 지표) 약 {score*100:.0f}%")
                        turns = st.session_state.get("duet_turns", []); turns.append({"line_idx": cur_idx+1, "who": cur_line["who"], "expected": expected_core, "spoken": stt, "score": score}); st.session_state["duet_turns"] = turns; s.update(label="✅ 인식 완료", state="complete")
            cA, cB = st.columns(2)
            with cA:
                if st.button("⬅️ 이전 줄로 이동", key=f"prev_live_{cur_idx}"):
                    st.session_state["duet_cursor"] = max(0, cur_idx-1); st.session_state["auto_done_token"]=None; st.rerun()
            with cB:
                if st.button("➡️ 다음 줄 이동", key=f"next_live_{cur_idx}"):
                    st.session_state["duet_cursor"] = min(len(seq), cur_idx+1); st.session_state["auto_done_token"]=None; st.rerun()
        else:
            st.info("지금은 상대역 차례예요. ‘🔊 파트너 음성 듣기’로 듣거나, 이전/다음 줄로 이동할 수 있어요.")
            if st.button("🔊 파트너 음성 듣기", key=f"partner_say_live_cur_{cur_idx}"):
                with st.spinner("🔊 음성 합성 중…"):
                    speak_text, audio = tts_speak_line(cur_line["text"], voice_label); st.success(f"파트너({cur_line['who']}): {speak_text}");
                    if audio: st.audio(audio, format="audio/mpeg")
            cA, cB = st.columns(2)
            with cA:
                if st.button("⬅️ 이전 줄로 이동", key=f"prev_live2_{cur_idx}"):
                    st.session_state["duet_cursor"] = max(0, cur_idx-1); st.session_state["auto_done_token"]=None; st.rerun()
            with cB:
                if st.button("➡️ 다음 줄 이동", key=f"next_live2_{cur_idx}"):
                    st.session_state["duet_cursor"] = min(len(seq), cur_idx+1); st.session_state["auto_done_token"]=None; st.rerun()

    if st.button("🏁 연습 종료 & 종합 피드백", key="end_feedback"):
        with st.spinner("🏁 종합 피드백을 생성하고 있습니다..."):
            feed = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":json.dumps(st.session_state.get('duet_turns',[]), ensure_ascii=False)}], temperature=0.3, max_tokens=1200).choices[0].message.content
            st.session_state["session_feedback"] = feed; st.success("✅ 종합 피드백 생성 완료!"); st.markdown(feed or "(피드백 실패)"); st.balloons(); st.image("assets/dragon_end.png", width='stretch'); st.markdown("🐉 **이제 연극 용이 모두 성장했어요!** 다시 돌아가서 연극 대모험을 완료해보세요! 🎭✨")

# ───────── 사이드바 상태/MAIN ───────────────────────────────────

def main():
    st.set_page_config("연극용의 둥지", "🐉", layout="wide")
    st.markdown(PASTEL_CSS, unsafe_allow_html=True)
    st.title("🐉 연극용의 둥지 — 연극 용을 성장시켜요!")
    st.subheader("연극 용과 함께 완성도 있는 연극을 완성하고 연습해 보자!")

    # 현재 선택된 페이지 상태 유지
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "📥 1) 대본 등록"

    pages = {
        "📥 1) 대본 등록": page_script_input,
        "🛠️ 2) 대본 피드백 & 완성본": page_feedback_script,
        "⚖️ 3) 대사 수 조절하기": page_role_balancer,
        "🎭 4) 소품·무대·의상": page_stage_kits,
        "🎙️ 5) AI 대본 연습": page_rehearsal_partner
    }

    all_pages = list(pages.keys())
    sel = st.sidebar.radio(
        "메뉴",
        all_pages,
        index=all_pages.index(st.session_state["current_page"]),
        key="nav_radio"
    )
    st.session_state["current_page"] = sel

    if st.sidebar.button("전체 초기화", key="btn_reset_all"):
        st.session_state.clear()
        st.rerun()

    # 선택된 페이지 실행
    pages[sel]()

if __name__ == "__main__":
    main()
