# -*- coding: utf-8 -*-
import os, re
from typing import List, Dict
import streamlit as st
from openai import OpenAI

# 시크릿 키
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ───────── 유틸 함수 ─────────
def clean_script_text(text: str) -> str:
    return (text or "").replace("：", ":").strip()

def _normalize_role(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip())

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

def extract_roles(script: str) -> List[str]:
    roles: List[str] = []
    for line in clean_script_text(script).splitlines():
        m = re.match(r"\s*([^:：]+)\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        who = _normalize_role(m.group(1))
        # 장면/무대/해설 등 머릿말 배제 (간단 버전)
        if who and not re.match(r"^(장면|씬|무대|배경|해설|효과음)$", who):
            if who not in roles:
                roles.append(who)
    return roles

# ───────── 추가 전용 함수(원본 보존) ─────────
def _augment_with_additions_only(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 3) -> str:
    current = _count_lines_by_role(original_script, roles)
    deficits = {r: max(0, targets.get(r, current.get(r, 0)) - current.get(r, 0)) for r in roles}
    if all(v == 0 for v in deficits.values()):
        return original_script

    numbered_lines = []
    for i, line in enumerate(clean_script_text(original_script).splitlines(), 1):
        numbered_lines.append(f"{i:04d}: {line}")
    numbered_script = "\n".join(numbered_lines)

    sys = (
        "당신은 초등 연극 대본 편집자입니다. 절대 기존 대사를 수정하거나 삭제하지 마세요. "
        "사용자가 지정한 '부족한 줄 수'만큼 새 대사를 만들어 '끼워 넣는' 방식으로만 작업합니다. "
        "등장인물은 지정된 목록만 사용하고, 지문은 괄호( )만 사용합니다."
    )

    deficit_list = "\n".join([f"- {r}: +{deficits[r]}줄" for r in roles if deficits[r] > 0])

    format_rule = (
        "출력은 '삽입 지시'만 여러 줄로 주세요. 각 줄은 다음 형식을 반드시 따르세요:\n"
        "INSERT AFTER LINE <번호>: <인물명>: <대사내용>\n"
        "규칙:\n"
        "- <번호>는 '기존 원본 대본'의 줄 번호입니다. 해당 줄 바로 '뒤'에 새 대사를 삽입합니다.\n"
        "- 만약 자연스러운 위치가 없으면 'INSERT AFTER LINE END'를 사용하여 맨 끝에 추가합니다.\n"
        "- 기존 문장은 절대 바꾸지 말고, 새로운 대사만 제시하세요.\n"
        "- 출력에 설명/표/머릿말/코드블록 없이 '삽입 지시' 줄들만 나열하세요."
    )

    user = f"""
[원본 대본(줄 번호 포함)]
{numbered_script}

[부족한 줄 수(추가해야 할 개수)]
{deficit_list}

요구사항:
1) 기존 대사는 절대 수정/삭제 금지 — '새 대사'만 추가
2) 각 인물의 부족한 줄 수만큼 정확히 추가
3) 대사 내용은 기존 맥락에 맞게 자연스럽게 주고받기(받아치기·반응·전환 보강)
4) 원래 대본의 기·승·전·결과 갈등–해결 흐름을 해치지 않도록 위치 선정
5) 등장인물은 {', '.join(roles)}만 사용
6) 지문은 괄호( )만 사용하고, 대사 수에는 포함하지 않음

{format_rule}
"""

    msg = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    insertion_lines = None

    for _ in range(max_tries):
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            temperature=0.5,
            max_tokens=1200,
        )
        text = (res.choices[0].message.content or "").strip()

        insert_after_map = {}
        pattern = r'^INSERT AFTER LINE (END|\d+):\s*([^:：]+)\s*[:：]\s*(.+)$'
        ok_count = {r: 0 for r in roles}

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
            ins_key = ("END" if where.upper() == "END" else int(where))
            insert_after_map.setdefault(ins_key, []).append(f"{who}: {content}")
            ok_count[who] += 1

        if all(ok_count[r] == deficits[r] for r in roles):
            insertion_lines = insert_after_map
            break

    base_lines = clean_script_text(original_script).splitlines()
    out = []
    for idx, line in enumerate(base_lines, 1):
        out.append(line)
        if insertion_lines and idx in insertion_lines:
            out.extend(insertion_lines[idx])

    if insertion_lines and "END" in insertion_lines:
        out.extend(insertion_lines["END"])

    return "\n".join(out)

# ───────── 삭제 전용 함수(가장 덜 중요한 대사 우선) ─────────
def _prune_with_deletions_only(client, original_script: str, roles: List[str], targets: Dict[str, int], max_tries: int = 3) -> str:
    current = _count_lines_by_role(original_script, roles)
    over = {r: max(0, current.get(r, 0) - targets.get(r, current.get(r, 0))) for r in roles}
    if all(v == 0 for v in over.values()):
        return original_script

    base_lines = clean_script_text(original_script).splitlines()
    numbered = [f"{i:04d}: {ln}" for i, ln in enumerate(base_lines, 1)]
    numbered_script = "\n".join(numbered)

    sys = (
        "당신은 초등 연극 대본 편집자입니다. '불필요하거나 중복되는 대사'를 골라 삭제하여 "
        "목표 줄 수로 줄이되, 이야기의 기·승·전·결과 갈등–해결의 흐름은 유지하세요. "
        "기존 대사 문구는 바꾸지 말고, '삭제'만 하세요. 필요하면 짧은 연결 지문(괄호)을 추가할 수 있습니다."
    )

    over_list = "\n".join([f"- {r}: -{over[r]}줄" for r in roles if over[r] > 0])

    format_rule = (
        "출력은 '삭제 지시'만 여러 줄로 주세요. 각 줄은 다음 형식을 반드시 따르세요:\n"
        "DELETE LINE <번호>\n"
        "규칙:\n"
        "- <번호>는 '원본 대본(줄 번호 포함)'의 번호입니다.\n"
        "- 필요 시 흐름 유지를 위해 'INSERT AFTER LINE <번호>: ( ... )' 형태의 짧은 '지문'만 추가할 수 있습니다."
    )

    user = f"""
[원본 대본(줄 번호 포함)]
{numbered_script}

[줄여야 할 개수(인물별)]
{over_list}

요구사항:
1) 각 인물의 대사 수를 목표에 맞도록 '삭제'만 수행
2) 불필요·중복·주제와 무관한 대사를 우선 삭제
3) 원래 기·승·전·결과 갈등–해결 관계 유지
4) 필요 시 매우 짧은 연결 지문(괄호)만 추가 가능
5) 등장인물은 {', '.join(roles)}만 사용
"""

    msg = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    deletions = set()
    insert_map = {}

    for _ in range(max_tries):
        res = client.chat_completions.create if hasattr(client, 'chat_completions') else client.chat.completions.create
        res = res(
            model="gpt-4o-mini",
            messages=msg,
            temperature=0.4,
            max_tokens=1200,
        )
        text = (res.choices[0].message.content or "").strip()

        del_pat = r'^DELETE LINE (\d+)\s*$'
        ins_pat = r'^INSERT AFTER LINE (END|\d+):\s*\((.+)\)\s*$'
        local_del = set()
        local_ins = {}

        for raw in text.splitlines():
            s = raw.strip()
            m1 = re.match(del_pat, s, re.IGNORECASE)
            if m1:
                local_del.add(int(m1.group(1)))
                continue
            m2 = re.match(ins_pat, s, re.IGNORECASE)
            if m2:
                where, content = m2.groups()
                key = ("END" if where.upper() == "END" else int(where))
                local_ins.setdefault(key, []).append(f"({content.strip()})")
                continue

        deletions |= local_del
        for k, v in local_ins.items():
            insert_map.setdefault(k, []).extend(v)
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

# ───────── 페이지 함수 ─────────
def page_script_input():
    st.header("📥 1) 대본 등록")
    val = st.text_area("대본 직접 입력 (형식: 이름: 내용)", height=260, value=st.session_state.get("script_raw", ""))
    if st.button("💾 저장"):
        st.session_state["script_raw"] = val.strip()
        st.success("저장되었습니다!")


def page_feedback_script():
    st.header("🛠️ 2) 대본 피드백 & 완성본")
    st.info("(데모에서는 원본을 그대로 사용합니다)")


def page_role_balancer():
    st.header("⚖️ 3) 대사 수 조절하기 — 자동 하이브리드")

    script = (
        st.session_state.get("script_balanced") or
        st.session_state.get("script_final") or
        st.session_state.get("script_raw", "")
    )
    if not script:
        st.warning("먼저 대본을 입력/생성하세요.")
        return

    roles = extract_roles(script)
    if not roles:
        st.info("‘이름: 내용’ 형식이어야 역할을 인식해요.")
        return

    counts = _count_lines_by_role(script, roles)

    st.subheader("📜 현재 대본")
    st.code(script, language="text", height=360)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 현재 대사 수")
        for r in roles:
            st.write(f"**{r}**: {counts.get(r, 0)}줄")
    with col2:
        st.subheader("🎯 목표 대사 수 (감소/증가 모두 허용)")
        targets: Dict[str, int] = {}
        for r in roles:
            targets[r] = st.number_input(
                f"{r} 목표",
                min_value=0,
                value=counts[r],
                step=1,
                key=f"tgt_{r}"
            )

    st.markdown("---")
    if st.button("🔁 재분배하기 (자동)", use_container_width=True):
        with st.spinner("자연스러운 흐름으로 삭제/추가를 반영 중..."):
            try:
                new_script = script

                # 1) 감소 필요 시: '삭제' 먼저 (가장 덜 중요한 대사 위주)
                need_prune = any(targets[r] < counts.get(r, 0) for r in roles)
                if need_prune:
                    new_script = _prune_with_deletions_only(client, new_script, roles, targets, max_tries=3)

                # 2) 증가 필요 시: 부족분만 '추가'
                after_prune_counts = _count_lines_by_role(new_script, roles)
                need_add = any(targets[r] > after_prune_counts.get(r, 0) for r in roles)
                if need_add:
                    new_script = _augment_with_additions_only(client, new_script, roles, targets, max_tries=3)

                st.session_state["script_balanced"] = new_script
                st.session_state["current_script"] = new_script
                final_counts = _count_lines_by_role(new_script, roles)

                st.success("✅ 적용 완료! 아래 결과를 확인하세요.")
                st.code(new_script, language="text", height=360)
                st.info("최종 줄 수: " + ", ".join([f"{r} {final_counts.get(r,0)}줄" for r in roles]))

            except Exception as e:
                st.error(f"재분배 중 오류: {e}")


def page_stage_kits():
    st.header("🎭 4) 소품·무대·의상")
    st.write("(생략)")


def page_rehearsal_partner():
    st.header("🎙️ 5) AI 대본 연습")
    st.write("(생략)")

# ───────── 사이드바/MAIN ─────────
def main():
    st.set_page_config("연극용의 둥지", "🐉", layout="wide")
    st.title("🐉 연극용의 둥지 — 연극 용을 성장시켜요!")

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "📥 1) 대본 등록"

    pages = {
        "📥 1) 대본 등록": page_script_input,
        "🛠️ 2) 대본 피드백 & 완성본": page_feedback_script,
        "⚖️ 3) 대사 수 조절하기": page_role_balancer,
        "🎭 4) 소품·무대·의상": page_stage_kits,
        "🎙️ 5) AI 대본 연습": page_rehearsal_partner,
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

    pages[sel]()

if __name__ == "__main__":
    main()
