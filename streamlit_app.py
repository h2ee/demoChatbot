# streamlit_app.py
# Role-based Creative Chatbot (Bubble UI + EmojiHub avatar + compact history)

import os
from typing import List, Dict

import requests
import streamlit as st
from openai import OpenAI, OpenAIError


# ------------------------------
# 0. EmojiHub (Avatar용 사람 이모지)
# ------------------------------
EMOJI_API_BASE = "https://emojihub.yurace.pro/api"


def get_avatar_emoji() -> str:
    """
    EmojiHub에서 'smileys and people' 카테고리의 랜덤 이모지 하나 가져오기.
    HTML 코드로 리턴해서 그대로 렌더링.
    실패하면 기본 이모지 사용.
    """
    try:
        # EmojiHub docs 기준: /random/category/smileys-and-people
        resp = requests.get(
            f"{EMOJI_API_BASE}/random/category/smileys-and-people", timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        html_codes = data.get("htmlCode") or []
        if html_codes:
            return "".join(html_codes)
    except Exception:
        pass
    # 실패 시 기본 사람 이모지
    return "🧑‍🎨"


# ------------------------------
# 1. Role 정의 + ASCII 아트
# ------------------------------
ROLE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Video Director 🎬": {
        "short": "Analyzes mood, camera angle, lighting",
        "system_prompt": (
            "You are a professional film director. Always analyze ideas in terms of "
            "visual storytelling — use camera movement, lighting, framing, editing, "
            "and emotional tone to explain your thoughts. Describe concepts as if "
            "you are planning a film scene or sequence."
        ),
        "example": "How can I shoot a dream sequence?",
        "ascii": r"""
  🎬 VIDEO DIRECTOR
  ─────────────────────
  [CAM]  ───►   [SCENE]
  angles · lighting · mood
""",
    },
    "Dance Instructor 💃": {
        "short": "Suggests movement, rhythm, expression",
        "system_prompt": (
            "You are a contemporary dance instructor. You think in terms of movement, "
            "rhythm, body weight, breath, and expression. When you answer, give concrete "
            "movement ideas and describe how the body should feel."
        ),
        "example": "How can I express sadness through movement?",
        "ascii": r"""
  💃 DANCE INSTRUCTOR
  ─────────────────────
  1·2·3·4 · steps & flow
  body · breath · emotion
""",
    },
    "Fashion Stylist 👗": {
        "short": "Explains color trends, materials, silhouette",
        "system_prompt": (
            "You are a professional fashion stylist. Give advice about silhouettes, "
            "textures, materials, color harmony, and styling details. Imagine you are "
            "preparing looks for a photoshoot or red carpet."
        ),
        "example": "What style fits a confident personality?",
        "ascii": r"""
  👗 FASHION STYLIST
  ─────────────────────
  color · fabric · shape
  runway-ready outfits
""",
    },
    "Acting Coach 🎭": {
        "short": "Teaches emotion delivery, scene breakdown",
        "system_prompt": (
            "You are an acting coach. Help performers explore emotion, subtext, and "
            "physicality. When you answer, break down the scene beat by beat and give "
            "specific exercises or line readings."
        ),
        "example": "How to express fear naturally on stage?",
        "ascii": r"""
  🎭 ACTING COACH
  ─────────────────────
  beats · objectives · subtext
  voice & body in sync
""",
    },
    "Art Curator 🖼️": {
        "short": "Interprets artwork, connects with data",
        "system_prompt": (
            "You are a museum art curator. Interpret artworks in terms of composition, "
            "color, symbolism, and historical context. Connect visual elements to ideas, "
            "emotions, and cultural references."
        ),
        "example": "How does this composition convey emotion?",
        "ascii": r"""
  🖼️ ART CURATOR
  ─────────────────────
  lines · color · symbols
  stories behind the frame
""",
    },
}


# ------------------------------
# 2. OpenAI 호출 함수
# ------------------------------
def call_openai_chat(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """
    OpenAI Chat Completions API를 호출해서 답변 텍스트만 반환.
    history는 [{"role": "user"/"assistant", "content": "..."}] 리스트.
    """
    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content.strip()
    except OpenAIError as e:
        # quota 에러일 때는 모의 답변으로 대체
        if "insufficient_quota" in str(e):
            return (
                "[Mock response]\n"
                "지금은 OpenAI 크레딧이 부족해서 실제 모델을 호출할 수 없습니다.\n"
                "대신, 이 역할이라면 이런 식으로 생각해 볼 수 있어요:\n\n"
                "- 장면의 감정, 구도, 리듬을 분리해서 하나씩 분석해 보기\n"
                "- 관객이 느끼길 원하는 감정을 먼저 정하고, 거기에 맞게 요소를 조합하기\n"
                "- 실제 촬영/퍼포먼스 전에 짧은 스케치를 여러 개 만들어 비교해 보기\n"
            )
        raise RuntimeError(f"OpenAI API error: {e}") from e


# ------------------------------
# 3. 말풍선 UI용 CSS
# ------------------------------
def inject_chat_css():
    st.markdown(
        """
<style>
.chat-container {
  display: flex;
  margin-bottom: 0.5rem;
}

.chat-bubble {
  padding: 0.6rem 0.9rem;
  border-radius: 12px;
  max-width: 100%;
  word-wrap: break-word;
  font-size: 0.95rem;
}

.chat-bubble-inner {
  display: flex;
  gap: 0.6rem;
  align-items: flex-start;
}

.chat-avatar {
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.7rem;
}

.chat-content {
  flex: 1;
}

.chat-user {
  justify-content: flex-end;
}

.chat-user .chat-bubble {
  background-color: #DCF8C6;
  border-bottom-right-radius: 2px;
}

.chat-bot {
  justify-content: flex-start;
}

.chat-bot .chat-bubble {
  background-color: #F1F0F0;
  border-bottom-left-radius: 2px;
}

.chat-role-header {
  font-size: 0.8rem;
  color: #555;
  margin-bottom: 0.15rem;
  font-weight: 600;
}

.chat-ascii {
  font-family: "Courier New", monospace;
  font-size: 0.7rem;
  white-space: pre;
  margin-bottom: 0.25rem;
  color: #444;
}

/* history 영역: 봇 말풍선 높이 고정 + overflow hidden
   (아스키 아트 3~4줄은 보이도록 넉넉하게 설정) */
.chat-history-bot .chat-bubble {
  max-height: 130px;
  overflow: hidden;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------
# 4. 말풍선 렌더 함수들
# ------------------------------
def render_user_bubble(text: str):
    st.markdown(
        f"""
<div class="chat-container chat-user">
  <div class="chat-bubble">
    {text}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_bot_bubble_main(text: str, role_name: str, ascii_art: str, emoji_html: str):
    """메인 영역의 최신 답변용 (전체 텍스트 다 보여줌)."""
    st.markdown(
        f"""
<div class="chat-container chat-bot">
  <div class="chat-bubble">
    <div class="chat-bubble-inner">
      <div class="chat-avatar">{emoji_html}</div>
      <div class="chat-content">
        <div class="chat-role-header">{role_name}</div>
        <div class="chat-ascii">{ascii_art}</div>
        <div>{text}</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_bot_bubble_history_preview(role_name: str, ascii_art: str, emoji_html: str):
    """
    히스토리 뷰에서 사용하는 '압축 버전' 말풍선.
    - 아바타 + Role header + ASCII 아트만 보임
    - 실제 긴 텍스트는 아래 expander에 따로 표시
    """
    st.markdown(
        f"""
<div class="chat-container chat-bot chat-history-bot">
  <div class="chat-bubble">
    <div class="chat-bubble-inner">
      <div class="chat-avatar">{emoji_html}</div>
      <div class="chat-content">
        <div class="chat-role-header">{role_name}</div>
        <div class="chat-ascii">{ascii_art}</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------
# 5. Streamlit UI
# ------------------------------
def main():
    st.set_page_config(
        page_title="Role-based Creative Chatbot",
        layout="wide",
    )
    inject_chat_css()

    # 세션 상태 초기화 (채팅 히스토리: role_name, avatar까지 저장)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # {"role", "content", "role_name", "avatar"}

    # -------- 사이드바: API & Role 설정 --------
    with st.sidebar:
        st.title("🔑 API & Role Settings")

        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="절대로 이 키를 깃허브에 커밋하지 마세요!",
        )

        model_name = st.selectbox(
            "Model",
            ["gpt-4.1-mini", "gpt-4.1"],
            index=0,
            help="과제용이면 작은 모델(gpt-4.1-mini)로 충분해요.",
        )

        role_name = st.selectbox(
            "Choose a role",
            list(ROLE_DEFINITIONS.keys()),
            index=0,
        )
        role_info = ROLE_DEFINITIONS[role_name]

        st.markdown("**Role description**")
        st.info(role_info["short"])

        st.markdown("**System prompt used for this role**")
        st.write(role_info["system_prompt"])

        st.markdown("---")
        st.caption("Built for ‘Art & Advanced Big Data’ – role-based chatbot demo")

    # -------- 메인 레이아웃 --------
    col_main, col_history = st.columns([2, 1])

    with col_main:
        st.title("🎭 Talk with Chatbot")
        st.write("Select a creative role on the left and ask your question below.")

        example_text = role_info["example"]
        user_input = st.text_area(
            "Enter your question or idea:",
            value=f"e.g., {example_text}",
            height=120,
        )

        if st.button("Generate Response"):
            if not api_key:
                st.error("먼저 왼쪽에서 OpenAI API Key를 입력하세요.")
            else:
                clean_input = (
                    "" if user_input.strip().startswith("e.g.,") else user_input.strip()
                )
                if not clean_input:
                    st.warning("질문을 입력한 뒤 버튼을 눌러주세요.")
                else:
                    with st.spinner(f"Thinking as {role_name}..."):
                        try:
                            # 이전 히스토리에서 role, content만 꺼내서 전달
                            history_for_api = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_history
                            ]
                            answer = call_openai_chat(
                                api_key=api_key,
                                model=model_name,
                                system_prompt=role_info["system_prompt"],
                                user_message=clean_input,
                                history=history_for_api,
                            )
                        except RuntimeError as e:
                            st.error(str(e))
                            answer = None

                        if answer is not None:
                            # 아바타 이모지 생성
                            avatar = get_avatar_emoji()

                            # 히스토리에 저장
                            st.session_state.chat_history.append(
                                {
                                    "role": "user",
                                    "content": clean_input,
                                    "role_name": "You",
                                    "avatar": "",
                                }
                            )
                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "role_name": role_name,
                                    "avatar": avatar,
                                }
                            )

        # 가장 최근 응답을 메인 영역에도 크게 보여주기
        if st.session_state.chat_history:
            last = st.session_state.chat_history[-1]
            if last["role"] == "assistant":
                st.markdown("### 💡 Latest response")
                render_bot_bubble_main(
                    last["content"],
                    last["role_name"],
                    ROLE_DEFINITIONS[last["role_name"]]["ascii"],
                    last.get("avatar", "🧑‍🎨"),
                )

    # -------- 오른쪽: 전체 대화 히스토리 (compact bubble + expander) --------
    with col_history:
        st.subheader("History")

        if not st.session_state.chat_history:
            st.info("아직 대화가 없습니다. 질문을 한 번 해보세요!")
        else:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    render_user_bubble(msg["content"])
                else:
                    role_name_msg = msg["role_name"]
                    ascii_art = ROLE_DEFINITIONS[role_name_msg]["ascii"]
                    avatar = msg.get("avatar", "🧑‍🎨")

                    # 1) 말풍선에는 아바타 + Role header + ASCII 아트까지만
                    render_bot_bubble_history_preview(
                        role_name_msg,
                        ascii_art,
                        avatar,
                    )

                    # 2) 실제 긴 답변은 펼치기(expander) 안에
                    with st.expander("Show full answer"):
                        st.markdown(msg["content"])

        if st.button("Clear history"):
            st.session_state.chat_history = []


if __name__ == "__main__":
    main()
