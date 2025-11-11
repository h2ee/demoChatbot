from typing import List, Dict

import streamlit as st
from openai import OpenAI, OpenAIError


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
# 2. OpenAI 텍스트 호출 함수
# ------------------------------
def call_openai_chat(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """텍스트 답변 생성."""
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
        # 크레딧 부족일 때는 모의 답변
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
# 3. 이미지: 아주 단순한 placeholder (picsum)
# ------------------------------
def generate_image_url(prompt: str) -> str:
    """
    OpenAI, Unsplash 모두 쓰지 않고,
    항상 picsum.photos 에서 랜덤 1024x1024 이미지를 사용.
    """
    return "https://picsum.photos/1024"


# ------------------------------
# 4. Streamlit UI
# ------------------------------
def main():
    st.set_page_config(
        page_title="Role-based Creative Chatbot",
        layout="wide",
    )

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        # 각 항목: {"role","content","role_name","image_url"}
        st.session_state.chat_history = []

    # -------- 사이드바 --------
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

    # -------- 메인 두 컬럼 --------
    col_main, col_history = st.columns([2, 1])

    # ===== 왼쪽: 입력 + Latest response =====
    with col_main:
        st.title("🎭 Role-based Creative Chatbot")
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
                            image_url = generate_image_url(clean_input)

                            # 히스토리 추가 (user + assistant)
                            st.session_state.chat_history.append(
                                {
                                    "role": "user",
                                    "content": clean_input,
                                    "role_name": "You",
                                    "image_url": None,
                                }
                            )
                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "role_name": role_name,
                                    "image_url": image_url,
                                }
                            )

        # --- Latest response: 이미지가 텍스트를 밀어내는 float 레이아웃 ---
        if st.session_state.chat_history:
            last = st.session_state.chat_history[-1]
            if last["role"] == "assistant":
                st.subheader("Latest")

                ascii_art = ROLE_DEFINITIONS[last["role_name"]]["ascii"].strip()
                short_desc = ROLE_DEFINITIONS[last["role_name"]]["short"]

                # 마지막 유저 메시지(질문) 찾아서 캡션에 한 줄 요약
                prev_user = None
                for msg in reversed(st.session_state.chat_history[:-1]):
                    if msg["role"] == "user":
                        prev_user = msg["content"]
                        break
                if prev_user:
                    caption_text = (
                        f'"{prev_user[:80]}{"…" if len(prev_user) > 80 else ""}"'
                    )
                else:
                    caption_text = "AI-generated concept image"

                with st.chat_message("assistant"):
                    # 역할 이름 + ASCII 헤더
                    st.markdown(f"**{last['role_name']}**")
                    st.markdown(f"```text\n{ascii_art}\n```")

                    img_url = last.get("image_url")

                    if img_url:
                        # 이미지가 왼쪽에 떠 있고, 텍스트가 오른쪽/아래로 흘러내리는 구조
                        st.markdown(
                            f"""
<div style="overflow:auto; margin-top:0.5rem;">

  <!-- 왼쪽 이미지 카드 -->
  <div style="
      float:left;
      margin-right:24px;
      margin-bottom:8px;
      border-radius:16px;
      border:2px solid #e0e0e0;
      background:#f5f5f5;
      padding:6px;
      max-width:260px;
  ">
    <img src="{img_url}"
         style="width:100%; border-radius:12px; display:block;">

    <div style="
        font-size:0.75rem;
        color:#aaaaaa;
        font-weight:300;
        margin-top:4px;
    ">
      {short_desc} · {caption_text}
    </div>
  </div>

  <!-- 오른쪽 + 아래쪽 텍스트 영역 -->
  <div style="overflow:hidden; font-size:0.95rem; line-height:1.6;">
""",
                            unsafe_allow_html=True,
                        )

                        # 본문 텍스트는 평소처럼 markdown으로
                        st.markdown(last["content"])

                        # float 해제
                        st.markdown(
                            """
  </div>
</div>
<div style="clear:both;"></div>
""",
                            unsafe_allow_html=True,
                        )
                    else:
                        # 이미지 없으면 그냥 텍스트만
                        st.markdown(last["content"])

    # ===== 오른쪽: History bubble view =====
    with col_history:
        st.subheader("Conversation History (bubble view)")

        if not st.session_state.chat_history:
            st.info("아직 대화가 없습니다. 질문을 한 번 해보세요!")
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    ascii_art = ROLE_DEFINITIONS[msg["role_name"]]["ascii"].strip()
                    with st.chat_message("assistant"):
                        # 위쪽: 작은 썸네일 + ASCII 아트
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            if msg.get("image_url"):
                                st.image(msg["image_url"], width=40)
                        with c2:
                            st.markdown(f"**{msg['role_name']}**")
                            st.markdown(f"```text\n{ascii_art}\n```")

                        # 아래쪽: 펼치기에서 전체 답변 + 큰 이미지
                        with st.expander("Show full answer"):
                            if msg.get("image_url"):
                                st.image(msg["image_url"], width=256)
                            st.markdown(msg["content"])

        if st.button("Clear history"):
            st.session_state.chat_history = []


if __name__ == "__main__":
    main()
