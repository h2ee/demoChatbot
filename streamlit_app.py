# streamlit_app.py
# Role-based Creative Chatbot using OpenAI API

import os
from typing import List, Dict

import streamlit as st
from openai import OpenAI, OpenAIError

# ------------------------------
# 1. Role 정의
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
    },
    "Dance Instructor 💃": {
        "short": "Suggests movement, rhythm, expression",
        "system_prompt": (
            "You are a contemporary dance instructor. You think in terms of movement, "
            "rhythm, body weight, breath, and expression. When you answer, give concrete "
            "movement ideas and describe how the body should feel."
        ),
        "example": "How can I express sadness through movement?",
    },
    "Fashion Stylist 👗": {
        "short": "Explains color trends, materials, silhouette",
        "system_prompt": (
            "You are a professional fashion stylist. Give advice about silhouettes, "
            "textures, materials, color harmony, and styling details. Imagine you are "
            "preparing looks for a photoshoot or red carpet."
        ),
        "example": "What style fits a confident personality?",
    },
    "Acting Coach 🎭": {
        "short": "Teaches emotion delivery, scene breakdown",
        "system_prompt": (
            "You are an acting coach. Help performers explore emotion, subtext, and "
            "physicality. When you answer, break down the scene beat by beat and give "
            "specific exercises or line readings."
        ),
        "example": "How to express fear naturally on stage?",
    },
    "Art Curator 🖼️": {
        "short": "Interprets artwork, connects with data",
        "system_prompt": (
            "You are a museum art curator. Interpret artworks in terms of composition, "
            "color, symbolism, and historical context. Connect visual elements to ideas, "
            "emotions, and cultural references."
        ),
        "example": "How does this composition convey emotion?",
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
            model=model,  # 예: "gpt-4.1-mini"
            messages=messages,
        )
    except OpenAIError as e:
        # Streamlit 쪽에서 그대로 보여줄 수 있도록 예외를 다시 던짐
        raise RuntimeError(f"OpenAI API error: {e}") from e

    return completion.choices[0].message.content.strip()


# ------------------------------
# 3. Streamlit UI
# ------------------------------
def main():
    st.set_page_config(
        page_title="Role-based Creative Chatbot",
        layout="wide",
    )

    # 세션 상태 초기화 (채팅 히스토리)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 각 항목: {"role": "user"/"assistant", "content": "..."}

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

    # -------- 메인 영역 --------
    col_main, col_history = st.columns([2, 1])

    with col_main:
        st.title("🎭 Role-based Creative Chatbot")
        st.write("Select a creative role on the left and ask your question below.")

        example_text = ROLE_DEFINITIONS[role_name]["example"]
        user_input = st.text_area(
            "Enter your question or idea:",
            value=f"e.g., {example_text}",
            height=120,
        )

        if st.button("Generate Response"):
            if not api_key:
                st.error("먼저 왼쪽에서 OpenAI API Key를 입력하세요.")
            else:
                with st.spinner("Thinking as " + role_name + "..."):
                    try:
                        # placeholder 예시 문장을 그대로 두고 버튼 누르면, 실제 입력으로 인식 안 되도록 처리
                        clean_input = (
                            "" if user_input.strip().startswith("e.g.,") else user_input.strip()
                        )
                        if not clean_input:
                            st.warning("질문을 입력한 뒤 버튼을 눌러주세요.")
                        else:
                            answer = call_openai_chat(
                                api_key=api_key,
                                model=model_name,
                                system_prompt=role_info["system_prompt"],
                                user_message=clean_input,
                                history=st.session_state.chat_history,
                            )

                            # 히스토리에 추가
                            st.session_state.chat_history.append(
                                {"role": "user", "content": clean_input}
                            )
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer}
                            )

                    except RuntimeError as e:
                        st.error(str(e))

        # 마지막 응답 보여주기
        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            if last_msg["role"] == "assistant":
                st.markdown("### 💡 Latest response")
                st.markdown(last_msg["content"])

    # -------- 오른쪽: 대화 히스토리 --------
    with col_history:
        st.subheader("Conversation History")

        if not st.session_state.chat_history:
            st.info("아직 대화가 없습니다. 질문을 한 번 해보세요!")
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**🧑 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 {role_name}:** {msg['content']}")
                st.markdown("---")

        if st.button("Clear history"):
            st.session_state.chat_history = []


if __name__ == "__main__":
    main()
