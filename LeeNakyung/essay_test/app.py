import streamlit as st
from rag import AnswerChatRAG
from rag import load_data, build_vectorstore, generate_feedback
from langchain.chat_models import ChatOpenAI
from rag import OPENAI_API_KEY
from langchain_core.output_parsers import StrOutputParser


st.title("📝 RAG 기반 논술 첨삭 도우미")

question_bank = load_data()

# 필터 옵션 생성
schools = sorted(set(d["school"] for d in question_bank.values()))
selected_school = st.selectbox("학교 선택", schools)

years = sorted(set(d["year"] for d in question_bank.values() if d["school"] == selected_school))
selected_year = st.selectbox("연도 선택", years)

qnums = sorted(set(d["qnum"] for d in question_bank.values()
                   if d["school"] == selected_school and d["year"] == selected_year))
selected_qnum = st.selectbox("문항 번호", qnums)

# question_id 조합
selected_qid = f"{selected_year}_{selected_school}_{selected_qnum}"
selected_entry = question_bank.get(selected_qid)

if selected_entry:
    data = selected_entry["data"]
    question_id = data["question_id"]
    intended_purpose = data["intended_purpose"]
    grading_criteria = data["grading_criteria"]
    sample_answer = data["sample_answer"]

    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
    parser = StrOutputParser()

    # 문항 정보 출력
    with st.expander("📘 문항 정보 열기"):
        st.markdown(f"**문항 ID:** `{question_id}`")
        st.markdown("**출제 의도:**")
        st.write(intended_purpose)
        st.markdown("**채점 기준:**")
        st.write(grading_criteria)
        st.markdown("**예시 답안:**")
        st.write(sample_answer)

    # 사용자 답안 상태 유지
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""
    if "feedback_button_clicked" not in st.session_state:
        st.session_state.feedback_button_clicked = False
    if "feedback_result" not in st.session_state:
        st.session_state.feedback_result = ""

    # 사용자 입력
    user_answer = st.text_area("✏️ 나의 답안 입력", height=300)
    st.session_state.user_answer = user_answer

    if st.button("📊 첨삭 받기") and user_answer.strip():
        st.session_state.feedback_button_clicked = True
        with st.spinner("AI가 첨삭 중입니다..."):
            llm = ChatOpenAI(model="gpt-4", temperature=0)
            vectorstore = build_vectorstore(question_bank)
            result = generate_feedback(
                llm,
                question_id=data["question_id"],
                grading_criteria=data["grading_criteria"],
                sample_answer=data["sample_answer"],
                user_answer=user_answer,
                vectorstore=vectorstore
            )
            st.subheader("📋 첨삭 결과")
            st.markdown(result)

    st.markdown("---")
    st.subheader("🧠 내 답변 기반 Q&A 챗봇")

    # if user_answer.strip():
    #     vectorstore = build_vectorstore(user_answer)
    #     user_q = st.chat_input("내 답변에 대해 궁금한 점을 물어보세요!")
    #
    #     if "answer_chat_history" not in st.session_state:
    #         st.session_state.answer_chat_history = []
    #
    #     # 이전 채팅 기록 출력
    #     for msg in st.session_state.answer_chat_history:
    #         with st.chat_message(msg["role"]):
    #             st.markdown(msg["content"])
    #
    #     if user_q:
    #         st.session_state.answer_chat_history.append({"role": "user", "content": user_q})
    #         with st.chat_message("user"):
    #             st.markdown(user_q)
    #
    #         chain = build_chain(vectorstore, openai_api_key=OPENAI_API_KEY)
    #
    #         # 응답 생성
    #         with st.chat_message("assistant"):
    #             with st.spinner("답변 생성 중..."):
    #                 output = chain.invoke({"question": user_q})
    #                 st.markdown(output)
    #                 st.session_state.answer_chat_history.append({"role": "assistant", "content": output})

    from rag import AnswerChatRAG

    if user_answer.strip():
        rag = AnswerChatRAG(user_answer, openai_api_key=OPENAI_API_KEY)
        user_q = st.chat_input("내 답변에 대해 궁금한 점을 물어보세요!")

        if "answer_chat_history" not in st.session_state:
            st.session_state.answer_chat_history = []

        # 이전 메시지 출력
        for msg in st.session_state.answer_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_q:
            st.session_state.answer_chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    output = rag.invoke(user_q)
                    st.markdown(output)
                    st.session_state.answer_chat_history.append({"role": "assistant", "content": output})

    else:
        st.info("먼저 답안을 입력하세요.")
