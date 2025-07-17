import streamlit as st
import re
import difflib

def display_correction_with_diff(student_answer, model_answer, correction_result):
    st.subheader("🤖 AI 멘토 첨삭 결과", divider='rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.info("👨‍🎓 학생 답안 (OCR 변환)")
        st.text_area("학생 답안 내용", value=student_answer, height=400, disabled=True, key="student_answer_area")
    with col2:
        st.warning("✅ 모범 답안")
        st.text_area("모범 답안 내용", value=model_answer, height=400, disabled=True, key="model_answer_area")
    st.markdown("---")
    st.info("✨ 논리왕 김멘토's 코멘트")
    pattern = r"학생 원문:\s*(.*?)\s*수정 제안:\s*(.*?)(?=\n\*\*\[|학생 원문:|\Z)"
    suggestions = re.findall(pattern, correction_result, re.DOTALL)
    main_correction = re.split(r'(\*\*\[이렇게 바꿔보세요)', correction_result)[0]
    st.markdown(main_correction)
    if suggestions:
        st.markdown("#### 💡 이렇게 바꿔보세요")
        for i, (original, suggestion) in enumerate(suggestions):
            original = original.strip().strip('"')
            suggestion = suggestion.strip().strip('"')
            with st.expander(f'수정 제안 #{i+1}: "{original}"', expanded=True):
                st.markdown(f'**- 원본:** {original}')
                d = difflib.Differ()
                diff_words = list(d.compare(original.split(), suggestion.split()))
                diff_html = ""
                for word in diff_words:
                    if word.startswith('+ '):
                        diff_html += f' <span style="background-color: #d4edda; padding: 2px 0; border-radius: 3px;">{word[2:]}</span>'
                    elif word.startswith('- '):
                        diff_html += f' <span style="background-color: #f8d7da; padding: 2px 0; border-radius: 3px; text-decoration: line-through;">{word[2:]}</span>'
                    else:
                        diff_html += f' {word[2:]}'
                st.markdown(f'**- 제안:**{diff_html.strip()}', unsafe_allow_html=True)
                st.markdown('---')
                st.markdown(f'**- 수정된 문장:**')
                st.success(suggestion)
    elif "**[이렇게 바꿔보세요" in correction_result:
         st.warning("AI가 수정 제안을 생성했지만, 형식이 맞지 않아 표시할 수 없습니다. 프롬프트를 확인해주세요.")
