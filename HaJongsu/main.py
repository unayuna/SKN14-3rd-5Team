# main.py (드롭다운 순서 및 한글 이름 표시 최종 버전)

import streamlit as st
import re
import difflib
import pandas as pd

from ocr_processor import OCRProcessor
from essay_grader import EssayGrader

@st.cache_resource
def load_system():
    print("시스템 로딩 시작...")
    grader = EssayGrader()
    # meta_list = [doc.metadata for doc in grader.documents]
    # meta_df = pd.DataFrame(meta_list)
    # universities = sorted(meta_df['university'].unique())
    
    # FAISS 인덱스에서 문서 메타데이터 추출
    all_docs = grader.vector_db.docstore._dict.values()
    meta_list = [doc.metadata for doc in all_docs]
    meta_df = pd.DataFrame(meta_list)
    universities = sorted(meta_df['university'].unique())    
    print("시스템 로딩 완료!")
    return grader, meta_df, universities

@st.cache_data
def load_uni_name_map():
    uni_name_map = {}
    try:
        with open("대학_국문_영문.txt", 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' - ')
                if len(parts) == 2:
                    kor_name, eng_name = parts[0], parts[1]
                    uni_name_map[eng_name] = kor_name
    except FileNotFoundError:
        print("[경고] '대학_국문_영문.txt' 파일을 찾을 수 없습니다.")
    return uni_name_map

@st.cache_resource
def load_ocr():
    return OCRProcessor()

def display_correction_with_diff(student_answer, model_answer, correction_result):
    # ... (시각화 함수 내용은 이전과 동일)
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


def main():
    st.set_page_config(page_title="AI 논술 첨삭 멘토", layout="wide")
    
    with st.spinner('AI 멘토를 준비하는 중입니다...'):
        grader, meta_df, universities = load_system()
        uni_name_map = load_uni_name_map()
        ocr_processor = load_ocr()

    if 'last_question_id' not in st.session_state:
        st.session_state.last_question_id = None

    with st.sidebar:
        st.header("📝 문제 선택")
        
        # [핵심 수정] 대학 이름 번역 로직 추가 및 순서 바로잡기
        # 1. 학교 선택 (한글 이름으로 보여주기)
        kor_universities = sorted([uni_name_map.get(uni, uni) for uni in universities])
        selected_kor_uni = st.selectbox("1. 학교를 선택하세요.", kor_universities)
        
        # 한글 이름을 다시 영문으로 변환 (내부 처리용)
        eng_uni_map = {v: k for k, v in uni_name_map.items()}
        selected_eng_uni = eng_uni_map.get(selected_kor_uni, selected_kor_uni)

        # 2. 년도 선택
        available_years = sorted(meta_df[meta_df['university'] == selected_eng_uni]['year'].unique(), reverse=True)
        selected_year = st.selectbox("2. 응시 년도를 선택하세요.", available_years)
        
        # 3. 문항 선택
        available_nums = sorted(meta_df[(meta_df['university'] == selected_eng_uni) & (meta_df['year'] == selected_year)]['number'].unique())
        selected_num = st.selectbox("3. 문항을 선택하세요.", available_nums)
        
        st.divider()
        
        # question_id는 영문으로 생성
        question_id = f"{selected_eng_uni}_{selected_year}_{selected_num}" 
        # 화면 표시는 한글로
        display_text = f"{selected_kor_uni} {selected_year}학년도 {selected_num}번 문항"
        st.info(f"**선택된 문제:**\n{display_text}")

        st.header("📄 답안 제출")
        uploaded_file = st.file_uploader("여기에 손글씨 답안 사진을 올려주세요.", type=['jpg', 'jpeg', 'png'])
        submit_button = st.button("첨삭 시작하기", type="primary")

    st.title("✍️ AI 논술 첨삭 멘토봇")
    
    is_new_problem = (question_id != st.session_state.last_question_id)
    with st.expander(f"[{display_text}] 문제 확인 (채점 기준)", expanded=is_new_problem):
        problem_criteria = grader.get_document_content(question_id, "채점기준")
        st.info(problem_criteria)
    
    # ... (이하 로직 동일)
    
    st.markdown("---")
    result_placeholder = st.empty()

    if submit_button and uploaded_file:
        with st.spinner("AI 멘토가 학생 답안을 열심히 읽고 있어요... 잠시만 기다려주세요! 🧐"):
            try:
                image_bytes = uploaded_file.getvalue()
                student_answer_text = ocr_processor.process_image(image_bytes)
                
                OCR_ERROR_MESSAGES = ["OCR 처리 중 문제가 발생했습니다", "이미지에서 텍스트를 추출하지 못했습니다"]
                if any(msg in student_answer_text for msg in OCR_ERROR_MESSAGES):
                    st.error(student_answer_text)
                    return
                
                model_answer_text = grader.get_document_content(question_id, "모범답안")
                correction_result = grader.grade_essay(question_id, student_answer_text)

                with result_placeholder.container():
                    display_correction_with_diff(student_answer_text, model_answer_text, correction_result)

            except Exception as e:
                st.error(f"첨삭 과정에서 예상치 못한 오류가 발생했습니다: {e}")
    
    elif submit_button and not uploaded_file:
        st.warning("파일을 먼저 업로드해주세요!")

    st.session_state.last_question_id = question_id

if __name__ == "__main__":
    main()