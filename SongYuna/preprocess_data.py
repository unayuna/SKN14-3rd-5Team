# data_preprocessor.py (메타데이터 추출 강화 버전)

import os
import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


JSON_DIR = "data_json"
PDF_DIR = 'test_pdf'
OUTPUT_FILE = "processed_data.pkl"


# 1. text splitter 준비
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=['\n\n', '\n', '.', ' ', ''],
    length_function=len,
    is_separator_regex=False,
)

# 2. 임베딩 모델 및 LLM
# embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0.01)
# print("설정 변수 정의 및 초기화 완료!")

chunks = []

def process_json_data():

    print(f"--- '{JSON_DIR}' 폴더의 JSON 파일들을 처리합니다. ---")
    if not os.path.exists(JSON_DIR):
        print(f"[오류] '{JSON_DIR}' 폴더를 찾을 수 없습니다.")
        return None

    for filename in os.listdir(JSON_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(JSON_DIR, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 3. 메타데이터 생성
            print(os.path.splitext(filename)[0])
            question_id = data.get("question_id", os.path.splitext(filename)[0])

            if not question_id:
                print(f"[경고] {filename}에 'question_id'가 없습니다. 파일명을 기반으로 생성합니다.")
                question_id = filename.replace(".json", "")

            # # [핵심 수정] 메타데이터는 파일명이 아닌, question_id를 기준으로 파싱합니다.
            # # 예: "2023_서강대_1" -> parts = ['2023', '서강대', '1']
            # parts = filename.replace(".json", "").split('_')
            
            # # 메타데이터 생성 시, 예외 상황에 대한 방어 코드를 추가합니다.
            # # 파일명 구조에 따라 유연하게 대처
            # university = parts[0] if len(parts) > 0 else "알수없음"
            # year = parts[1] if len(parts) > 1 else "알수없음"
            # number = parts[2] if len(parts) > 2 else "기타"

            # base_metadata = {
            #     "question_id": question_id,
            #     "university": university,
            #     "year": year,
            #     "number": number
            # }

            content_map = {
                "출제의도": data.get("intended_purpose"),
                "채점기준": data.get("grading_criteria"),
                "모범답안": data.get("sample_answer")
            }
            
            current_docs = []
            for content_type, content_text in content_map.items():
                if content_text:
                    doc = Document(
                        page_content=content_text,
                        metadata={
                            "question_id": question_id,
                            "content_type": content_type,
                        }
                    )
                    current_docs.append(doc)
                print(f"{filename} 처리 완료. (ID: {question_id})")
            
            # 4. chunk
            current_chunk = splitter.split_documents(current_docs)
            chunks.extend(current_chunk) # extend 사용!

    if not current_docs:
        print("[경고] 처리할 문서가 하나도 없습니다.")
        return None


#     with open(OUTPUT_FILE, 'wb') as f:
#         pickle.dump(current_docs, f)
#     print(f"\n🎉 데이터 전처리 완료! 총 {len(current_docs)}개의 문서 조각이 생성되었습니다.")
#     return current_docs

# if __name__ == '__main__':
#     processed_data = process_json_data()
#     if processed_data:
#         print("\n[샘플 데이터 확인]")
#         for doc in processed_data[:5]:
#             print(doc, "\n" + "-"*30)