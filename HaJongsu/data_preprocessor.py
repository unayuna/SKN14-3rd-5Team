# data_preprocessor.py (메타데이터 추출 강화 버전)

import os
import json
from langchain_core.documents import Document
import pickle

JSON_DATA_DIR = "./01_data_preprocessing/json"
OUTPUT_FILE = "./01_data_preprocessing/faiss/preprocessed_documents.pkl"

def process_json_data():
    all_documents = []
    print(f"--- '{JSON_DATA_DIR}' 폴더의 JSON 파일들을 처리합니다. ---")
    if not os.path.exists(JSON_DATA_DIR):
        print(f"[오류] '{JSON_DATA_DIR}' 폴더를 찾을 수 없습니다.")
        return None

    for filename in os.listdir(JSON_DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(JSON_DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # [핵심 수정] question_id는 JSON 파일 안의 값을 최우선으로 사용합니다.
            question_id = data.get("question_id")
            if not question_id:
                print(f"[경고] {filename}에 'question_id'가 없습니다. 파일명을 기반으로 생성합니다.")
                question_id = filename.replace(".json", "")

            # [핵심 수정] 메타데이터는 파일명이 아닌, question_id를 기준으로 파싱합니다.
            # 예: "2023_서강대_1" -> parts = ['2023', '서강대', '1']
            parts = filename.replace(".json", "").split('_')
            
            # 메타데이터 생성 시, 예외 상황에 대한 방어 코드를 추가합니다.
            # 파일명 구조에 따라 유연하게 대처
            university = parts[0] if len(parts) > 0 else "알수없음"
            year = parts[1] if len(parts) > 1 else "알수없음"
            number = parts[2] if len(parts) > 2 else "기타"

            base_metadata = {
                "question_id": question_id,
                "university": university,
                "year": year,
                "number": number
            }

            content_map = {
                "출제의도": data.get("intended_purpose"),
                "채점기준": data.get("grading_criteria"),
                "모범답안": data.get("sample_answer")
            }
            
            for content_type, content in content_map.items():
                if content:
                    doc_metadata = base_metadata.copy()
                    doc_metadata["source_type"] = content_type
                    all_documents.append(Document(page_content=content, metadata=doc_metadata))
            print(f"✅ {filename} 처리 완료. (ID: {question_id})")

    if not all_documents:
        print("[경고] 처리할 문서가 하나도 없습니다.")
        return None

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_documents, f)
    print(f"\n🎉 데이터 전처리 완료! 총 {len(all_documents)}개의 문서 조각이 생성되었습니다.")
    return all_documents

if __name__ == '__main__':
    processed_data = process_json_data()
    if processed_data:
        print("\n[샘플 데이터 확인]")
        for doc in processed_data[:5]:
            print(doc, "\n" + "-"*30)