import json, os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

def load_json_to_documents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid = data.get("question_id", "")
    sample = data.get("sample_answer", "")
    purpose = data.get("intended_purpose", "")
    grading = data.get("grading_criteria", "")

    content = f"[모범답안]\n{sample.strip()}\n\n[출제의도]\n{purpose.strip()}\n\n[채점기준]\n{grading.strip()}"
    
    metadata = {"question_id" : qid,
                "sample_answer" : sample,
                "intended_purpose" : purpose,
                "grading_criteria" : grading}
    document = Document(page_content=content, metadata=metadata)
    

    return [document]

def build_faiss_index(json_folder, index_save_path):
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    documents = []
    for f in json_files:
        documents.extend(load_json_to_documents(os.path.join(json_folder, f)))
    embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
    faiss_index = FAISS.from_documents(documents=documents, embedding=embeddings)
    faiss_index.save_local(index_save_path)
    print(f"✅ Saved FAISS index to: {index_save_path}")

if __name__ == "__main__":
    build_faiss_index(json_folder='./01_data_preprocessing/json', index_save_path='./01_data_preprocessing/faiss')
