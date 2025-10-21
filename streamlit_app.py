import os
import streamlit as st
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chains.retrieval_qa import RetrievalQA

# MODEL AYARLARI
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DB_DIR = "./chroma_db"
# Lütfen KONTROL EDİN: PDF_PATH değerini kendi dosya adınızla değiştirin!
import os
import streamlit as st
from functools import lru_cache

# ... diğer importlar ...

# MODEL AYARLARI
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DB_DIR = "./chroma_db"

# GÜVENLİ PDF YOLU TANIMLAMA (Streamlit Cloud Uyumlu)
current_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(current_dir, "veri_seti", "ucak_kontrol_sistemleri.pdf")

@st.cache_resource
def setup_rag_system():
    # 1. API Anahtarını Alma
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("HATA: GEMINI_API_KEY bulunamadı. Lütfen Terminalde export komutunu çalıştırın.")
        return None

    # 2. Veri Yükleme
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
    except Exception as e:
        st.error(f"HATA: PDF yüklenirken bir sorun oluştu. Dosya yolunu kontrol edin: {e}")
        return None

    # 3. Metin Parçalama (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Embedding Modeli ve Vektör Veritabanı Oluşturma
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    retriever = vector_store.as_retriever()

    # 5. RAG Zincirini Kurma (llm tanımı burada önce gelmeli)
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, 
        temperature=0.1,
        google_api_key=api_key
    ) 
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever
    )
    return qa_chain

# 6. Streamlit Arayüzü
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("Uçak Kontrol Sistemleri RAG Chatbot 🤖")
    st.caption("Veri Kaynağı: Uçak Kontrol Sistemleri PDF'i")

    qa_chain = setup_rag_system()
    
    if qa_chain is None:
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Merhaba! Uçuş kontrol sistemleri hakkında ne sormak istersiniz?"}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Cevap aranıyor..."):
            try:
                yanit = qa_chain.run(prompt)
                st.session_state["messages"].append({"role": "assistant", "content": yanit})
                st.chat_message("assistant").write(yanit)
            except Exception as e:
                hata_mesaji = f"Bir hata oluştu: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": hata_mesaji})
                st.chat_message("assistant").write(hata_mesaji)

if __name__ == "__main__":
    main()