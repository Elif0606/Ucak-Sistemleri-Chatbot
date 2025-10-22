import os
import streamlit as st
from functools import lru_cache

# LangChain Çekirdek ve Topluluk Bileşenleri
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Yeni LCEL (Expression Language) Zincir Fonksiyonları
# İki fonksiyonu da doğrudan topluluk paketinin kökünden çekmeyi deniyoruz.
from langchain_community.chains import create_stuff_documents_chain
from langchain_community.chains import create_retrieval_chain

# --- SABİT AYARLAR ---
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "embedding-001"
VECTOR_DB_DIR = "./chroma_db"

# GÜVENLİ PDF YOLU TANIMLAMA
current_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(current_dir, "veri_seti", "ucak_kontrol_sistemleri.pdf")

# API Anahtarını Streamlit Secrets'tan güvenli bir şekilde al
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    # Pydantic v1 hatasını atlamak için ortam değişkenini zorla tanımla
    os.environ["GEMINI_API_KEY"] = API_KEY 
except KeyError:
    st.error("HATA: GEMINI_API_KEY Streamlit Secrets'ta tanımlı değil! Lütfen secrets.toml dosyasını kontrol edin.")
    API_KEY = None

# Gömme fonksiyonunu (Embedding Function) sadece API anahtarı varsa oluştur
if API_KEY:
    EMBEDDING_FUNCTION = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
else:
    EMBEDDING_FUNCTION = None


@st.cache_resource
def setup_rag_system():
    if EMBEDDING_FUNCTION is None:
        return None

    # 1. LLM ve Embedding Modelini Tanımlama
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,  
        temperature=0.1,
        # API anahtarı zaten os.environ'da olduğu için burada tekrar vermeye gerek yok
    )

    # 2. Veri Yükleme
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
    except Exception as e:
        st.error(f"HATA: PDF yüklenirken bir sorun oluştu. Dosya yolu: {e}")
        return None

    # 3. Metin Parçalama (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Vektör Veritabanı Oluşturma ve Retriever Tanımlama
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_FUNCTION,
        persist_directory=VECTOR_DB_DIR
    )
    retriever = vector_store.as_retriever()

    # 5. Yeni LCEL RAG Zincirini Kurma (RetrievalQA yerine)
    
    # Prompt Şablonu
    system_prompt = (
        "Sen bir uçak sistemleri uzmanısın. Yalnızca verilen bağlamı kullanarak soruları Türkçe yanıtla. "
        "Cevabı bağlamda bulamazsan 'Verilen bağlamda bu bilgi bulunmamaktadır.' de."
        "\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Bağlamı Birleştirme Zinciri
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Nihai Retrieval Zinciri
    qa_chain = create_retrieval_chain(
        retriever, 
        combine_docs_chain
    )
    return qa_chain


# --- STREAMLIT ARAYÜZÜ ---

def main():
    # 1. Sayfa Ayarları
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("Uçak Kontrol Sistemleri RAG Chatbot 🤖")
    st.caption("Veri Kaynağı: Uçak Kontrol Sistemleri PDF'i")

    # 2. RAG Sistemini Kurma
    qa_chain = setup_rag_system()
    
    if qa_chain is None:
        return

    # 3. Sohbet Geçmişini Başlatma
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Merhaba! Uçuş kontrol sistemleri hakkında ne sormak istersiniz?"}
        ]

    # 4. Sohbet Geçmişini Görüntüleme
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # 5. Kullanıcı Girişini İşleme
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        # Kullanıcı mesajını ekle
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Cevabı üretme
        with st.spinner("Cevap aranıyor..."):
            try:
                # Yeni LCEL zincirini çağırma yöntemi
                response = qa_chain.invoke({"input": prompt})
                yanit = response['answer']
                
                # Asistan cevabını ekle
                st.session_state["messages"].append({"role": "assistant", "content": yanit})
                st.chat_message("assistant").write(yanit)
            
            except Exception as e:
                # Hata durumunda mesaj
                hata_mesaji = f"Üzgünüm, RAG zincirinde bir hata oluştu: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": hata_mesaji})
                st.chat_message("assistant").write(hata_mesaji)

# --- UYGULAMAYI BAŞLATMA ---
if __name__ == "__main__":
    main()