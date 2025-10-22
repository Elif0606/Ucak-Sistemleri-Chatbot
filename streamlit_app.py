import os
import streamlit as st
import google.genai as genai
from google.genai.errors import APIError

# RAG için gerekli kütüphaneler
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- SABİT AYARLAR ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Hızlı ve yerel bir embedding modeli
VECTOR_DB_FILE = "faiss_index.bin"
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "veri_seti", "ucak_kontrol_sistemleri.pdf")

# API Anahtarını Streamlit Secrets'tan güvenli bir şekilde al
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    # 20. satır: Genellikle bu eski bir LangChain veya SDK metoduydu.
    # En güncel SDK'da bu satıra gerek yoktur, Client objesi otomatik yapılandırılır.
    # Bu satırı SİLİN veya YORUM SATIRI yapın.
    # genai.configure(api_key=API_KEY) 
except KeyError:
    st.error("HATA: GEMINI_API_KEY Streamlit Secrets'ta tanımlı değil! Lütfen secrets.toml dosyasını kontrol edin.")
    API_KEY = None

# --- RAG SİSTEMİ KURULUMU (Cache Edilmiş) ---

@st.cache_resource
def setup_rag_system():
    if not API_KEY:
        return None, None

    # 1. Metni PDF'ten Çıkarma
    try:
        reader = PdfReader(PDF_PATH)
        text = "".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        st.error(f"HATA: PDF yüklenemedi veya okunamadı: {e}")
        return None, None

    # 2. Metni Parçalama (Basit Metin Parçalayıcı)
    # Cümle bazlı parçalama, daha sonra embedding yapısını korumak için
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    # 3. Embedding Modelini Yükleme (Lokal Model)
    # Bu model, FAISS ile kullanılacak gömmeleri hızlıca oluşturur
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 4. Gömmeleri Oluşturma ve FAISS İndeksi Kurma
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    
    # FAISS indeksi: Vektörleri aramak için
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return embedding_model, index, sentences


# --- CEVAP OLUŞTURMA FONKSİYONU ---

def generate_rag_response(prompt, embedding_model, index, sentences):
    
    # 1. Prompt'un Gömmesini Oluşturma
    query_embedding = embedding_model.encode([prompt], convert_to_tensor=True).cpu().numpy()
    
    # 2. En Yakın Parçaları FAISS ile Arama
    k = 3 # En iyi 3 parçayı al
    distances, indices = index.search(query_embedding, k)
    
    # 3. Bağlamı (Context) Birleştirme
    context_chunks = [sentences[i] for i in indices[0]]
    context = "\n".join(context_chunks)

    # 4. Gemini için Prompt Şablonu
    system_prompt = (
        "Sen bir uçak sistemleri uzmanısın. Yalnızca aşağıdaki BAĞLAMI kullanarak soruları Türkçe yanıtla. "
        "Cevabı bağlamda bulamazsan 'Verilen bağlamda bu bilgi bulunmamaktadır.' de."
        "\n\n--- BAĞLAM ---\n"
        f"{context}"
    )
    
    # 5. Gemini API Çağrısı
    try:
        # Client objesini API_KEY ile başlatmak yeterlidir.
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": "Soru: " + prompt}]}
            ],
            config={"temperature": 0.1}
        )
        return response.text

    except APIError as e:
        return f"Gemini API Hatası: {e}"
    except Exception as e:
        return f"Beklenmedik Hata: {e}"


# --- STREAMLIT ARAYÜZÜ ---

def main():
    st.set_page_config(page_title="Acil RAG Chatbot", layout="wide")
    st.title("Acil Uçak Kontrol Sistemleri RAG Chatbot 🚀")
    st.caption("Veri Kaynağı: Uçak Kontrol Sistemleri PDF'i (LangChain'siz Acil Çözüm)")

    # 1. RAG Sistemini Kurma
    rag_data = setup_rag_system()
    if rag_data is None:
        return
    embedding_model, index, sentences = rag_data

    # 2. Sohbet Geçmişini Başlatma
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Merhaba! Proje teslimi için acil durum modu devrede. Uçuş kontrol sistemleri hakkında ne sormak istersiniz?"}
        ]

    # 3. Sohbet Geçmişini Görüntüleme
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # 4. Kullanıcı Girişini İşleme
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Cevap aranıyor..."):
            yanit = generate_rag_response(prompt, embedding_model, index, sentences)
            
            st.session_state["messages"].append({"role": "assistant", "content": yanit})
            st.chat_message("assistant").write(yanit)

if __name__ == "__main__":
    main()