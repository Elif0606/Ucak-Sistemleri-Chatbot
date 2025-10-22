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
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    # 3. Gemini Embedding Modelini Hazırlama
    embedding_model_name = "text-embedding-004"
    client = genai.Client(api_key=API_KEY)
    
    # 4. Gömmeleri Oluşturma ve FAISS İndeksi Kurma
    st.info("İlk kullanımda güçlü Gemini embedding modeli ile indeks oluşturuluyor. Lütfen bekleyin...")
    
    # BATCH BOYUTU: Gemini'nin limiti 100 olduğu için 90 kullanıyoruz.
    BATCH_SIZE = 90
    all_embeddings = []

    try:
        # Metinleri 90'ar kişilik gruplara ayırma (Batching)
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i + BATCH_SIZE]
            
            # Gemini API çağrısı
            response = client.models.embed_content(
                model=embedding_model_name,
                contents=batch # Doğru parametre adı
            )
            
            # YENİ VE GÜVENLİ KOD BAŞLANGICI (Boyut Kontrolü)
            
            # Gelen gömmeleri NumPy'a çeviriyoruz
            current_embeddings = np.array(response.embeddings)
            
            # GÜVENLİK KONTROLÜ: Boyut doğru değilse (768), bu batch'i atla.
            # Gemini'nin embedding boyutu 768'dir.
            if current_embeddings.shape[1] == 768:
                all_embeddings.append(current_embeddings)
            else:
                st.warning(f"UYARI: {i+1}. batch'te boyut tutarsızlığı bulundu. Atlanıyor. (Boyut: {current_embeddings.shape[1]})")
                
            st.caption(f"İlerleme: {i + len(batch)}/{len(sentences)} cümle gömmesi tamamlandı.")
            
            # YENİ VE GÜVENLİ KOD SONU
        
    except Exception as e:
        # Eğer API çağrısında hata olursa (Ağ/API Anahtarı)
        st.error(f"Gemini Embedding Hatası: {e}")
        return None, None, None # Hata durumunda fonksiyonu sonlandır
    
    # ******* ÖNEMLİ: GİRİNTİ BURADA BİTİYOR *******
    
    # 5. Güvenlik Kontrolü ve NumPy'a Dönüştürme (Doğru Girinti Seviyesi)
    if not all_embeddings:
        st.error("HATA: Gemini API'den hiçbir gömme (embedding) alınamadı. Lütfen API anahtarınızı veya PDF içeriğini kontrol edin.")
        return None, None, None

    try:
        # NumPy'a güvenli dönüştürme için np.vstack kullanma ve FAISS için float32'ye zorlama
        embeddings = np.vstack(all_embeddings).astype(np.float32) 
    except ValueError as e:
        st.error(f"Embeddings dizisi oluşturulurken hata: {e}. Muhtemelen boş bir gömme geldi.")
        return None, None, None

    # FAISS indeksi oluşturma (Bu kısım da fonksiyonun ana girinti seviyesinde olmalı)
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return embedding_model_name, index, sentences

# --- CEVAP OLUŞTURMA FONKSİYONU ---

def generate_rag_response(prompt, embedding_model_name, index, sentences):
    client = genai.Client(api_key=API_KEY)
    
    # 1. Prompt'un Gömmesini Oluşturma (Gemini ile)
    try:
        query_response = client.models.embed_content(
            model=embedding_model_name,
            contents=[prompt]
        )
        query_embedding = np.array(query_response.embeddings)
    except Exception as e:
        return f"Sorgu Gömmesi Hatası: {e}"
    
    # 2. En Yakın Parçaları FAISS ile Arama
    k = 4 # Bağlamı artırmak için 4 parçayı alalım
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