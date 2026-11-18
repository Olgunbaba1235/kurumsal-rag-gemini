# 💬 Kurumsal RAG Asistanı

Şirket PDF belgelerini (örnek: `iade_politikasi.pdf`) okuyup sorulara yalnızca belgedeki bilgilere dayanarak cevap veren Streamlit sohbet botu.

**Akış:** PDF → parçalara bölme → ChromaDB (yerel vektör veritabanı) → en alakalı 3 parça → Gemini → Türkçe, kaynak sayfalı cevap

| Dosya | Görev |
|---|---|
| `veri_isleyici.py` | PDF'ten metin çıkarma (pypdf), parçalama ve ChromaDB'de vektör araması (`VektorVeritabani`) |
| `rag_motoru.py` | `.env` dosyasındaki API anahtarıyla, en alakalı parçaları kullanarak Gemini'den cevap üreten `RAGMotoru` |
| `app.py` | Streamlit sohbet arayüzü |

## Kurulum

Python 3.10 veya üzeri gerekir.

```bash
pip install -r requirements.txt
```

1. `.env.example` dosyasını `.env` adıyla kopyalayıp Gemini API anahtarınızı yazın ([Google AI Studio](https://aistudio.google.com/apikey)).
2. Belgenizi `iade_politikasi.pdf` adıyla proje klasörüne koyun.
3. Uygulamayı başlatın:

```bash
streamlit run app.py
```

> `.env` dosyası ve PDF'ler `.gitignore` sayesinde GitHub'a gönderilmez. API anahtarınızı asla koda yazmayın.
