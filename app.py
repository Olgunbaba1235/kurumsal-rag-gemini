"""
app.py — Streamlit sohbet arayüzü
=================================
Çalıştırmak için (proje klasöründe):  streamlit run app.py

🔑 API anahtarı bu dosyaya değil, ".env" dosyasına yazılır (bkz. rag_motoru.py).
"""

import streamlit as st  # Web arayüzü kütüphanesi

from rag_motoru import MODEL_ADI, LLMHatasi, RAGMotoru, YapilandirmaHatasi
from veri_isleyici import PDF_YOLU, BelgeParcasi, VektorVeritabani

# Sayfa başlığı ve sekme simgesi (ilk Streamlit komutu olmalı)
st.set_page_config(page_title="İade Asistanı", page_icon="💬")


# st.cache_resource: motor yalnızca bir kez kurulur, her mesajda yeniden oluşturulmaz.
# pdf_degisim_zamani önbellek anahtarına dahildir; PDF güncellenince motor yeniden kurulur.
@st.cache_resource(max_entries=1, show_spinner="📄 iade_politikasi.pdf okunuyor ve indeksleniyor...")
def motoru_hazirla(pdf_yolu: str, pdf_degisim_zamani: float) -> RAGMotoru:
    vektor_db = VektorVeritabani()  # Yerel ChromaDB veritabanını aç
    vektor_db.pdf_indeksle(pdf_yolu)  # PDF'i parçalara bölüp kaydet (değişmediyse atlanır)
    return RAGMotoru(vektor_db)  # API anahtarını .env'den okuyup motoru kur


def kaynaklari_goster(kaynaklar: list[BelgeParcasi]) -> None:
    """Cevabın dayandığı belge parçalarını katlanabilir bir kutuda gösterir."""
    if not kaynaklar:
        return
    with st.expander(f"📚 Kaynaklar ({len(kaynaklar)} parça)"):
        for parca in kaynaklar:
            st.markdown(f"**Sayfa {parca.sayfa}** · benzerlik: {parca.benzerlik:.2f}")
            ozet = parca.metin if len(parca.metin) <= 400 else parca.metin[:400] + "…"
            st.caption(ozet.replace("$", "\\$"))  # "$" işareti formül olarak yorumlanmasın


def sohbeti_temizle() -> None:
    st.session_state.mesajlar = []  # Sohbet geçmişini sıfırla


# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("💬 İade Politikası Asistanı")
st.caption("İade, değişim ve geri ödeme kurallarımızla ilgili sorularınızı yanıtlarım.")

# ── 1) Açılışta PDF'i ara ve indeksle ───────────────────────────────────────
if not PDF_YOLU.is_file():  # PDF yoksa kullanıcıyı uyar ve dur
    st.warning(
        f"**{PDF_YOLU.name}** bulunamadı. Dosyayı proje klasörüne koyup sayfayı yenileyin:\n\n"
        f"`{PDF_YOLU.parent}`"
    )
    st.stop()

try:
    motor = motoru_hazirla(str(PDF_YOLU), PDF_YOLU.stat().st_mtime)
except YapilandirmaHatasi as hata:  # .env'de API anahtarı yok
    st.error(f"⚙️ **Kurulum eksik:** {hata}")
    st.stop()
except Exception as hata:  # PDF okunamadı, embedding modeli indirilemedi vb.
    st.error(f"Belge hazırlanırken bir hata oluştu: {hata}")
    st.stop()

# ── Kenar çubuğu: belge bilgisi ve sohbeti temizleme düğmesi ─────────────────
with st.sidebar:
    st.subheader("⚙️ Asistan bilgisi")
    st.markdown(
        f"**Belge:** {PDF_YOLU.name}  \n"
        f"**İndekslenen parça:** {motor.vektor_db.parca_sayisi}  \n"
        f"**Model:** `{MODEL_ADI}`"
    )
    st.button("🗑️ Sohbeti temizle", on_click=sohbeti_temizle)

# ── 2) Sohbet ────────────────────────────────────────────────────────────────
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []  # Sayfa yenilense de mesajlar oturum boyunca saklanır

# Karşılama mesajı
with st.chat_message("assistant"):
    st.markdown(
        "Merhaba! 👋 İade politikamızla ilgili sorularınızı yanıtlayabilirim. "
        "Örneğin: *Ürünü kaç gün içinde iade edebilirim?*"
    )

# Önceki mesajları ekrana yeniden çiz
for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])
        kaynaklari_goster(mesaj.get("kaynaklar", []))

# Kullanıcı yeni bir soru yazdığında
if soru := st.chat_input("Sorunuzu yazın..."):
    with st.chat_message("user"):
        st.markdown(soru)  # Soruyu ekrana yaz

    with st.chat_message("assistant"):
        try:
            with st.spinner("Yanıt hazırlanıyor..."):
                sonuc = motor.soru_cevapla(soru)  # RAG motorundan cevap al
        except LLMHatasi as hata:
            st.error(str(hata))  # Hata varsa kullanıcıya göster
        else:
            st.markdown(sonuc.cevap)  # Cevabı ekrana yaz
            kaynaklari_goster(sonuc.kaynaklar)  # Dayandığı parçaları göster
            # Soru ve cevabı geçmişe ekle
            st.session_state.mesajlar.extend(
                [
                    {"role": "user", "content": soru},
                    {"role": "assistant", "content": sonuc.cevap, "kaynaklar": sonuc.kaynaklar},
                ]
            )
