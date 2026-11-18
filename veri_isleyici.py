"""
veri_isleyici.py — PDF okuma, parçalama ve vektör veritabanı
============================================================
• pdf_metnini_cikar() : pypdf ile PDF'in metnini sayfa sayfa çıkarır.
• metni_parcala()     : Metni, cümleleri bölmeden, birbiriyle örtüşen parçalara ayırır.
• VektorVeritabani    : Parçaları yerel ChromaDB koleksiyonuna kaydeder ve benzerlik araması yapar.

Bu dosya API anahtarı GEREKTİRMEZ; tüm işlemler bilgisayarınızda yapılır.
Hızlı test:  python veri_isleyici.py "kaç gün içinde iade edebilirim"
"""

import hashlib  # PDF'in değişip değişmediğini anlamak için dosya özeti (hash) hesaplar
import re  # Metni temizlemek ve bölmek için düzenli ifadeler
import sys  # Terminalden verilen test sorusunu okumak için
import unicodedata  # PDF'teki "ﬁ" gibi bitişik harfleri normal harflere çevirmek için
from dataclasses import dataclass
from pathlib import Path

import chromadb  # Yerel vektör veritabanı
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pypdf import PdfReader  # PDF okuyucu
from pypdf.errors import PdfReadError

# ── Ayarlar ──────────────────────────────────────────────────────────────────
PROJE_KLASORU = Path(__file__).resolve().parent  # Bu dosyanın bulunduğu klasör
PDF_YOLU = PROJE_KLASORU / "iade_politikasi.pdf"  # Okunacak belge
CHROMA_KLASORU = PROJE_KLASORU / "chroma_db"  # Veritabanının diske kaydedileceği klasör
KOLEKSIYON_ADI = "iade_politikasi"  # ChromaDB'deki koleksiyonun adı
PARCA_BOYUTU = 500  # Bir parçanın en fazla karakter sayısı
PARCA_ORTUSMESI = 100  # Ardışık parçaların ortak karakter sayısı (bağlam kopmasın diye)


@dataclass
class BelgeParcasi:
    """Veritabanında yapılan aramada bulunan tek bir metin parçası."""

    metin: str  # Parçanın metni
    sayfa: int  # PDF'teki sayfa numarası
    benzerlik: float  # 0–1 arası; 1'e yaklaştıkça soruyla daha alakalı


# ── 1) PDF OKUMA ─────────────────────────────────────────────────────────────
def pdf_metnini_cikar(pdf_yolu: str | Path = PDF_YOLU) -> list[tuple[int, str]]:
    """PDF'in metnini sayfa sayfa çıkarır. Dönüş: [(sayfa_no, metin), ...]"""
    pdf_yolu = Path(pdf_yolu)
    _pdf_var_mi(pdf_yolu)  # Dosya yoksa anlaşılır bir hata ver

    try:
        okuyucu = PdfReader(pdf_yolu)  # PDF'i aç
        if okuyucu.is_encrypted:
            okuyucu.decrypt("")  # Yalnızca parolasız (izin kısıtlamalı) PDF'ler açılabilir
        sayfalar = [
            (sayfa_no, _metni_temizle(sayfa.extract_text() or ""))  # Her sayfanın metnini al
            for sayfa_no, sayfa in enumerate(okuyucu.pages, start=1)
        ]
    except PdfReadError as hata:  # Bozuk veya okunamayan PDF
        raise ValueError(f"'{pdf_yolu.name}' okunamadı: {hata}") from hata

    sayfalar = [(no, metin) for no, metin in sayfalar if metin]  # Boş sayfaları at
    if not sayfalar:  # Hiç metin yoksa PDF büyük ihtimalle taranmış bir görüntüdür
        raise ValueError(
            f"'{pdf_yolu.name}' içinden metin çıkarılamadı. Belge taranmış (görüntü) bir "
            "PDF olabilir; bu durumda önce OCR ile metne dönüştürülmesi gerekir."
        )
    return sayfalar


def _pdf_var_mi(pdf_yolu: Path) -> None:
    if not pdf_yolu.is_file():
        raise FileNotFoundError(
            f"'{pdf_yolu.name}' bulunamadı. Dosyayı şu klasöre koyun: {pdf_yolu.parent}"
        )


def _metni_temizle(metin: str) -> str:
    """PDF'ten gelen ham metni düzeltir."""
    metin = unicodedata.normalize("NFKC", metin)  # Bitişik harfleri ayır ("ﬁ" → "fi")
    metin = re.sub(r"[ \t]+", " ", metin)  # Art arda boşlukları teke indir
    metin = re.sub(r" *\n *", "\n", metin)  # Satır başı/sonundaki boşlukları sil
    metin = re.sub(r"\n(?=[a-zçğıöşü])", " ", metin)  # Küçük harfle süren satır = bölünmüş cümle
    metin = re.sub(r"\n{3,}", "\n\n", metin)  # Fazla boş satırları azalt
    return metin.strip()


# ── 2) PARÇALAMA (CHUNKING) ──────────────────────────────────────────────────
_KESME_NOKTALARI = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")  # Tercih sırasına göre
_BOSLUK = re.compile(r"\s")


def metni_parcala(
    metin: str, parca_boyutu: int = PARCA_BOYUTU, ortusme: int = PARCA_ORTUSMESI
) -> list[str]:
    """Metni en fazla `parca_boyutu` karakterlik, birbiriyle örtüşen parçalara böler.

    Kesme noktası olarak sırasıyla paragraf sonu, satır sonu, cümle sonu ve kelime
    arası tercih edilir; böylece cümleler mümkün olduğunca ortadan bölünmez.
    """
    if not 0 <= ortusme < parca_boyutu:
        raise ValueError("ortusme, 0 ile parca_boyutu arasında olmalı.")

    metin = metin.strip()
    parcalar: list[str] = []
    bas = 0  # Sıradaki parçanın başlangıç konumu
    while bas < len(metin):
        son = min(bas + parca_boyutu, len(metin))  # Parçanın en fazla uzanabileceği yer
        if son < len(metin):  # Son parça değilse en uygun kesme noktasını ara
            pencere = metin[bas:son]
            for ayirici in _KESME_NOKTALARI:
                konum = pencere.rfind(ayirici)  # Penceredeki son ayırıcıyı bul
                if konum >= parca_boyutu // 2:  # Parça çok kısa kalmasın
                    son = bas + konum + len(ayirici)
                    break
            else:  # Uygun yer yoksa kelimeyi bölmemek için penceredeki son boşluktan kes
                konum = max((b.start() for b in _BOSLUK.finditer(pencere)), default=0)
                if konum > 0:
                    son = bas + konum + 1

        parca = metin[bas:son].strip()
        if parca:
            parcalar.append(parca)
        if son >= len(metin):  # Metnin sonuna gelindi
            break

        # Örtüşme: sonraki parça, bu parçanın son ~`ortusme` karakterinde bir kelime başından başlar
        bosluk = _BOSLUK.search(metin, max(son - ortusme, bas + 1), son)
        bas = bosluk.end() if bosluk else son
        while metin[bas].isspace():  # Parça boşlukla başlamasın
            bas += 1
    return parcalar


# ── 3) VEKTÖR VERİTABANI (ChromaDB) ──────────────────────────────────────────
class VektorVeritabani:
    """Belge parçalarını yerel ChromaDB koleksiyonunda saklar ve anlamsal arama yapar."""

    def __init__(self, klasor: str | Path = CHROMA_KLASORU):
        # Verileri diske yazan kalıcı istemci: uygulama kapansa da veriler kaybolmaz
        istemci = chromadb.PersistentClient(
            path=str(klasor),
            settings=Settings(anonymized_telemetry=False),  # Kullanım istatistiği gönderme
        )
        # Koleksiyonu aç, yoksa oluştur
        self._koleksiyon = istemci.get_or_create_collection(
            name=KOLEKSIYON_ADI,
            # Metni vektöre çeviren yerel model (ilk çalıştırmada ~80 MB indirilir).
            # İngilizce ağırlıklıdır. Türkçe aramayı iyileştirmek için "pip install
            # sentence-transformers" kurup bu satırı şununla değiştirin ve chroma_db klasörünü silin:
            # embedding_functions.SentenceTransformerEmbeddingFunction(
            #     model_name="paraphrase-multilingual-MiniLM-L12-v2")
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
            configuration={"hnsw": {"space": "cosine"}},  # Benzerlik ölçüsü: kosinüs
        )

    @property
    def parca_sayisi(self) -> int:
        """Veritabanındaki toplam parça sayısı."""
        return self._koleksiyon.count()

    def pdf_indeksle(self, pdf_yolu: str | Path = PDF_YOLU) -> int:
        """PDF'i okur, parçalar ve veritabanına kaydeder. Dönüş: belgenin parça sayısı.

        PDF değişmediyse tekrar indekslemez; güncellendiyse eski parçaları silip yenilerini ekler.
        """
        pdf_yolu = Path(pdf_yolu)
        _pdf_var_mi(pdf_yolu)
        kaynak = pdf_yolu.name
        ozet = hashlib.sha256(pdf_yolu.read_bytes()).hexdigest()  # Dosya içeriğinin parmak izi

        # Bu sürüm zaten kayıtlıysa yeniden indekslemeye gerek yok
        ayni_surum = {"$and": [{"kaynak": kaynak}, {"ozet": ozet}]}
        if self._koleksiyon.get(where=ayni_surum, limit=1)["ids"]:
            return len(self._koleksiyon.get(where={"kaynak": kaynak})["ids"])

        # Önce yeni sürümü hazırla; okuma hatası olursa eski veriler silinmemiş olur
        kimlikler, metinler, ust_veriler = [], [], []
        for sayfa_no, sayfa_metni in pdf_metnini_cikar(pdf_yolu):
            for sira, parca in enumerate(metni_parcala(sayfa_metni)):
                kimlikler.append(f"{pdf_yolu.stem}-{ozet[:8]}-s{sayfa_no}-p{sira}")  # Benzersiz kimlik
                metinler.append(parca)
                ust_veriler.append({"kaynak": kaynak, "sayfa": sayfa_no, "ozet": ozet})

        self._koleksiyon.delete(where={"kaynak": kaynak})  # Belgenin eski sürümünü kaldır
        for i in range(0, len(kimlikler), 100):  # Büyük belgeler için 100'erli gruplar hâlinde ekle
            self._koleksiyon.upsert(
                ids=kimlikler[i : i + 100],
                documents=metinler[i : i + 100],  # Vektörler bu metinlerden otomatik üretilir
                metadatas=ust_veriler[i : i + 100],
            )
        return len(kimlikler)

    def ara(self, sorgu: str, k: int = 3) -> list[BelgeParcasi]:
        """Soruya anlamca en yakın `k` parçayı, en alakalısı başta olacak şekilde döndürür."""
        toplam = self._koleksiyon.count()
        if toplam == 0 or not sorgu.strip():  # Boş veritabanı veya boş soru
            return []
        sonuc = self._koleksiyon.query(query_texts=[sorgu], n_results=min(k, toplam))
        return [
            BelgeParcasi(
                metin=metin,
                sayfa=int(ust_veri["sayfa"]),
                benzerlik=round(1 - mesafe, 3),  # Kosinüs mesafesini benzerliğe çevir
            )
            for metin, ust_veri, mesafe in zip(
                sonuc["documents"][0], sonuc["metadatas"][0], sonuc["distances"][0], strict=True
            )
        ]


if __name__ == "__main__":
    vektor_db = VektorVeritabani()
    print(f"{vektor_db.pdf_indeksle()} parça indekslendi → {CHROMA_KLASORU}")
    soru = " ".join(sys.argv[1:]) or "Ürünü kaç gün içinde iade edebilirim?"
    print(f"\nSoru: {soru}")
    for bulunan in vektor_db.ara(soru):
        print(f"\n[Sayfa {bulunan.sayfa} · benzerlik {bulunan.benzerlik:.2f}]\n{bulunan.metin}")
