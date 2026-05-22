from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder

DB_KLASORU = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(
    persist_directory=DB_KLASORU,
    embedding_function=embeddings
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_gpu=1
)

# Türkçe/çok dilli reranker modeli
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Adım 1: Geniş retrieval — 20 chunk getir
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)

def rerank(soru, docs, top_k=5):
    # Soru + her chunk'ı çift olarak reranker'a ver
    pairs = [(soru, doc.page_content) for doc in docs]
    # Reranker her çiftin alakalılık skorunu hesaplar
    skorlar = reranker.predict(pairs)
    # Skora göre sırala, en iyi top_k tanesini al
    sirali = sorted(zip(skorlar, docs), key=lambda x: x[0], reverse=True)
    return [(skor, doc) for skor, doc in sirali[:top_k]]

def sor(soru):
    # Adım 1: 20 chunk getir
    docs = retriever.invoke(soru)
    print(f"   {len(docs)} chunk getirildi, reranking yapılıyor...")

    # Adım 2: Reranker ile en iyi 5'i seç
    en_iyi_sonuclar = rerank(soru, docs, top_k=5)
    en_iyi_docs = [doc for _, doc in en_iyi_sonuclar]
    print(f"   En iyi 5 chunk seçildi.")

    # Adım 3: En yüksek skor çok düşükse belgede bilgi yok demektir
    en_yuksek_skor = en_iyi_sonuclar[0][0] if en_iyi_sonuclar else -99
    if en_yuksek_skor < 0:
        print(f"\n⚠️  En yüksek alakalılık skoru: {en_yuksek_skor:.3f} — belgelerinizde bu konuya ait karar bulunamadı.")
        print("   İpucu: 'kararlar' klasörüne ilgili .txt dosyaları ekleyip indexle.py'yi tekrar çalıştırın.\n")
        return

    # Adım 3: Seçilen chunk'ları kaynak bilgisiyle birlikte bağlam olarak ver
    baglam_parcalari = []
    for i, (skor, doc) in enumerate(en_iyi_sonuclar, 1):
        kaynak = doc.metadata.get("source", "bilinmiyor")
        baglam_parcalari.append(f"[Karar {i} | Kaynak: {kaynak}]\n{doc.page_content}")
    baglam = "\n\n---\n\n".join(baglam_parcalari)

    mesaj = f"""Sen bir hukuk asistanısın. Sana verilen mahkeme kararı belgelerini analiz ederek kullanıcının sorusunu Türkçe yanıtlıyorsun.

Talimatlar:
- Verilen belge parçalarındaki kararları özetle ve soruyla ilişkilendir.
- Her ilgili kararı ayrı ayrı açıkla: ne hakkında olduğunu, hangi tarafların olduğunu, mahkemenin nasıl karar verdiğini belirt.
- Belgelerde soruyla ilgili bilgi varsa mutlaka kullan, "bilgi yok" deme.
- Birden fazla ilgili karar varsa hepsini listele.
- Cevabını net, anlaşılır ve madde madde yaz.

Belgeler:
{baglam}

Soru: {soru}

Cevap:"""

    cevap = llm.invoke([HumanMessage(content=mesaj)])
    print(f"\nCevap:\n{cevap.content}")

    print("\nKaynak Dokümanlar (reranked, alakalılık skoruyla):")
    gorulen = set()
    for skor, doc in en_iyi_sonuclar:
        kaynak = doc.metadata.get("source", "bilinmiyor")
        if kaynak not in gorulen:
            print(f"   • {kaynak}  (skor: {skor:.3f})")
            gorulen.add(kaynak)
    print()

print("RAG + Reranker sistemi hazır. Çıkmak için 'q' yaz.\n")

while True:
    soru = input("Soru: ").strip()
    if soru.lower() in ("q", "quit", "çıkış"):
        print("Görüşürüz!")
        break
    if not soru:
        continue
    sor(soru)