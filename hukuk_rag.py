from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder
from datetime import datetime
import os

DB_KLASORU = "./chroma_db"
LLM_MODEL  = "llama3.2:3b"   # Daha büyük model → daha iyi hukuki cevap ---  ollama pull llama3.1:8b

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb   = Chroma(persist_directory=DB_KLASORU, embedding_function=embeddings)

llm = ChatOllama(model=LLM_MODEL, temperature=0)

reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)

# ──────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────

def retrieve_and_rerank(sorgu, top_k=5):
    docs = retriever.invoke(sorgu)
    pairs = [(sorgu, doc.page_content) for doc in docs]
    skorlar = reranker.predict(pairs)
    sirali = sorted(zip(skorlar, docs), key=lambda x: x[0], reverse=True)
    return [(skor, doc) for skor, doc in sirali[:top_k]]

def baglam_olustur(sonuclar):
    parcalar = []
    for i, (skor, doc) in enumerate(sonuclar, 1):
        kaynak = doc.metadata.get("source", "bilinmiyor")
        parcalar.append(f"[KARAR {i} | Kaynak: {kaynak} | Skor: {skor:.2f}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parcalar)

def kaynaklari_yazdir(sonuclar):
    print("\n📁 Kaynak Kararlar:")
    gorulen = set()
    for skor, doc in sonuclar:
        kaynak = doc.metadata.get("source", "bilinmiyor")
        if kaynak not in gorulen:
            durum = "✅ Alakalı" if skor > 0 else "⚠️  Düşük alaka"
            print(f"   {durum} | {kaynak}  (skor: {skor:.3f})")
            gorulen.add(kaynak)
    print()

def dusuk_skor_kontrolu(sonuclar):
    if not sonuclar or sonuclar[0][0] < 0:
        print(f"\n⚠️  En yüksek skor: {sonuclar[0][0]:.3f}")
        print("   Bu konuya ait emsal karar veritabanında bulunamadı.")
        print("   İpucu: Get_data.py ile ilgili anahtar kelimeyle daha fazla karar indirin.\n")
        return True
    return False

# ──────────────────────────────────────────────
# MOD 1: EMSAL KARAR BUL
# ──────────────────────────────────────────────

def emsal_karar_bul(konu):
    print("\n🔍 Emsal kararlar aranıyor...")
    sonuclar = retrieve_and_rerank(konu, top_k=5)

    if dusuk_skor_kontrolu(sonuclar):
        return

    baglam = baglam_olustur(sonuclar)

    prompt = f"""Sen Türk hukuku alanında uzman bir hukuk asistanısın.
Aşağıdaki mahkeme kararlarını inceleyerek kullanıcının sorduğu konuyla ilgili emsal kararları tespit et ve açıkla.

Görevin:
- Her ilgili kararı ayrı ayrı özetle
- Kararın konusunu, tarafları, mahkemenin gerekçesini ve sonucunu belirt
- Kararın neden emsal niteliği taşıdığını açıkla
- Varsa kararlar arasındaki ortak hukuki ilkeleri vurgula

Kararlar:
{baglam}

Aranan Konu: {konu}

Yanıtın:"""

    print("⚖️  LLM analiz ediyor...\n")
    cevap = llm.invoke([HumanMessage(content=prompt)])
    print(f"Cevap:\n{cevap.content}")
    kaynaklari_yazdir(sonuclar)

# ──────────────────────────────────────────────
# MOD 2: DURUMUMA BENZER KARAR BUL
# ──────────────────────────────────────────────

def benzer_karar_bul(durum_aciklamasi):
    print("\n🔍 Benzer kararlar aranıyor...")
    sonuclar = retrieve_and_rerank(durum_aciklamasi, top_k=5)

    if dusuk_skor_kontrolu(sonuclar):
        return

    baglam = baglam_olustur(sonuclar)

    prompt = f"""Sen Türk hukuku alanında uzman bir hukuk asistanısın.
Müvekkil aşağıdaki durumu yaşıyor. Verilen mahkeme kararlarını inceleyerek bu duruma en çok benzeyen kararları tespit et.

Görevin:
- Müvekkilin durumu ile verilen kararları karşılaştır
- Her benzer kararı açıkla: ne kadar benzediğini, benzerlik/farklılık noktalarını belirt
- Kararların müvekkil için olumlu/olumsuz içerimlerini değerlendir
- Tahmini hukuki sonucu belirt (kesin olmayan, yol gösterici analiz)

Kararlar:
{baglam}

Müvekkilin Durumu:
{durum_aciklamasi}

Analiz:"""

    print("⚖️  LLM analiz ediyor...\n")
    cevap = llm.invoke([HumanMessage(content=prompt)])
    print(f"Cevap:\n{cevap.content}")
    kaynaklari_yazdir(sonuclar)

# ──────────────────────────────────────────────
# MOD 3: İSTİNAF KARARI YAZ
# ──────────────────────────────────────────────

def istinaf_karari_yaz(dava_bilgileri):
    print("\n🔍 İlgili kararlar ve gerekçeler aranıyor...")
    sonuclar = retrieve_and_rerank(dava_bilgileri, top_k=5)

    if dusuk_skor_kontrolu(sonuclar):
        return

    baglam = baglam_olustur(sonuclar)

    prompt = f"""Sen Türk hukuku alanında uzman bir hukuk asistanısın.
Aşağıdaki dava bilgilerine ve emsal kararlara dayanarak profesyonel bir istinaf dilekçesi taslağı hazırla.

İstinaf dilekçesi şu bölümleri içermeli:
1. MAHKEME VE DOSYA BİLGİLERİ (kullanıcının verdiği bilgilerden doldur)
2. İSTİNAF NEDENLERİ (madde madde, hukuki gerekçeli)
3. EMSAL KARARLAR (verilen kararlardan uygun olanları atıf olarak kullan)
4. SONUÇ VE TALEP

Önemli: Dilekçe resmi hukuki dil kullanmalı, atıflar "... tarihli ... sayılı karar" formatında olmalı.

Emsal Kararlar:
{baglam}

Dava Bilgileri:
{dava_bilgileri}

İstinaf Dilekçesi Taslağı:"""

    print("⚖️  İstinaf dilekçesi hazırlanıyor...\n")
    cevap = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n{cevap.content}")
    kaynaklari_yazdir(sonuclar)

    # TXT olarak kaydet
    cikti_klasoru = "./ciktilar"
    os.makedirs(cikti_klasoru, exist_ok=True)
    zaman_damgasi = datetime.now().strftime("%Y%m%d_%H%M%S")
    dosya_adi = os.path.join(cikti_klasoru, f"istinaf_dilekce_{zaman_damgasi}.txt")

    with open(dosya_adi, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  İSTİNAF DİLEKÇESİ TASLAĞI\n")
        f.write(f"  Oluşturulma tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("DAVA BİLGİLERİ:\n")
        f.write("-" * 40 + "\n")
        f.write(dava_bilgileri + "\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("DİLEKÇE:\n")
        f.write("-" * 40 + "\n")
        f.write(cevap.content + "\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("YARARLANILAN EMSAL KARARLAR:\n")
        f.write("-" * 40 + "\n")
        gorulen = set()
        for skor, doc in sonuclar:
            kaynak = doc.metadata.get('source', 'bilinmiyor')
            if kaynak not in gorulen:
                f.write(f"  - {kaynak}  (skor: {skor:.3f})\n")
                gorulen.add(kaynak)

    print(f"💾 Dilekçe kaydedildi: {dosya_adi}\n")

# ──────────────────────────────────────────────
# ANA MENÜ
# ──────────────────────────────────────────────

def menu():
    print("\n" + "="*55)
    print("  ⚖️   HUKUK RAG SİSTEMİ — Emsal Karar Asistanı")
    print("="*55)
    print("  1  →  Emsal karar bul")
    print("  2  →  Durumuma benzer karar bul")
    print("  3  →  İstinaf kararı taslağı yaz")
    print("  q  →  Çıkış")
    print("="*55)

print("Sistem başlatılıyor...")
print(f"✅ Veritabanı bağlandı: {DB_KLASORU}")
print(f"✅ LLM modeli: {LLM_MODEL}")

while True:
    menu()
    secim = input("\nSeçiminiz (1/2/3/q): ").strip()

    if secim in ("q", "quit", "çıkış"):
        print("Görüşürüz!")
        break

    elif secim == "1":
        konu = input("\n🔎 Emsal aranacak konu/anahtar kelime:\n> ").strip()
        if konu:
            emsal_karar_bul(konu)

    elif secim == "2":
        print("\n📋 Müvekkilinizin durumunu detaylıca açıklayın:")
        print("   (Taraflar, olay, talep, mevcut karar varsa belirtin)")
        durum = input("> ").strip()
        if durum:
            benzer_karar_bul(durum)

    elif secim == "3":
        print("\n📋 Dava bilgilerini girin:")
        print("   (Mahkeme, dosya no, taraflar, ilk derece kararı, itiraz gerekçeniz)")
        dava = input("> ").strip()
        if dava:
            istinaf_karari_yaz(dava)

    else:
        print("❌ Geçersiz seçim. 1, 2, 3 veya q girin.")
