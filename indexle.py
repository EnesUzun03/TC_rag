import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # metni küçük parçalara bölmek için
from langchain_chroma import Chroma # vektör veritabanı (embedding’leri(vektör gibi düşün) saklar) , benzerlik araması yapar
from langchain_ollama import OllamaEmbeddings #metni sayısal vektöre çevirir (embedding üretir)

TXT_KLASORU = "./kararlar"
DB_KLASORU  = "./chroma_db"

def load_txt_files(directory):
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                try:
                    # Önce utf-8 ile okumayı dene
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    try:
                        # Hata verirse Türkçe windows-1254 (cp1254) kodlaması ile dene
                        with open(path, 'r', encoding='windows-1254') as f:
                            text = f.read()
                    except Exception as e:
                        print(f"Atlanıyor: {path} okuma hatası: {e}")
                        continue
                docs.append(Document(page_content=text, metadata={"source": path}))
    return docs

print("TXT'ler yükleniyor...")
belgeler = load_txt_files(TXT_KLASORU)
print(f"   {len(belgeler)} sayfa bulundu.")

# Çok kısa ve anlamsız sayfaları filtrele
print("Anlamsız sayfalar temizleniyor...")
temiz_belgeler = []
for belge in belgeler:
    metin = belge.page_content.strip()
    # 100 karakterden kısa sayfaları atla (kapak, boş sayfa vb.)
    if len(metin) < 100:
        continue
    # Sadece rakam/özel karakter olan sayfaları atla / anlamsız içeriği embedding olarak kaydetmek istemeyiz
    harf_sayisi = sum(1 for c in metin if c.isalpha())
    if harf_sayisi < 50:
        continue
    temiz_belgeler.append(belge)

print(f"   {len(temiz_belgeler)} anlamlı sayfa kaldı ({len(belgeler) - len(temiz_belgeler)} sayfa atlandı).")

print("Chunk'lara bölünüyor...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # Her parça maksimum 800 karakter , Daha büyük chunk → daha fazla bağlam
    chunk_overlap=150,     # Chunklar arasında 150 karakter overlap ,Daha fazla overlap → cümle kopması azalır . Overlap, metni parçalara (chunk) bölerken bu parçaların birbirleriyle kısmen örtüşecek şekilde oluşturulmasıdır.
    separators=["\n\n", "\n", ". ", " "], # bölme noktaları (öncelik sırasına göre) , Öncelikle paragraflara böl, sonra satırlara, sonra cümlelere, en son boşluklara böl
    length_function=len,
)
# input sayfalar -> output chunk'lar (parçalar) , Her chunk'un metadata'sında hangi sayfadan geldiği bilgisi var
chunks = splitter.split_documents(temiz_belgeler) 

# Çok kısa chunk'ları filtrele , 80 karakterden kısa chunk çöpe gider
chunks = [c for c in chunks if len(c.page_content.strip()) > 80]
print(f"   {len(chunks)} chunk oluşturuldu.")

print("Vektörler hesaplanıyor (birkaç dakika sürebilir)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text") #metni sayısal vektöre çevirir (embedding üretir) , Aynı anlama gelen metinler benzer vektörlere sahip olur , Benzer vektörler → benzer içerik

"""Vektör veritabani oluşturuluyor...
Her chunk → embeddinge çevrilir
embedding + metin birlikte saklanir
Chroma DB oluşturulur
diske yazilir (DB_KLASORU)"""
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_KLASORU
)

print(f"Tamamlandi! {DB_KLASORU} klasörüne kaydedildi.")