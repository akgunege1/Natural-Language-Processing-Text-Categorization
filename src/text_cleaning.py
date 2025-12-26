
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# Türkçe stopwords listesi
TURKISH_STOPWORDS = [
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu',
    'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'daha', 'en', 'gibi', 'hem', 'hep',
    'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kadar', 'kendi', 'ki', 'kim', 'mı', 'mu',
    'mü', 'nasıl', 'ne', 'neden', 'nerede', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki',
    'şey', 'şu', 'tüm', 've', 'veya', 'ya', 'yani', 'bir', 'iki', 'üç', 'dört', 'beş',
    'altı', 'yedi', 'sekiz', 'dokuz', 'on', 'burada', 'şura', 'orada', 'nerede', 'nereye',
    'nereden', 'kim', 'kime', 'kimi', 'kimden', 'kimin', 'hangi', 'hangisi', 'hangi', 'şu',
    'o', 'bu', 'şunlar', 'bunlar', 'onlar', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
    'benim', 'senin', 'onun', 'bizim', 'sizin', 'onların', 'beni', 'seni', 'onu', 'bizi',
    'sizi', 'onları', 'bana', 'sana', 'ona', 'bize', 'size', 'onlara', 'benden', 'senden',
    'ondan', 'bizden', 'sizden', 'onlardan', 'benimle', 'seninle', 'onunla', 'bizimle',
    'sizinle', 'onlarla', 'benimki', 'seninki', 'onunki', 'bizimki', 'sizinki', 'onlarınki'
]

def clean_text(text):
    """
    Metni temizler:
    - Küçük harfe çevirme
    - Sayıları temizleme
    - Özel karakterleri temizleme
    - Kısa kelimeleri (2 karakterden kısa) temizleme
    - Stopwords temizleme
    
    Args:
        text: Temizlenecek metin
    
    Returns:
        str: Temizlenmiş metin
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Küçük harfe çevir
    text = text.lower()
    
    # Sayıları temizle
    text = re.sub(r'\d+', '', text)
    
    # Özel karakterleri temizle (sadece Türkçe harfler ve boşluk kalacak)
    text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
    
    # Birden fazla boşluğu tek boşluğa çevir
    text = re.sub(r'\s+', ' ', text)
    
    # Kelimelere ayır
    words = text.split()
    
    # Kısa kelimeleri (2 karakterden kısa) ve stopwords'leri temizle
    cleaned_words = [
        word for word in words 
        if len(word) > 2 and word not in TURKISH_STOPWORDS
    ]
    
    # Tekrar birleştir
    cleaned_text = ' '.join(cleaned_words)
    
    return cleaned_text.strip()

def clean_dataset(input_path, output_path):
    """
    Dataset'i temizler ve kaydeder.
    
    Args:
        input_path: Ham veri dosya yolu
        output_path: Temizlenmiş veri kayıt yolu
    """
    # Ham veriyi yükle
    print(f"Ham veri yükleniyor: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8')
    
    print(f"Toplam {len(df)} örnek temizleniyor...")
    
    # Metinleri temizle
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Boş metinleri filtrele
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Sadece gerekli kolonları al
    df_cleaned = df[['cleaned_text', 'category']].copy()
    df_cleaned.columns = ['text', 'category']
    
    # Kaydet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Temizlenmiş veri kaydedildi: {output_path}")
    print(f"Temizlenmiş örnek sayısı: {len(df_cleaned)}")
    print(f"Kategori dağılımı:\n{df_cleaned['category'].value_counts()}")
    
    return df_cleaned

def load_cleaned_dataset(filepath):
    """
    Temizlenmiş dataset'i yükler.
    
    Args:
        filepath: Temizlenmiş veri dosya yolu
    
    Returns:
        DataFrame: Temizlenmiş veri
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Temizlenmiş veri bulunamadı: {filepath}\n"
            f"Lütfen önce text_cleaning.py'yi çalıştırarak veriyi temizleyin."
        )
    
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Temizlenmiş veri yüklendi: {filepath}")
    print(f"Örnek sayısı: {len(df)}")
    
    return df

if __name__ == "__main__":
    # Ham veriyi temizle ve kaydet
    clean_dataset(
        input_path="data/raw/dataset.csv",
        output_path="data/cleaned/cleaned_dataset.csv"
    )



