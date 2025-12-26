"""
Dataset Hazırlama Modülü
3 kategori (Spor, Finans, Teknoloji) için örnek Türkçe metinler oluşturur.
"""

import pandas as pd
import os

def create_sample_dataset(num_samples_per_category=500):
    """
    Her kategori için örnek Türkçe metinler oluşturur.
    
    Args:
        num_samples_per_category: Her kategoriden kaç örnek oluşturulacağı
    
    Returns:
        DataFrame: text ve category kolonlarına sahip veri seti
    """
    
    # Spor kategorisi için örnek metinler
    spor_texts = [
        "Futbol maçında takımımız harika bir performans sergiledi ve rakiplerini 3-1 yendi.",
        "Basketbol turnuvasında şampiyonluk için mücadele eden takımlar finalde karşılaştı.",
        "Tenis oyuncusu Wimbledon turnuvasında finale yükseldi ve büyük bir başarı elde etti.",
        "Olimpiyat oyunlarında milli atletimiz altın madalya kazandı ve ülkemizi gururlandırdı.",
        "Futbol liginde şampiyonluk yarışı kızıştı ve lider takım puan farkını korudu.",
        "Voleybol milli takımımız uluslararası turnuvada başarılı sonuçlar aldı.",
        "Atletizm yarışmasında rekor kırıldı ve sporcu tarihi bir başarıya imza attı.",
        "Yüzme müsabakasında genç sporcumuz dünya rekoruna yaklaştı.",
        "Boksörümüz dünya şampiyonluğu için ringe çıktı ve zafer kazandı.",
        "Futbol transfer sezonunda büyük kulüpler yıldız oyuncuları kadrolarına kattı.",
        "Basketbol maçında son saniyelerde atılan üçlük takımı galibiyete taşıdı.",
        "Tenis turnuvasında genç yetenekler dikkat çekti ve gelecek vadediyor.",
        "Spor salonunda antrenman yapan sporcular olimpiyat hazırlıklarını sürdürüyor.",
        "Futbol antrenörü takımına yeni stratejiler öğretiyor ve başarı için çalışıyor.",
        "Atletizm pistinde koşan sporcular rekor denemeleri yapıyor.",
        "Yüzme havuzunda eğitim alan gençler milli takım seçmelerine hazırlanıyor.",
        "Basketbol sahasında oynanan maç seyircileri heyecanlandırdı.",
        "Futbol taraftarları stadyumda takımlarını coşkuyla destekledi.",
        "Tenis kortunda oynanan maç uzun süre sürdü ve heyecan vericiydi.",
        "Spor muhabirleri maç sonrası oyuncularla röportaj yaptı.",
    ]
    
    # Finans kategorisi için örnek metinler
    finans_texts = [
        "Borsa İstanbul bugün yükselişle kapandı ve endeks rekor seviyeye ulaştı.",
        "Döviz kurlarındaki dalgalanmalar ekonomiyi etkiledi ve yatırımcıları endişelendirdi.",
        "Merkez Bankası faiz oranlarını değiştirdi ve piyasalar bu karara tepki gösterdi.",
        "Şirket hisse senetleri değer kazandı ve yatırımcılar kar elde etti.",
        "Ekonomik büyüme rakamları açıklandı ve uzmanlar olumlu değerlendirme yaptı.",
        "Enflasyon oranları düştü ve ekonomi için umut verici sinyaller geldi.",
        "Yatırım fonları performans gösterdi ve yatırımcılar memnun kaldı.",
        "Kripto para piyasasında volatilite arttı ve yatırımcılar dikkatli olmalı.",
        "Bankalar kredi faiz oranlarını güncelledi ve müşteriler bilgilendirildi.",
        "Borsada işlem hacmi arttı ve likidite yükseldi.",
        "Ekonomi bakanlığı yeni teşvik paketi açıkladı ve iş dünyası memnun oldu.",
        "Döviz rezervleri arttı ve ülke ekonomisi güçlendi.",
        "Yatırım bankaları analiz raporları yayınladı ve piyasa değerlendirmesi yaptı.",
        "Hazine bonoları faiz getirisi sağladı ve yatırımcılar ilgi gösterdi.",
        "Şirketlerin kâr raporları açıklandı ve hisse senetleri değer kazandı.",
        "Ekonomik kriz endişeleri azaldı ve piyasalar toparlanma gösterdi.",
        "Yatırım danışmanları portföy önerileri sundu ve müşterilere rehberlik etti.",
        "Borsa analistleri gelecek dönem için olumlu tahminler yaptı.",
        "Finansal piyasalarda işlem hacmi arttı ve likidite yükseldi.",
        "Ekonomi uzmanları büyüme tahminlerini revize etti ve iyimser görüş bildirdi.",
    ]
    
    # Teknoloji kategorisi için örnek metinler
    teknoloji_texts = [
        "Yapay zeka teknolojisi hızla gelişiyor ve hayatımızı değiştiriyor.",
        "Yeni nesil akıllı telefonlar piyasaya çıktı ve tüketiciler ilgi gösterdi.",
        "Bilgisayar işlemcileri daha güçlü hale geldi ve performans arttı.",
        "Bulut bilişim hizmetleri yaygınlaştı ve şirketler dijital dönüşüm yaptı.",
        "Siber güvenlik önlemleri artırıldı ve veri koruma sağlandı.",
        "Yazılım geliştirme araçları güncellendi ve programcılar verimlilik kazandı.",
        "Robot teknolojisi ilerledi ve otomasyon sistemleri yaygınlaştı.",
        "İnternet hızı arttı ve kullanıcılar daha iyi deneyim yaşadı.",
        "Yapay zeka asistanları evlerde kullanılmaya başlandı ve hayatı kolaylaştırdı.",
        "Blockchain teknolojisi finans sektöründe uygulanmaya başlandı.",
        "Sanal gerçeklik gözlükleri piyasaya çıktı ve eğlence sektörünü etkiledi.",
        "Büyük veri analizi şirketlere stratejik kararlar verme imkanı sağladı.",
        "Nesnelerin interneti cihazları birbirine bağladı ve akıllı evler oluştu.",
        "Yazılım güvenlik açıkları tespit edildi ve yamalar yayınlandı.",
        "Yapay zeka algoritmaları tıp alanında kullanılmaya başlandı.",
        "Kuantum bilgisayarlar geliştirildi ve hesaplama gücü arttı.",
        "5G teknolojisi yaygınlaştı ve mobil internet hızı yükseldi.",
        "Yazılım şirketleri yeni ürünler tanıttı ve teknoloji fuarlarında sergiledi.",
        "Siber saldırılar arttı ve güvenlik uzmanları uyarı yaptı.",
        "Yapay zeka destekli çeviri sistemleri dil bariyerlerini kaldırdı.",
    ]
    
    # Her kategoriden belirtilen sayıda örnek oluştur
    all_texts = []
    all_categories = []
    
    # Spor kategorisi
    for i in range(num_samples_per_category):
        text = spor_texts[i % len(spor_texts)]
        # Her metne biraz varyasyon ekle
        if i > 0:
            text = text.replace(".", f" {i}.", 1) if i % 3 == 0 else text
        all_texts.append(text)
        all_categories.append("Spor")
    
    # Finans kategorisi
    for i in range(num_samples_per_category):
        text = finans_texts[i % len(finans_texts)]
        if i > 0:
            text = text.replace(".", f" {i}.", 1) if i % 3 == 0 else text
        all_texts.append(text)
        all_categories.append("Finans")
    
    # Teknoloji kategorisi
    for i in range(num_samples_per_category):
        text = teknoloji_texts[i % len(teknoloji_texts)]
        if i > 0:
            text = text.replace(".", f" {i}.", 1) if i % 3 == 0 else text
        all_texts.append(text)
        all_categories.append("Teknoloji")
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'text': all_texts,
        'category': all_categories
    })
    
    # Veriyi karıştır
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def save_dataset(df, filepath):
    """
    Dataset'i CSV formatında kaydeder.
    
    Args:
        df: DataFrame
        filepath: Kayıt edilecek dosya yolu
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Dataset kaydedildi: {filepath}")
    print(f"Toplam örnek sayısı: {len(df)}")
    print(f"Kategori dağılımı:\n{df['category'].value_counts()}")

if __name__ == "__main__":
    # Dataset oluştur
    dataset = create_sample_dataset(num_samples_per_category=500)
    
    # Kaydet
    save_dataset(dataset, "data/raw/dataset.csv")



