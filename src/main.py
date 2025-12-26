"""
Ana Pipeline Dosyası
Tüm modülleri koordine eder ve sonuçları görselleştirir.
"""

import os
import sys


# Proje kök dizinini sys.path'e ekle (import'lardan önce)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modülleri import et
from data_preparation import create_sample_dataset, save_dataset
from text_cleaning import clean_dataset, load_cleaned_dataset
from vectorization import (
    BagOfWordsVectorizer, 
    TFIDFVectorizer,
    NGramVectorizer,
    Word2VecVectorizer, 
    BERTVectorizer
)
from model_training import ModelTrainer

def check_and_create_raw_data():
    """Ham veri yoksa oluşturur"""
    raw_data_path = "data/raw/dataset.csv"
    
    if not os.path.exists(raw_data_path):
        print("Ham veri bulunamadı, oluşturuluyor...")
        dataset = create_sample_dataset(num_samples_per_category=500)
        save_dataset(dataset, raw_data_path)
    else:
        print(f"Ham veri mevcut: {raw_data_path}")

def check_and_clean_data():
    """Temizlenmiş veri yoksa oluşturur"""
    cleaned_data_path = "data/cleaned/cleaned_dataset.csv"
    raw_data_path = "data/raw/dataset.csv"
    
    if not os.path.exists(cleaned_data_path):
        print("Temizlenmiş veri bulunamadı, oluşturuluyor...")
        clean_dataset(raw_data_path, cleaned_data_path)
    else:
        print(f"Temizlenmiş veri mevcut: {cleaned_data_path}")

def main():
    """Ana pipeline fonksiyonu"""
    print("="*80)
    print("TÜRKÇE METİN SINIFLANDIRMA PROJESİ")
    print("="*80)
    
    # 1. Ham veriyi kontrol et ve oluştur
    print("\n[1/5] Ham veri kontrolü...")
    check_and_create_raw_data()
    
    # 2. Temizlenmiş veriyi kontrol et ve oluştur
    print("\n[2/5] Temizlenmiş veri kontrolü...")
    check_and_clean_data()
    
    # 3. Temizlenmiş veriyi yükle
    print("\n[3/5] Temizlenmiş veri yükleniyor...")
    df = load_cleaned_dataset("data/cleaned/cleaned_dataset.csv")
    
    X = df['text']
    y = df['category']
    
    print(f"Toplam örnek sayısı: {len(df)}")
    print(f"Kategori dağılımı:\n{y.value_counts()}")
    
    # 4. Vektörleştirme ve model eğitimi
    print("\n[4/5] Vektörleştirme ve model eğitimi...")
    
    # Vektörleştirme yöntemlerini oluştur (Word2Vec opsiyonel)
    vectorization_methods = {
        'Bag of Words': BagOfWordsVectorizer(max_features=5000),
        'TF-IDF': TFIDFVectorizer(max_features=5000),
        'N-Gram': NGramVectorizer(max_features=5000),
        'BERT': BERTVectorizer(model_name="dbmdz/bert-base-turkish-cased", batch_size=16)
    }
    
    # Word2Vec'i sadece gensim yüklüyse ekle
    try:
        vectorization_methods['Word2Vec'] = Word2VecVectorizer(vector_size=100, window=5, min_count=2)
    except (ImportError, NameError) as e:
        print(f"\nUyarı: Word2Vec atlandı - {str(e)}")
        print("Diğer vektörleştirme yöntemleri ile devam ediliyor...\n")
    
    all_results = {}
    
    for method_name, vectorizer in vectorization_methods.items():
        print(f"\n{'='*80}")
        print(f"VEKTÖRLEŞTİRME: {method_name}")
        print(f"{'='*80}")
        
        try:
            # Vektörleştir
            print(f"{method_name} ile vektörleştirme yapılıyor...")
            X_vectorized = vectorizer.fit_transform(X)
            
            # Sparse matrix'i dense'e çevir (gerekirse)
            if hasattr(X_vectorized, 'toarray'):
                X_vectorized = X_vectorized.toarray()
            
            print(f"Vektör boyutu: {X_vectorized.shape}")
            
            # Model eğitimi
            print(f"\n{method_name} vektörleri ile model eğitimi...")
            trainer = ModelTrainer(X_vectorized, y, test_size=0.2, random_state=42)
            trainer.train_all_models()
            
            # Sonuçları yazdır
            trainer.print_summary()
            
            # Görselleştirmeler
            print(f"\n{method_name} için görselleştirmeler oluşturuluyor...")
            
            # Confusion matrix
            trainer.plot_confusion_matrices(
                save_path=f"results/confusion_matrix_{method_name.replace(' ', '_')}.png"
            )
            
            # Accuracy karşılaştırması
            trainer.plot_accuracy_comparison(
                save_path=f"results/accuracy_comparison_{method_name.replace(' ', '_')}.png"
            )
            
            # Sonuçları kaydet
            trainer.save_results("results", method_name.replace(' ', '_'))
            
            # Modelleri ve vektörleştiriciyi kaydet
            trainer.save_all_models("models", method_name.replace(' ', '_'), vectorizer)
            
            # Sonuçları sakla
            all_results[method_name] = trainer.get_all_results()
            
        except Exception as e:
            print(f"{method_name} için hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. Tüm yöntemlerin karşılaştırması
    print("\n[5/5] Tüm vektörleştirme yöntemlerinin karşılaştırması...")
    
    if all_results:
        # Her yöntem için en iyi modeli bul
        comparison_data = []
        
        for method_name, results in all_results.items():
            for model_name, result in results.items():
                comparison_data.append({
                    'Vektörleştirme': method_name,
                    'Model': model_name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['classification_report']['macro avg']['precision'],
                    'Recall': result['classification_report']['macro avg']['recall'],
                    'F1-Score': result['classification_report']['macro avg']['f1-score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # En iyi sonuçları göster
        print("\n" + "="*80)
        print("TÜM YÖNTEMLERİN KARŞILAŞTIRMASI")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # En iyi modeli bul
        best_row = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        print(f"\nEn iyi model:")
        print(f"  Vektörleştirme: {best_row['Vektörleştirme']}")
        print(f"  Model: {best_row['Model']}")
        print(f"  Accuracy: {best_row['Accuracy']:.4f}")
        print(f"  F1-Score: {best_row['F1-Score']:.4f}")
        
        # Karşılaştırma görselleştirmesi
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy karşılaştırması
        pivot_acc = comparison_df.pivot(index='Model', columns='Vektörleştirme', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title('Accuracy Karşılaştırması', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(title='Vektörleştirme')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Precision karşılaştırması
        pivot_prec = comparison_df.pivot(index='Model', columns='Vektörleştirme', values='Precision')
        pivot_prec.plot(kind='bar', ax=axes[0, 1], rot=45)
        axes[0, 1].set_title('Precision Karşılaştırması', fontweight='bold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend(title='Vektörleştirme')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Recall karşılaştırması
        pivot_rec = comparison_df.pivot(index='Model', columns='Vektörleştirme', values='Recall')
        pivot_rec.plot(kind='bar', ax=axes[1, 0], rot=45)
        axes[1, 0].set_title('Recall Karşılaştırması', fontweight='bold')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend(title='Vektörleştirme')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # F1-Score karşılaştırması
        pivot_f1 = comparison_df.pivot(index='Model', columns='Vektörleştirme', values='F1-Score')
        pivot_f1.plot(kind='bar', ax=axes[1, 1], rot=45)
        axes[1, 1].set_title('F1-Score Karşılaştırması', fontweight='bold')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].legend(title='Vektörleştirme')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/final_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\nKarşılaştırma görselleştirmesi kaydedildi: results/final_comparison.png")
        plt.show()
        
        # CSV olarak kaydet
        comparison_df.to_csv("results/final_comparison.csv", index=False, encoding='utf-8')
        print(f"Karşılaştırma tablosu kaydedildi: results/final_comparison.csv")
    
    print("\n" + "="*80)
    print("PROJE TAMAMLANDI!")
    print("="*80)
    print("\nSonuçlar 'results/' klasöründe bulunabilir.")

if __name__ == "__main__":
    # Çalışma dizinini proje kök dizinine ayarla
    # (project_root zaten yukarıda hesaplandı)
    os.chdir(project_root)
    
    main()



