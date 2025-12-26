"""
Tahmin Sistemi
Kullanıcıdan metin alır ve kategori tahmini yapar.
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
from collections import Counter

# Proje kök dizinini sys.path'e ekle (import'lardan önce)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from text_cleaning import clean_text
from model_training import ModelTrainer

def load_all_models():
    """
    Tüm kaydedilmiş modelleri ve vektörleştiricileri yükler.
    
    Returns:
        dict: {method_name: {model_name: {'model': model, 'vectorizer': vectorizer, 'accuracy': acc}}}
    """
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("HATA: 'models/' klasörü bulunamadı!")
        print("Lütfen önce 'python3 src/main.py' komutunu çalıştırarak modelleri eğitin.")
        return None
    
    # Tüm models_info dosyalarını bul
    info_files = glob.glob(os.path.join(models_dir, "models_info_*.json"))
    
    if not info_files:
        print("HATA: Eğitilmiş model bulunamadı!")
        print("Lütfen önce 'python3 src/main.py' komutunu çalıştırarak modelleri eğitin.")
        return None
    
    all_models = {}
    
    for info_file in info_files:
        with open(info_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Vektörleştirme yöntemini dosya adından çıkar
        method_name = os.path.basename(info_file).replace('models_info_', '').split('_')[0]
        method_name = method_name.replace('_', ' ')
        
        if method_name not in all_models:
            all_models[method_name] = {}
        
        # Vektörleştiriciyi yükle
        vectorizer_file = None
        for model_data in model_info.values():
            if 'vectorizer_file' in model_data:
                vectorizer_file = model_data['vectorizer_file']
                break
        
        if vectorizer_file and os.path.exists(vectorizer_file):
            try:
                vectorizer = ModelTrainer.load_vectorizer(vectorizer_file)
                # Eğer load metodu varsa kullan
                if hasattr(vectorizer, 'load'):
                    vectorizer.load(vectorizer_file)
            except Exception as e:
                print(f"Uyarı: {method_name} vektörleştiricisi yüklenirken hata: {e}")
                continue
        
        # Her modeli yükle
        for model_name, data in model_info.items():
            model_file = data['file']
            accuracy = data.get('accuracy', 0.0)
            
            if os.path.exists(model_file):
                try:
                    model = ModelTrainer.load_model(model_file)
                    all_models[method_name][model_name] = {
                        'model': model,
                        'vectorizer': vectorizer,
                        'accuracy': accuracy
                    }
                except Exception as e:
                    print(f"Uyarı: {model_name} modeli yüklenirken hata: {e}")
                    continue
    
    return all_models

def predict_text(text, all_models):
    """
    Verilen metin için tüm modellerden tahmin yapar.
    
    Args:
        text: Tahmin yapılacak metin
        all_models: Yüklenmiş modeller sözlüğü
    
    Returns:
        list: [(method_name, model_name, prediction, accuracy), ...]
    """
    # Metni temizle
    cleaned_text = clean_text(text)
    
    if not cleaned_text or len(cleaned_text.strip()) == 0:
        return None
    
    predictions = []
    
    for method_name, models in all_models.items():
        for model_name, model_data in models.items():
            try:
                model = model_data['model']
                vectorizer = model_data['vectorizer']
                accuracy = model_data['accuracy']
                
                # Metni vektörleştir
                # Tek bir metin için listeye çevir
                text_vector = vectorizer.transform([cleaned_text])
                
                # Sparse matrix'i dense'e çevir (gerekirse)
                if hasattr(text_vector, 'toarray'):
                    text_vector = text_vector.toarray()
                
                # Tahmin yap
                prediction = model.predict(text_vector)[0]
                
                predictions.append((method_name, model_name, prediction, accuracy))
                
            except Exception as e:
                print(f"Uyarı: {method_name} - {model_name} tahmininde hata: {e}")
                continue
    
    return predictions

def main():
    """Ana tahmin fonksiyonu"""
    print("="*80)
    print("TÜRKÇE METİN SINIFLANDIRMA - TAHMİN SİSTEMİ")
    print("="*80)
    
    # Modelleri yükle
    print("\nModeller yükleniyor...")
    all_models = load_all_models()
    
    if not all_models:
        return
    
    print(f"\n{len(all_models)} vektörleştirme yöntemi ve toplam {sum(len(models) for models in all_models.values())} model yüklendi.")
    
    # Kullanıcıdan metin al
    print("\n" + "="*80)
    print("Lütfen sınıflandırılacak metni girin (Çıkmak için 'q' yazın):")
    print("="*80)
    
    while True:
        text = input("\nMetin: ").strip()
        
        if text.lower() == 'q':
            print("Çıkılıyor...")
            break
        
        if not text:
            print("Lütfen geçerli bir metin girin!")
            continue
        
        # Tahmin yap
        predictions = predict_text(text, all_models)
        
        if not predictions:
            print("Tahmin yapılamadı. Lütfen tekrar deneyin.")
            continue
        
        # Kategorilere göre grupla ve accuracy ortalamasını hesapla
        category_votes = {}
        category_accuracies = {}
        
        for method_name, model_name, prediction, accuracy in predictions:
            if prediction not in category_votes:
                category_votes[prediction] = []
                category_accuracies[prediction] = []
            
            category_votes[prediction].append((method_name, model_name))
            category_accuracies[prediction].append(accuracy)
        
        # En çok oy alan kategoriyi bul
        most_common_category = max(category_votes.items(), key=lambda x: len(x[1]))
        final_category = most_common_category[0]
        
        # Bu kategori için ortalama accuracy hesapla
        avg_accuracy = np.mean(category_accuracies[final_category])
        
        # Sonuçları göster
        print("\n" + "="*80)
        print("TAHMIN SONUÇLARI")
        print("="*80)
        print(f"\nGirilen Cümle:")
        print(f"  {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\nKategori: {final_category}")
        print(f"Accuracy Score: {avg_accuracy:.4f}")
        print("="*80)

if __name__ == "__main__":
    # Çalışma dizinini proje kök dizinine ayarla
    os.chdir(project_root)
    
    main()

