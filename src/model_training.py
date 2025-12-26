"""
Model Eğitimi ve Değerlendirme Modülü
Decision Tree, Naive Bayes, SVM, MLP modellerini eğitir ve değerlendirir.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

class ModelTrainer:
    """Model eğitimi ve değerlendirme sınıfı"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Args:
            X: Özellik vektörleri (numpy array veya sparse matrix)
            y: Etiketler
            test_size: Test seti oranı
            random_state: Rastgelelik tohumu
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Modeller
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=20),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', random_state=random_state, probability=True),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # Eğitilmiş modeller
        self.trained_models = {}
        
        # Sonuçlar
        self.results = {}
    
    def train_all_models(self):
        """Tüm modelleri eğitir"""
        print("Modeller eğitiliyor...")
        
        for name, model in self.models.items():
            print(f"\n{name} eğitiliyor...")
            try:
                # Eğit
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                
                # Test et
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Sonuçları kaydet
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'y_pred': y_pred,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'classification_report': classification_report(
                        self.y_test, y_pred, output_dict=True
                    )
                }
                
                print(f"{name} Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"{name} eğitilirken hata: {str(e)}")
                continue
    
    def evaluate_model(self, model_name):
        """
        Belirli bir modeli değerlendirir.
        
        Args:
            model_name: Model adı
        
        Returns:
            dict: Değerlendirme sonuçları
        """
        if model_name not in self.results:
            raise ValueError(f"Model bulunamadı: {model_name}")
        
        return self.results[model_name]
    
    def get_all_results(self):
        """Tüm sonuçları döndürür"""
        return self.results
    
    def plot_confusion_matrices(self, save_path=None):
        """Tüm modeller için confusion matrix görselleştirir"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )
            
            axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_accuracy_comparison(self, save_path=None):
        """Model accuracy karşılaştırması görselleştirir"""
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # Değerleri çubukların üzerine yaz
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{acc:.4f}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy karşılaştırması kaydedildi: {save_path}")
        
        plt.show()
    
    def save_results(self, directory, vectorization_method):
        """
        Sonuçları kaydeder.
        
        Args:
            directory: Kayıt dizini
            vectorization_method: Vektörleştirme yöntemi adı
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sonuçları DataFrame olarak kaydet
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision (macro avg)': [
                r['classification_report']['macro avg']['precision']
                for r in self.results.values()
            ],
            'Recall (macro avg)': [
                r['classification_report']['macro avg']['recall']
                for r in self.results.values()
            ],
            'F1-Score (macro avg)': [
                r['classification_report']['macro avg']['f1-score']
                for r in self.results.values()
            ]
        })
        
        results_file = os.path.join(
            directory,
            f'results_{vectorization_method}_{timestamp}.csv'
        )
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        print(f"Sonuçlar kaydedildi: {results_file}")
        
        # Modelleri kaydet
        for name, result in self.results.items():
            model_file = os.path.join(
                directory,
                f'model_{vectorization_method}_{name.replace(" ", "_")}_{timestamp}.pkl'
            )
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)
        
        return results_file
    
    def save_all_models(self, directory, vectorization_method, vectorizer):
        """
        Tüm modelleri ve vektörleştiriciyi kaydeder.
        
        Args:
            directory: Kayıt dizini
            vectorization_method: Vektörleştirme yöntemi adı
            vectorizer: Vektörleştirici nesnesi
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = vectorization_method.replace(" ", "_")
        
        # Vektörleştiriciyi kaydet
        vectorizer_file = os.path.join(directory, f'vectorizer_{method_name}_{timestamp}.pkl')
        try:
            vectorizer.save(vectorizer_file)
        except Exception as e:
            # Eğer save metodu yoksa pickle ile kaydet
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"Vektörleştirici kaydedildi: {vectorizer_file}")
        
        # Her model için kaydet
        model_info = {}
        for name, result in self.results.items():
            model_name = name.replace(" ", "_")
            model_file = os.path.join(
                directory,
                f'model_{method_name}_{model_name}_{timestamp}.pkl'
            )
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)
            
            # Model bilgilerini kaydet
            model_info[model_name] = {
                'file': model_file,
                'accuracy': result['accuracy'],
                'vectorizer_file': vectorizer_file
            }
        
        # Model bilgilerini JSON olarak kaydet
        import json
        info_file = os.path.join(directory, f'models_info_{method_name}_{timestamp}.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"Tüm modeller kaydedildi: {directory}")
        return model_info
    
    @staticmethod
    def load_model(model_file):
        """
        Kaydedilmiş modeli yükler.
        
        Args:
            model_file: Model dosya yolu
        
        Returns:
            Eğitilmiş model
        """
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        return model
    
    @staticmethod
    def load_vectorizer(vectorizer_file):
        """
        Kaydedilmiş vektörleştiriciyi yükler.
        
        Args:
            vectorizer_file: Vektörleştirici dosya yolu
        
        Returns:
            Vektörleştirici nesnesi
        """
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    
    def print_summary(self):
        """Tüm sonuçları özet olarak yazdırır"""
        print("\n" + "="*80)
        print("MODEL DEĞERLENDİRME SONUÇLARI")
        print("="*80)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Precision (macro avg): {result['classification_report']['macro avg']['precision']:.4f}")
            print(f"  Recall (macro avg): {result['classification_report']['macro avg']['recall']:.4f}")
            print(f"  F1-Score (macro avg): {result['classification_report']['macro avg']['f1-score']:.4f}")
        
        print("\n" + "="*80)



