"""
Vektörleştirme Modülü
Üç farklı yöntem: Bag of Words, Word2Vec, BERT
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import os
import pickle
from tqdm import tqdm

# Gensim'i opsiyonel yap (Python 3.14 ile uyumsuz olabilir)
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Uyarı: gensim yüklenemedi. Word2Vec vektörleştirme kullanılamayacak.")
    print("Python 3.14 ile gensim uyumsuz. Python 3.11 veya 3.12 kullanmanız önerilir.")

class BagOfWordsVectorizer:
    """Bag of Words vektörleştirme sınıfı"""
    
    def __init__(self, max_features=5000):
        """
        Args:
            max_features: Maksimum özellik sayısı
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigram ve bigram
            min_df=2,  # En az 2 dokümanda geçmeli
            max_df=0.95  # En fazla %95 dokümanda geçmeli
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Metinleri vektörleştirir ve modeli eğitir.
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            scipy.sparse matrix: Vektörleştirilmiş metinler
        """
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return vectors
    
    def transform(self, texts):
        """
        Yeni metinleri vektörleştirir (model önceden eğitilmiş olmalı).
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            scipy.sparse matrix: Vektörleştirilmiş metinler
        """
        if not self.is_fitted:
            raise ValueError("Model önce fit edilmelidir!")
        return self.vectorizer.transform(texts)
    
    def save(self, filepath):
        """Modeli kaydeder"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Bag of Words modeli kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Modeli yükler"""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"Bag of Words modeli yüklendi: {filepath}")

class TFIDFVectorizer:
    """TF-IDF vektörleştirme sınıfı"""
    
    def __init__(self, max_features=5000):
        """
        Args:
            max_features: Maksimum özellik sayısı
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigram ve bigram
            min_df=2,  # En az 2 dokümanda geçmeli
            max_df=0.95  # En fazla %95 dokümanda geçmeli
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Metinleri vektörleştirir ve modeli eğitir.
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            scipy.sparse matrix: Vektörleştirilmiş metinler
        """
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return vectors
    
    def transform(self, texts):
        """
        Yeni metinleri vektörleştirir (model önceden eğitilmiş olmalı).
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            scipy.sparse matrix: Vektörleştirilmiş metinler
        """
        if not self.is_fitted:
            raise ValueError("Model önce fit edilmelidir!")
        return self.vectorizer.transform(texts)
    
    def save(self, filepath):
        """Modeli kaydeder"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"TF-IDF modeli kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Modeli yükler"""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"TF-IDF modeli yüklendi: {filepath}")

class NGramVectorizer:
    """N-Gram vektörleştirme sınıfı (1-3 gram: unigram, bigram, trigram)"""
    
    def __init__(self, max_features=5000):
        """
        Args:
            max_features: Maksimum özellik sayısı
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Unigram, bigram ve trigram
            min_df=2,  # En az 2 dokümanda geçmeli
            max_df=0.95  # En fazla %95 dokümanda geçmeli
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Metinleri vektörleştirir ve modeli eğitir.
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            scipy.sparse matrix: Vektörleştirilmiş metinler
        """
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return vectors
    
    def transform(self, texts):
        """
        Yeni metinleri vektörleştirir (model önceden eğitilmiş olmalı).
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            scipy.sparse matrix: Vektörleştirilmiş metinler
        """
        if not self.is_fitted:
            raise ValueError("Model önce fit edilmelidir!")
        return self.vectorizer.transform(texts)
    
    def save(self, filepath):
        """Modeli kaydeder"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"N-Gram modeli kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Modeli yükler"""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"N-Gram modeli yüklendi: {filepath}")

class Word2VecVectorizer:
    """Word2Vec vektörleştirme sınıfı"""
    
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "gensim yüklü değil. Word2Vec kullanılamaz. "
                "Python 3.11 veya 3.12 kullanarak gensim'i yükleyebilirsiniz."
            )
        """
        Args:
            vector_size: Vektör boyutu
            window: Pencere boyutu
            min_count: Minimum kelime sayısı
            workers: İşlemci sayısı
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Metinleri tokenize eder, Word2Vec modelini eğitir ve belge vektörleri oluşturur.
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            numpy array: Belge vektörleri
        """
        # Tokenize et
        tokenized_texts = [simple_preprocess(text) for text in texts]
        
        # Word2Vec modelini eğit
        print("Word2Vec modeli eğitiliyor...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=0  # CBOW kullan
        )
        
        self.is_fitted = True
        
        # Belge vektörleri oluştur (ortalama pooling)
        doc_vectors = []
        for tokens in tokenized_texts:
            if len(tokens) > 0:
                # Her kelime için vektör al ve ortala
                word_vectors = [
                    self.model.wv[word] 
                    for word in tokens 
                    if word in self.model.wv
                ]
                if len(word_vectors) > 0:
                    doc_vector = np.mean(word_vectors, axis=0)
                else:
                    # Eğer hiç kelime bulunamazsa sıfır vektör
                    doc_vector = np.zeros(self.vector_size)
            else:
                doc_vector = np.zeros(self.vector_size)
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)
    
    def transform(self, texts):
        """
        Yeni metinleri vektörleştirir (model önceden eğitilmiş olmalı).
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            numpy array: Belge vektörleri
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model önce fit edilmelidir!")
        
        # Tokenize et
        tokenized_texts = [simple_preprocess(text) for text in texts]
        
        # Belge vektörleri oluştur
        doc_vectors = []
        for tokens in tokenized_texts:
            if len(tokens) > 0:
                word_vectors = [
                    self.model.wv[word] 
                    for word in tokens 
                    if word in self.model.wv
                ]
                if len(word_vectors) > 0:
                    doc_vector = np.mean(word_vectors, axis=0)
                else:
                    doc_vector = np.zeros(self.vector_size)
            else:
                doc_vector = np.zeros(self.vector_size)
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)
    
    def save(self, filepath):
        """Modeli kaydeder"""
        if self.model is None:
            raise ValueError("Kaydedilecek model yok!")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Word2Vec modeli kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Modeli yükler"""
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(filepath)
        self.is_fitted = True
        self.vector_size = self.model.wv.vector_size
        print(f"Word2Vec modeli yüklendi: {filepath}")

class BERTVectorizer:
    """BERT vektörleştirme sınıfı"""
    
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased", batch_size=16, max_length=128):
        """
        Args:
            model_name: BERT model adı
            batch_size: Batch boyutu
            max_length: Maksimum token uzunluğu
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"BERT modeli yükleniyor: {model_name}")
        print(f"Cihaz: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Inference modu
        
        self.is_fitted = True  # BERT önceden eğitilmiş olduğu için her zaman hazır
    
    def _get_embeddings_batch(self, texts):
        """
        Bir batch metin için embedding'leri alır.
        
        Args:
            texts: Metin listesi
        
        Returns:
            numpy array: Embedding vektörleri
        """
        # Tokenize et
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Cihaza taşı
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Model çıktısı
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Ortalama pooling (attention mask ile ağırlıklandırılmış)
        last_hidden_state = outputs.last_hidden_state
        embeddings = []
        
        for i in range(last_hidden_state.shape[0]):
            # Attention mask ile ağırlıklandırılmış ortalama
            mask = attention_mask[i].unsqueeze(-1).expand(last_hidden_state[i].shape).float()
            masked_embeddings = last_hidden_state[i] * mask
            summed = torch.sum(masked_embeddings, dim=0)
            summed_mask = torch.clamp(mask.sum(0), min=1e-9)
            mean_pooled = summed / summed_mask
            embeddings.append(mean_pooled.cpu().numpy())
        
        return np.array(embeddings)
    
    def fit_transform(self, texts):
        """
        Metinleri BERT ile vektörleştirir.
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            numpy array: Belge vektörleri
        """
        texts = list(texts)
        all_embeddings = []
        
        print("BERT ile vektörleştirme yapılıyor...")
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._get_embeddings_batch(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def transform(self, texts):
        """
        Yeni metinleri vektörleştirir (BERT her zaman hazır olduğu için fit_transform ile aynı).
        
        Args:
            texts: Metin listesi veya Series
        
        Returns:
            numpy array: Belge vektörleri
        """
        return self.fit_transform(texts)
    
    def save(self, filepath):
        """Model bilgilerini kaydeder (model zaten transformers'dan yüklenecek)"""
        model_info = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_length': self.max_length
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"BERT model bilgileri kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Model bilgilerini yükler ve modeli tekrar yükler"""
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)
        
        self.model_name = model_info['model_name']
        self.batch_size = model_info['batch_size']
        self.max_length = model_info['max_length']
        
        # Modeli tekrar yükle
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"BERT modeli yüklendi: {self.model_name}")



