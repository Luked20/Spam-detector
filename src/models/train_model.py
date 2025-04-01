import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

# Download recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Lista de palavras comuns em spam
SPAM_WORDS = {
    'free', 'offer', 'limited', 'time', 'deal', 'discount', 'save', 'money',
    'guaranteed', 'promise', 'winner', 'prize', 'award', 'congratulations',
    'urgent', 'important', 'exclusive', 'special', 'amazing', 'incredible',
    'unbelievable', 'act now', 'click here', 'order now', 'buy now',
    'limited time', 'special offer', 'best price', 'lowest price',
    'huge discount', 'massive discount', 'incredible offer', 'amazing deal',
    'once in a lifetime', 'don\'t miss out', 'hurry up', 'expires soon',
    'limited stock', 'while supplies last', 'best deal', 'great offer',
    'special price', 'exclusive offer', 'premium', 'luxury', 'elite',
    'vip', 'exclusive', 'limited edition', 'special edition', 'premium offer'
}

def extract_features(text: str) -> Dict[str, float]:
    """
    Extrai features adicionais do texto
    """
    features = {}
    
    # Features básicas
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['special_chars'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text))
    features['numbers'] = len(re.findall(r'\d+', text))
    words = text.split()
    features['uppercase_words'] = sum(1 for w in words if w.isupper())
    features['exclamation_marks'] = text.count('!')
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    
    # Features adicionais para spam
    text_lower = text.lower()
    
    # Contagem de palavras de spam
    spam_word_count = sum(1 for word in words if word.lower() in SPAM_WORDS)
    features['spam_word_count'] = spam_word_count
    
    # Razão de palavras de spam
    features['spam_word_ratio'] = spam_word_count / len(words) if len(words) > 0 else 0
    
    # Contagem de frases de spam
    spam_phrases = sum(1 for phrase in SPAM_WORDS if phrase in text_lower)
    features['spam_phrase_count'] = spam_phrases
    
    # Contagem de números em sequência
    features['number_sequences'] = len(re.findall(r'\d{3,}', text))
    
    # Contagem de caracteres especiais em sequência
    features['special_char_sequences'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]{2,}', text))
    
    # Contagem de URLs suspeitos
    features['suspicious_urls'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    
    # Contagem de palavras em maiúsculas em sequência
    features['uppercase_sequences'] = len(re.findall(r'[A-Z]{3,}', text))
    
    # Contagem de pontos de exclamação em sequência
    features['exclamation_sequences'] = len(re.findall(r'!{2,}', text))
    
    # Contagem de palavras repetidas
    word_freq = {}
    for word in words:
        word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
    features['repeated_words'] = sum(1 for freq in word_freq.values() if freq > 2)
    
    return features

def preprocess_text(text: str) -> str:
    """
    Pré-processa o texto
    """
    # Converter para minúsculas
    text = text.lower()
    
    # Remover URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
    
    # Remover números
    text = re.sub(r'\d+', 'NUM', text)
    
    # Remover caracteres especiais exceto pontuação básica
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Adicionar palavras de spam como features
    spam_features = []
    for word in SPAM_WORDS:
        if word in text:
            spam_features.append(f"SPAM_{word}")
    
    return text + " " + " ".join(spam_features)

def load_data():
    """
    Carrega os dados
    """
    data_dir = Path("data")
    data_path = data_dir / "processed_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {data_path}")
    
    # Carregar dados
    df = pd.read_csv(data_path)
    
    # Extrair features adicionais
    print("Extraindo features adicionais...")
    additional_features = df['message'].apply(extract_features)
    additional_features_df = pd.DataFrame(additional_features.tolist())
    
    # Pré-processar mensagens
    print("Pré-processando mensagens...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Separar features e target
    X_text = df['processed_message']
    y = df['label']
    
    return X_text, y, additional_features_df

def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusão'):
    """
    Plota a matriz de confusão
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_model():
    """
    Treina o modelo Naïve Bayes com validação cruzada e otimização de hiperparâmetros
    """
    print("Carregando dados...")
    X_text, y, additional_features = load_data()
    
    # Separar dados em treino e teste
    X_train_text, X_test_text, y_train, y_test, X_train_add, X_test_add = train_test_split(
        X_text, y, additional_features, test_size=0.2, random_state=42
    )
    
    print("\nDivisão dos dados:")
    print(f"Treino: {len(X_train_text)} amostras")
    print(f"Teste: {len(X_test_text)} amostras")
    
    # Criar e treinar o vetorizador TF-IDF
    print("\nCriando e treinando vetorizador TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=15000,  # Aumentado para capturar mais palavras
        ngram_range=(1, 3),  # Incluindo trigramas
        stop_words='english',
        min_df=2,  # Ignorar palavras que aparecem em menos de 2 documentos
        max_df=0.95,  # Ignorar palavras que aparecem em mais de 95% dos documentos
        use_idf=True,  # Usar IDF
        smooth_idf=True,  # Suavizar IDF
        sublinear_tf=True  # Usar TF sublinear
    )
    
    # Vetorizar os dados
    X_train_vectorized = vectorizer.fit_transform(X_train_text)
    X_test_vectorized = vectorizer.transform(X_test_text)
    
    # Normalizar features adicionais usando MinMaxScaler
    scaler = MinMaxScaler()
    X_train_add_scaled = scaler.fit_transform(X_train_add)
    X_test_add_scaled = scaler.transform(X_test_add)
    
    # Combinar features de texto com features adicionais
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_vectorized, X_train_add_scaled])
    X_test_combined = hstack([X_test_vectorized, X_test_add_scaled])
    
    # Definir parâmetros para Grid Search
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }
    
    # Criar modelo base
    base_model = MultinomialNB()
    
    # Criar Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nIniciando Grid Search...")
    grid_search.fit(X_train_combined, y_train)
    
    # Melhor modelo
    best_model = grid_search.best_estimator_
    
    print("\nMelhores parâmetros encontrados:")
    print(grid_search.best_params_)
    print(f"Melhor score: {grid_search.best_score_:.4f}")
    
    # Avaliar no conjunto de teste
    y_pred = best_model.predict(X_test_combined)
    
    print("\nRelatório de classificação no conjunto de teste:")
    print(classification_report(y_test, y_pred))
    
    # Plotar matriz de confusão
    plot_confusion_matrix(y_test, y_pred)
    
    # Salvar o modelo, vetorizador e scaler
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "spam_detector.joblib"
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
    scaler_path = model_dir / "feature_scaler.joblib"
    
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModelo salvo em: {model_path}")
    print(f"Vetorizador salvo em: {vectorizer_path}")
    print(f"Scaler salvo em: {scaler_path}")
    
    # Análise de features mais importantes
    feature_importance = pd.DataFrame({
        'feature': list(vectorizer.get_feature_names_out()) + list(additional_features.columns),
        'importance': best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 10 features mais importantes para detecção de spam:")
    print(feature_importance.head(10))
    
    # Plotar top 20 features mais importantes
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Features Mais Importantes para Detecção de Spam')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    train_model() 