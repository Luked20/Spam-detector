from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np
from typing import Dict, Any
import re
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

# Carregar o modelo, vetorizador e scaler
model_dir = Path("models")
model_path = model_dir / "spam_detector.joblib"
vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
scaler_path = model_dir / "feature_scaler.joblib"

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("Erro: Modelo, vetorizador ou scaler não encontrado!")
    model = None
    vectorizer = None
    scaler = None

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

app = FastAPI(
    title="API de Detecção de Spam",
    description="API para classificar e-mails como spam ou não spam usando Naïve Bayes",
    version="1.0.0"
)

class Email(BaseModel):
    content: str

class PredictionResponse(BaseModel):
    prediction: int  # 0 para ham (não spam), 1 para spam
    probability: float
    message: str

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

@app.get("/")
async def root():
    """
    Endpoint raiz
    """
    return {
        "message": "API de Detecção de Spam",
        "status": "online",
        "endpoints": {
            "/predict": "POST - Classifica um e-mail como spam ou não spam",
            "/health": "GET - Verifica a saúde da API"
        }
    }

@app.get("/health")
async def health_check():
    """
    Verifica a saúde da API
    """
    if model is None or vectorizer is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo, vetorizador ou scaler não carregado. A API não está pronta para uso."
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(email: Email) -> Dict[str, Any]:
    """
    Classifica um e-mail como spam ou não spam
    """
    if model is None or vectorizer is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo, vetorizador ou scaler não carregado. A API não está pronta para uso."
        )
    
    # Pré-processar o texto
    processed_text = preprocess_text(email.content)
    
    # Vetorizar o texto
    text_vector = vectorizer.transform([processed_text])
    
    # Extrair features adicionais
    additional_features = extract_features(email.content)
    additional_features_array = np.array([[v for v in additional_features.values()]])
    
    # Normalizar features adicionais
    additional_features_scaled = scaler.transform(additional_features_array)
    
    # Combinar features
    combined_features = hstack([text_vector, additional_features_scaled])
    
    # Fazer a predição
    prediction = model.predict(combined_features)[0]
    probability = model.predict_proba(combined_features)[0][1]  # Probabilidade de ser spam
    
    # Determinar a mensagem
    message = "Este e-mail é spam" if prediction == 1 else "Este e-mail não é spam"
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "message": message
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 