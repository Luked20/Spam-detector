import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from pathlib import Path
import numpy as np

# Download recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
    def clean_text(self, text):
        """
        Limpa o texto removendo caracteres especiais e números
        """
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokeniza o texto
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords dos tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """
        Aplica lemmatization nos tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        """
        Aplica todo o pipeline de pré-processamento
        """
        # Limpar texto
        cleaned_text = self.clean_text(text)
        
        # Tokenizar
        tokens = self.tokenize(cleaned_text)
        
        # Remover stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatizar
        tokens = self.lemmatize(tokens)
        
        # Juntar tokens de volta em texto
        return ' '.join(tokens)
    
    def vectorize_texts(self, texts):
        """
        Converte textos em vetores TF-IDF
        """
        return self.tfidf_vectorizer.fit_transform(texts)

def process_dataset():
    """
    Processa o dataset completo
    """
    # Definir diretório de dados
    data_dir = Path("data")
    
    # Ler dados processados
    input_path = data_dir / "processed_data.csv"
    if not input_path.exists():
        print(f"Arquivo de dados não encontrado em: {input_path}")
        return
    
    # Carregar dados
    df = pd.read_csv(input_path)
    
    # Criar preprocessor
    preprocessor = TextPreprocessor()
    
    print("Iniciando pré-processamento...")
    
    # Aplicar pré-processamento em todas as mensagens
    df['processed_message'] = df['message'].apply(preprocessor.preprocess_text)
    
    # Vetorizar textos
    print("Vetorizando textos...")
    tfidf_matrix = preprocessor.vectorize_texts(df['processed_message'])
    
    # Criar DataFrame com vetores TF-IDF
    feature_names = preprocessor.tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names
    )
    
    # Combinar com labels
    final_df = pd.concat([
        df[['label', 'message', 'processed_message']],
        tfidf_df
    ], axis=1)
    
    # Salvar dados processados
    output_path = data_dir / "vectorized_data.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\nDados processados salvos em: {output_path}")
    print(f"\nEstatísticas do processamento:")
    print(f"Total de amostras: {len(df)}")
    print(f"Total de features TF-IDF: {len(feature_names)}")
    print(f"\nExemplo de mensagem original:\n{df['message'].iloc[0]}")
    print(f"\nExemplo de mensagem pré-processada:\n{df['processed_message'].iloc[0]}")
    print(f"\nTop 10 features mais importantes:")
    tfidf_means = tfidf_df.mean().sort_values(ascending=False)
    print(tfidf_means.head(10))

if __name__ == "__main__":
    process_dataset() 