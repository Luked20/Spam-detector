import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from pathlib import Path

# Download recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Pré-processa o texto:
    1. Converte para minúsculas
    2. Remove caracteres especiais e números
    3. Remove stopwords
    4. Aplica lemmatization
    """
    # Converter para minúsculas
    text = text.lower()
    
    # Remover caracteres especiais e números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenização
    tokens = word_tokenize(text)
    
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Juntar tokens de volta em texto
    return ' '.join(tokens)

def preprocess_dataset():
    """
    Pré-processa todo o dataset
    """
    # Definir diretório de dados
    data_dir = Path("data")
    
    # Ler dados processados
    input_path = data_dir / "processed_data.csv"
    if not input_path.exists():
        print(f"Arquivo de dados não encontrado em: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    print("Iniciando pré-processamento...")
    
    # Aplicar pré-processamento em todas as mensagens
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Salvar dados pré-processados
    output_path = data_dir / "preprocessed_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dados pré-processados salvos em: {output_path}")
    print(f"Exemplo de mensagem original:\n{df['message'].iloc[0]}")
    print(f"\nExemplo de mensagem pré-processada:\n{df['processed_message'].iloc[0]}")

if __name__ == "__main__":
    preprocess_dataset() 