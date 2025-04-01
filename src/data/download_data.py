import os
import pandas as pd
import requests
from pathlib import Path

def download_sms_spam_dataset():
    """
    Download do SMS Spam Collection Dataset
    """
    # URL do dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Criar diretório de dados se não existir
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download do arquivo
    print("Baixando o dataset...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Salvar o arquivo zip
        zip_path = data_dir / "smsspamcollection.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extrair o arquivo
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remover o arquivo zip
        zip_path.unlink()
        
        print("Dataset baixado e extraído com sucesso!")
        
        # Ler e preparar os dados
        prepare_data(data_dir)
    else:
        print(f"Erro ao baixar o dataset. Status code: {response.status_code}")

def prepare_data(data_dir):
    """
    Preparar os dados para treinamento
    """
    # Ler o arquivo de dados
    data_path = data_dir / "SMSSpamCollection"
    if not data_path.exists():
        print("Arquivo de dados não encontrado!")
        return
    
    # Ler os dados
    df = pd.read_csv(data_path, sep='\t', names=['label', 'message'])
    
    # Converter labels para binário (spam = 1, ham = 0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Salvar dados processados
    processed_data_path = data_dir / "processed_data.csv"
    df.to_csv(processed_data_path, index=False)
    
    print(f"Dados processados salvos em: {processed_data_path}")
    print(f"Total de amostras: {len(df)}")
    print(f"Distribuição de classes:\n{df['label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    download_sms_spam_dataset() 