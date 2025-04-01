# Detector de Spam com Machine Learning

Este projeto implementa um sistema de detecção de spam usando técnicas de Machine Learning, especificamente o algoritmo Naïve Bayes. O sistema é capaz de classificar e-mails como spam ou não spam (ham) com base em várias características do texto.

## 🚀 Funcionalidades

- **Classificação de E-mails**: Identifica automaticamente se um e-mail é spam ou não
- **Probabilidade de Spam**: Fornece a probabilidade de um e-mail ser spam
- **Interface Web**: Interface amigável para análise de e-mails
- **API REST**: Endpoint para integração com outros sistemas
- **Features Avançadas**: Utiliza múltiplas características para detecção de spam

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **FastAPI**: Framework para a API REST
- **Streamlit**: Interface web interativa
- **scikit-learn**: Biblioteca para Machine Learning
- **NLTK**: Processamento de linguagem natural
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **matplotlib**: Visualização de dados

## 📊 Features do Modelo

O modelo utiliza várias características para detectar spam:

### Features de Texto
- Vetorização TF-IDF com 15.000 features
- N-gramas (1-3 palavras)
- Palavras e frases comuns em spam
- Padrões de texto suspeitos

### Features Numéricas
- Razão de caracteres maiúsculos
- Contagem de caracteres especiais
- Contagem de números
- Contagem de URLs
- Sequências de caracteres especiais
- Sequências de números
- Palavras repetidas
- Contagem de palavras de spam
- Razão de palavras de spam
- Contagem de frases suspeitas

## 🏗️ Estrutura do Projeto

```
spam-detector/
├── data/
│   ├── processed_data.csv
│   └── vectorized_data.csv
├── models/
│   ├── spam_detector.joblib
│   ├── tfidf_vectorizer.joblib
│   └── feature_scaler.joblib
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── streamlit_app.py
│   └── models/
│       └── train_model.py
├── requirements.txt
└── README.md
```

## 🚀 Como Usar

### Pré-requisitos

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Treine o modelo:
```bash
python src/models/train_model.py
```

3. Inicie a API FastAPI:
```bash
python src/api/app.py
```

4. Inicie a interface Streamlit:
```bash
streamlit run src/api/streamlit_app.py
```

### Acessando o Sistema

- **Interface Web**: http://localhost:8501
- **API FastAPI**: http://localhost:8000
- **Documentação da API**: http://localhost:8000/docs

## 🔍 Como Funciona

1. **Pré-processamento**:
   - Conversão para minúsculas
   - Remoção de URLs
   - Remoção de números
   - Limpeza de caracteres especiais
   - Adição de features de spam

2. **Vetorização**:
   - Transformação do texto em vetores TF-IDF
   - Extração de features numéricas
   - Normalização das features

3. **Classificação**:
   - Modelo Naïve Bayes Multinomial
   - Otimização de hiperparâmetros via Grid Search
   - Validação cruzada para avaliação

## 📈 Métricas de Performance

O modelo é avaliado usando:
- F1-score
- Matriz de confusão
- Relatório de classificação
- Análise de features mais importantes

## 🔒 Segurança

- Validação de entrada
- Tratamento de erros
- Sanitização de dados
- Proteção contra injeção de código

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Autores

- Seu Nome - [seu-email@exemplo.com]

## 🙏 Agradecimentos

- Dataset de e-mails spam/ham
- Comunidade de Machine Learning
- Contribuidores do projeto 