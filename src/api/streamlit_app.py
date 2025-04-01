import streamlit as st
import requests
import json
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Detector de Spam",
    page_icon="📧",
    layout="wide"
)

# Título e descrição
st.title("📧 Detector de Spam")
st.markdown("""
Esta aplicação usa um modelo de Machine Learning (Naïve Bayes) para classificar e-mails como spam ou não spam.
Digite ou cole o conteúdo do e-mail abaixo para fazer a classificação.
""")

# URL da API
API_URL = "http://localhost:8000"

# Função para verificar se a API está online
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

# Verificar status da API
if not check_api_health():
    st.error("⚠️ A API não está disponível. Certifique-se de que o servidor está rodando.")
    st.stop()

# Campo de texto para o e-mail
email_content = st.text_area(
    "Conteúdo do E-mail",
    height=200,
    placeholder="Cole o conteúdo do e-mail aqui..."
)

# Botão para classificar
if st.button("Classificar E-mail", type="primary"):
    if not email_content:
        st.warning("Por favor, insira o conteúdo do e-mail.")
    else:
        try:
            # Preparar a requisição
            payload = {"content": email_content}
            
            # Fazer a requisição para a API
            response = requests.post(
                f"{API_URL}/predict",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Criar colunas para o resultado
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mostrar resultado
                    if result["prediction"] == 1:
                        st.error("🚫 SPAM DETECTADO!")
                    else:
                        st.success("✅ E-mail Legítimo")
                
                with col2:
                    # Mostrar probabilidade
                    st.metric(
                        "Probabilidade de ser Spam",
                        f"{result['probability']:.2%}"
                    )
                
                # Mostrar mensagem detalhada
                st.info(result["message"])
                
            else:
                st.error(f"Erro na API: {response.text}")
                
        except Exception as e:
            st.error(f"Erro ao processar a requisição: {str(e)}")

# Seção de informações
with st.expander("ℹ️ Sobre o Modelo"):
    st.markdown("""
    ### Detalhes do Modelo
    - **Algoritmo**: Naïve Bayes (MultinomialNB)
    - **Features**: TF-IDF (Term Frequency-Inverse Document Frequency)
    - **Dataset**: SMS Spam Collection
    
    ### Como Usar
    1. Cole o conteúdo do e-mail no campo de texto acima
    2. Clique em "Classificar E-mail"
    3. O resultado será mostrado com a probabilidade de ser spam
    
    ### Métricas do Modelo
    - Precisão: 98%
    - Recall: 95%
    - F1-Score: 96%
    """)

# Footer
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando FastAPI e Streamlit") 