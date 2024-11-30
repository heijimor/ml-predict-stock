# Use uma imagem base do Python
FROM python:3.11-slim AS base

# Defina o diretório de trabalho dentro do container
WORKDIR .

# Copie o arquivo de requisitos e instale as dependências
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação e o modelo salvo para o container
COPY . .

# Expor as portas da API e do servidor de métricas
EXPOSE 8000

# Comando para iniciar a API
CMD ["python", "app.py"]