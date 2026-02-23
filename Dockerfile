FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model_artifacts.joblib .
COPY SriLanka_Weather_Dataset.csv .

ENV PORT=8501
EXPOSE ${PORT}

CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
