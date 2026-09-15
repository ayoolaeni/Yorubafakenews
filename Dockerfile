FROM python:3.11-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements-app.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
