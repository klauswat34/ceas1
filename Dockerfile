FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
