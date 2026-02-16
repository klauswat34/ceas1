FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
