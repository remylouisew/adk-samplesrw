FROM python:3.13-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#RUN adduser --disabled-password --gecos "" remyw && \
    #chown -R remyw:remyw /app

COPY . .

#RUN chmod -R +x /app

#USER myuser

ENV PATH="/home/remyw/.local/bin:$PATH"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]