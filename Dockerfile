FROM python:3.9-slim

# ??????
WORKDIR /app

# ??????
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

# ??? requirements.txt????? COPY ????? pip ???
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ???????
COPY main.py config.py run.py app.yaml __init__.py ./ 
COPY resources/   resources/
COPY services/    services/
COPY templates/   templates/
COPY scripts/     scripts/

# ?? Cloud Run ?????
ENV PORT=8080

# ?? Gunicorn ????
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "600", "main:app"]

