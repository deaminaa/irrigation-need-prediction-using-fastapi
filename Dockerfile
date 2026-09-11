FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output appears immediately in Docker logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# LightGBM's wheel links against OpenMP, which python:*-slim does not ship.
# gcc used to supply libgomp1 as a transitive dependency; depending on it
# directly keeps the runtime working without shipping a compiler.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY irrigation.py .
COPY irrigation_model.pkl .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Only the Streamlit UI is published. FastAPI is bound to 127.0.0.1 and reached
# over loopback from inside the container, so it is never exposed to the network.
EXPOSE 8501

# Both services have to answer. A check that only asks Streamlit reports the
# container healthy while every prediction is failing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; \
u.urlopen('http://127.0.0.1:8000/health', timeout=4).read(); \
u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()"

# Start both services
CMD ["./start.sh"]
