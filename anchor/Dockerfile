# Anchor AI Integrity Framework — Docker image
#
# Single image, two personalities:
#   anchor-eval  →  python /app/scripts/run_eval.py  (network_mode: none)
#   anchor-ui    →  streamlit run /app/anchor/ui/app.py
#
# Build context is the repo root (d:\ANCHOR\anchor), so paths below are
# relative to that directory.

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tcpdump  : needed inside the container by prove_isolation.sh
# libgomp1 : required by faiss-cpu
# libglib2-0: required by opencv headless
# build-essential + pkg-config: compile pycocotools
RUN apt-get update && apt-get install -y --no-install-recommends \
        tcpdump \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt ./

# Use headless opencv (no GUI) and install everything else from requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir opencv-python-headless==4.10.0.84 \
    && grep -v "^opencv-python==" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# ── Source code ───────────────────────────────────────────────────────────────
COPY . /app/anchor
COPY scripts/ /app/scripts/

# Make the eval entrypoint and isolation script executable
RUN chmod +x /app/scripts/run_eval.py /app/scripts/prove_isolation.sh 2>/dev/null || true

# PYTHONPATH so "import anchor.*" resolves from /app
ENV PYTHONPATH=/app

# ── Default command (overridden per-service in docker-compose.yml) ────────────
CMD ["python", "/app/scripts/run_eval.py", "--help"]
