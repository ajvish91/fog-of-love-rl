FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for native wheels where needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY AgileRL /app/AgileRL
WORKDIR /app/AgileRL
RUN pip install --no-cache-dir -e .

WORKDIR /app
COPY . .

# FoL package lives under scripts/fol/
ENV PYTHONPATH=/app/scripts

# Weights & Biases: pass at runtime, e.g. docker run -e WANDB_API_KEY=...
ENV WANDB_API_KEY=""

# Quick smoke test by default (debug on). Override CMD for full training.
ENTRYPOINT ["python", "-m", "fol.training.run_fol"]
CMD ["--config", "configs/smoke_arg_localized.yaml"]
