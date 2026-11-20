# This digest is the verified linux/arm64 manifest from docs/ENVIRONMENT.md.
FROM --platform=linux/arm64 python@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.lock.txt pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes -r requirements.lock.txt

COPY src/ ./src/
COPY flows/ ./flows/
COPY scripts/ ./scripts/
COPY config/ ./config/

# DVC remains a host-side data-versioning command. The image only contains the
# locked project environment used for training, MLflow and batch scoring.
CMD ["python", "-m", "dropout.score", "--help"]
