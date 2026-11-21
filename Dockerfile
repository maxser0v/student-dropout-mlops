# The available Python manifest was verified for linux/arm64.
FROM --platform=linux/arm64 python@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

WORKDIR /app

COPY requirements.lock.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes -r requirements.lock.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib

COPY pyproject.toml ./
COPY src/ ./src/
COPY flows/ ./flows/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Build/runtime identity arguments are deliberately declared after dependency
# installation so metadata changes do not invalidate the locked wheel layer.
ARG RUNTIME_UID=10001
ARG RUNTIME_GID=10001
RUN if ! getent group "${RUNTIME_GID}" >/dev/null; then groupadd --gid "${RUNTIME_GID}" appgroup; fi \
    && useradd --no-log-init --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser

ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ARG SOURCE_TREE_SHA256=unknown
ARG SOURCE_REVISION_SOURCE=unspecified
ARG SOURCE_TREE_HASH_ALGORITHM=sha256
LABEL org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.revision.dirty="${GIT_DIRTY}" \
      org.opencontainers.image.source-tree.sha256="${SOURCE_TREE_SHA256}" \
      org.opencontainers.image.revision.source="${SOURCE_REVISION_SOURCE}" \
      org.opencontainers.image.source-tree.hash-algorithm="${SOURCE_TREE_HASH_ALGORITHM}"

# A concrete non-root default. Compose may select the matching host UID/GID
# for writable bind mounts, but validates that the application identity stays nonzero.
USER ${RUNTIME_UID}:${RUNTIME_GID}

# DVC remains a host-side data-versioning command.
CMD ["python", "-m", "dropout.score", "--help"]
