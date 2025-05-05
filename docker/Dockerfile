FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /bin/

LABEL org.opencontainers.image.authors="Bram Vanroy"
LABEL org.opencontainers.image.title="MAchine Translation Evaluation Online - Demo"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Application User ---
RUN useradd -m -u 1000 mateo_user
USER mateo_user
ENV HOME="/home/mateo_user"
WORKDIR $HOME

# --- Application Code & Dependencies ---
ARG REPO_BRANCH=v1.8.0
RUN git clone --depth 1 --branch ${REPO_BRANCH} https://github.com/BramVanroy/mateo-demo.git $HOME/mateo-demo

WORKDIR $HOME/mateo-demo

ARG USE_CUDA=false
ARG TORCH_CUDA_VERSION="126"
ARG TORCH_VERSION="2.7.0"
# Conditionally install PyTorch (CPU or CUDA 12.1)
RUN uv venv .venv --python 3.11 \
    && . .venv/bin/activate \
    && uv pip install --no-cache-dir --upgrade pip wheel setuptools \
    && if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing PyTorch with CUDA 12.1 support"; \
        uv pip install --no-cache-dir torch=="$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}"; \
    else \
        echo "Installing PyTorch with CPU support"; \
        uv pip install --no-cache-dir torch=="${TORCH_VERSION}+cpu" --index-url https://download.pytorch.org/whl/cpu; \
    fi \
    && uv pip install --no-cache-dir --upgrade $HOME/mateo-demo[patch] \
    && uv cache clean

# --- Runtime Configuration ---
# Set runtime environment variables
ENV PORT=7860 \
    SERVER="localhost" \
    BASE="" \
    DEMO_MODE=false \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PRELOAD_METRICS=true \
    PATH="$HOME/.local/bin:$HOME/mateo-demo/.venv/bin:${PATH}" \
    ENABLE_XSRF_PROTECTION=true

# --- Set default cache dir for the user ---
RUN mkdir -p "$HOME/.cache" \
    && chown -R mateo_user:mateo_user "$HOME/.cache"

# --- Entrypoint & CMD ---
# Expose the port the app runs on
EXPOSE $PORT

# Healthcheck (adjust start-period if runtime loading takes longer)
# Note: SERVER env var needs to be passed at runtime if not localhost
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl --fail http://${SERVER:-localhost}:${PORT}${BASE}/_stcore/health || exit 1

# Set the working directory for the application
WORKDIR $HOME/mateo-demo/src/mateo_st

RUN chmod +x $HOME/mateo-demo/docker/entrypoint.sh
ENTRYPOINT $HOME/mateo-demo/docker/entrypoint.sh

# CMD is now empty as the entrypoint handles the final command execution
CMD []