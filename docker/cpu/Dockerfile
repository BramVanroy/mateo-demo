FROM python:3.11-slim-bookworm

# Metadata as per https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.authors="Bram Vanroy"
LABEL org.opencontainers.image.title="MAchine Translation Evaluation Online - Demo"

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies in a single RUN command to reduce image layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 mateo_user
USER mateo_user
ENV HOME="/home/mateo_user"

# Environment variables
ENV PORT=7860 \
    SERVER="localhost" \
    BASE="" \
    DEMO_MODE="" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH="${HOME}/.local/bin:${PATH}" \
    USE_CUDA=false

WORKDIR ${HOME}/mateo

# Clone the repository
RUN git clone --depth 1 --branch v1.4.1 https://github.com/BramVanroy/mateo-demo.git
WORKDIR mateo-demo

# Install Python dependencies with conditional torch installation
RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools \
    && python -m pip install --no-cache-dir torch==2.6.0+cpu -f https://download.pytorch.org/whl/torch \
    && python -m pip install --no-cache-dir --upgrade .

# Pre-download default models
RUN huggingface-cli download bert-base-multilingual-cased model.safetensors tokenizer.json vocab.txt; \
    huggingface-cli download facebook/nllb-200-distilled-600M pytorch_model.bin sentencepiece.bpe.model tokenizer.json; \
    python -c "import comet; from comet import download_model; download_model('Unbabel/wmt22-comet-da')"; \
    python -c "import evaluate; evaluate.load('bleurt', 'BLEURT-20')"

# Expose the port the app runs on
EXPOSE $PORT

# Healthcheck to ensure the service is running
HEALTHCHECK CMD curl --fail http://$SERVER:$PORT$BASE/_stcore/health || exit 1

# Set the working directory to the Streamlit app
WORKDIR src/mateo_st

# Simplify the CMD script with conditional --use_cuda flag
CMD streamlit run 01_🎈_MATEO.py \
    --server.port $PORT \
    --server.address $(if [ "$SERVER" = "localhost" ]; then echo "0.0.0.0"; else echo $SERVER; fi) \
    $(if [ -n "$BASE" ]; then echo "--server.baseUrlPath $BASE"; fi) \
    $(if [ "$DEMO_MODE" = "true" ]; then echo "--server.maxUploadSize 1"; fi) \
    -- \
    $(if [ "$DEMO_MODE" = "true" ]; then echo "--demo_mode"; fi)
