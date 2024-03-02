FROM ubuntu:latest
LABEL authors="Bram Vanroy"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get -y install build-essential curl git software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update \
    && apt-get -y install python3.10 python3.10-dev python3-pip python3.10-distutils \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME /home/user
ENV PATH $HOME/.local/bin:$PATH

WORKDIR $HOME
RUN git clone https://github.com/BramVanroy/mateo-demo.git
WORKDIR $HOME/mateo-demo

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir --upgrade .

ENV PORT=7860
ENV SERVER="localhost"
ENV BASE=""
ENV NO_CUDA=""
ENV DEMO_MODE=""
ENV HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE $PORT
HEALTHCHECK CMD curl --fail http://$SERVER:$PORT$BASE/_stcore/health

WORKDIR $HOME/mateo-demo/src/mateo_st

CMD if [ -z "$BASE" ]; then \
        cmd="streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER"; \
    else \
        cmd="streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER --server.baseUrlPath $BASE"; \
    fi; \
    if [ "$DEMO_MODE" = "true" ]; then \
        cmd="$cmd --server.maxUploadSize 1"; \
    fi; \
    if [ "$NO_CUDA" = "true" ] || [ "$DEMO_MODE" = "true" ]; then \
        opts="--"; \
        [ "$NO_CUDA" = "true" ] && opts="$opts --no_cuda"; \
        [ "$DEMO_MODE" = "true" ] && opts="$opts --demo_mode"; \
        cmd="$cmd $opts"; \
    fi; \
    exec $cmd
