FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get -y install build-essential curl git software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update \
    && apt-get -y install python3.9 python3.9-dev python3-pip python3.9-distutils

RUN ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

LABEL authors="Bram Vanroy"

RUN git clone https://github.com/BramVanroy/mateo-demo.git
WORKDIR /mateo-demo

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir --upgrade .

ENV PORT=7860
ENV SERVER="localhost"
ENV BASE=""
ENV NO_CUDA=""
ENV DEMO_MODE=""

EXPOSE $PORT
HEALTHCHECK CMD curl --fail http://$SERVER:$PORT$BASE/_stcore/health

WORKDIR /mateo-demo/src/mateo_st

CMD if [ -z "$BASE" ]; then \
        cmd="streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER"; \
    else \
        cmd="streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER --server.baseUrlPath $BASE"; \
    fi; \
    if [ "$NO_CUDA" = "true" ] || [ "$DEMO_MODE" = "true" ]; then \
        opts="--"; \
        [ "$NO_CUDA" = "true" ] && opts="$opts --no_cuda"; \
        [ "$DEMO_MODE" = "true" ] && opts="$opts --demo_mode"; \
        cmd="$cmd $opts"; \
    fi; \

    exec $cmd
