FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get -y install git software-properties-common build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y python3.9 python3.9-dev python3-pip python3.9-distutils \
    && ln -s /usr/bin/python3.9 /usr/bin/python

LABEL authors="Bram Vanroy"

RUN git clone https://github.com/BramVanroy/mateo-demo.git
WORKDIR /mateo-demo

RUN python -m pip install -r requirements.txt
RUN python -m pip install --no-cache-dir --upgrade .

ENV PORT=5004
ENV SERVER="localhost"
ENV BASE=""
ENV NO_CUDA=""
EXPOSE $PORT
HEALTHCHECK CMD curl --fail http://$SERVER:$PORT/_stcore/health

WORKDIR /mateo-demo/src/mateo_st

CMD if [ -z "$BASE" ]; then \
        cmd="streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER"; \
    else \
        cmd="streamlit run 01_🎈_MATEO.py --server.port $PORT --browser.serverAddress $SERVER --server.baseUrlPath $BASE"; \
    fi; \
    if [ "$NO_CUDA" = "true" ]; then \
        cmd="$cmd -- --no_cuda"; \
    fi; \
    exec $cmd
