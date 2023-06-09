FROM ubuntu:latest
LABEL authors="Bram Vanroy"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get -y install build-essential curl git software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update \
    && apt-get -y install python3.9 python3.9-dev python3-pip python3.9-distutils \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME /home/user
ENV PATH $HOME/.local/bin:$PATH

WORKDIR $HOME
RUN git clone https://github.com/BramVanroy/mateo-demo.git
WORKDIR $HOME/mateo-demo

RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir --upgrade .

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

WORKDIR $HOME/mateo-demo/src/mateo_st

CMD streamlit run 01_🎈_MATEO.py --server.port 7860 --server.enableXsrfProtection false
