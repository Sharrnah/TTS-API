FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    build-essential \
    git \
    git-lfs \
    libglib2.0-0 \
    wget \
    llvm \
    gcc g++ make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# install debian python package
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    python3.11 \
    python3-dev \
    python3.11-dev \
    python3-pip \
    python3-venv python3-wheel \
    libffi-dev libnacl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install coqui dependencies
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    espeak-ng libsndfile1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
#    rm ~/miniconda.sh
#ENV PATH=$CONDA_DIR/bin:$PATH
#
## install python 3.10 and pip
#RUN conda install python=3.10 && conda install -c anaconda pip
#
#
#RUN mkdir -p /root/.huggingface/
#
#RUN conda install -c "nvidia/label/cuda-12.2.2" cuda-toolkit
#RUN conda install pytorch torchvision torchaudio -c pytorch -c conda-forge

WORKDIR /app
EXPOSE 8000

RUN pip install --no-cache-dir --upgrade pip
RUN pip --version
RUN pip install --no-cache-dir llvmlite --ignore-installed
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir  --ignore-installed -r requirements.txt -U
RUN test -d /root/.cache || mkdir -p /root/.cache
COPY cache/. /root/.cache/

COPY _preinstall.py /app/_preinstall.py
RUN python3 _preinstall.py

COPY main.py /app/main.py
COPY audio_tools.py /app/audio_tools.py
COPY bark_tts.py /app/bark_tts.py
COPY coqui_tts.py /app/coqui_tts.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
