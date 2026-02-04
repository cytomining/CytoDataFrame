# modified from: https://github.com/CellProfiler/CellProfiler/blob/main/distribution/docker/Dockerfile
# must be amd64, wxPython wheel for arm64 is not available
FROM ubuntu:20.04

# bypass apt installation frontend interaction
ARG DEBIAN_FRONTEND=noninteractive

ARG cp_version=4.2.8

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    make \
    gcc \
    build-essential \
    libgtk-3-dev \
    wget \
    git \
    sqlite3 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    openjdk-11-jdk-headless \
    default-libmysqlclient-dev \
    libnotify-dev \
    libsdl2-dev \
    libwebkit2gtk-4.0-dev && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# hadolint ignore=DL3013
RUN python3.9 -m venv $VIRTUAL_ENV && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    wheel \
    cython \
    numpy \
    https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.2.1-cp39-cp39-linux_x86_64.whl \
    cellprofiler==$cp_version

WORKDIR /usr/local/src
ENTRYPOINT ["cellprofiler"]

CMD ["--run", "--run-headless", "--help"]
