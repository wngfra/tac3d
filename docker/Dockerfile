# Copyright (C) 2022 wngfra/captjulian
# 
# tac3d is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tac3d is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with tac3d. If not, see <http://www.gnu.org/licenses/>.

ARG FROM_IMAGE=ubuntu:20.04
FROM ${FROM_IMAGE}

USER root

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ g++-multilib libgflags-dev libgoogle-glog-dev wget \
    python3-dev python3-pip python3-venv python3-wheel \
    libusb-1.0-0 libgrpc++1 libi2c0 usbutils \
    tmux vim && \
    apt-get autoclean && apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

#RUN wget http://ftp.debian.org/debian/pool/main/p/protobuf/libprotobuf17_3.6.1-4~bpo9+1_amd64.deb && \
#    dpkg -i libprotobuf17_3.6.1-4~bpo9+1_amd64.deb

RUN python3 -m venv /loihi
WORKDIR /loihi
ENV PATH=/loihi/bin:$PATH
RUN pip3 install -U --quiet pip wheel setuptools

COPY *.tar.gz /loihi/
RUN mkdir nxsdk-apps && \
    tar xzf nxsdk-apps-1.0.0.tar.gz -C nxsdk-apps --strip-components 1

RUN pip3 install --quiet \
    numpy==1.21 \
    cython \
    grpcio-tools==1.16 \
    protobuf==3.20 \
    jupyterlab \
    ipympl \
    nengo-extras \
    nengo-loihi \
    ./nxsdk-1.0.0.tar.gz

ENV KAPOHOBAY=1
