#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential \
  libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev \
  libflann-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libsqlite3-dev \
  libglew-dev \
  qtbase5-dev libqt5opengl5-dev \
  libcgal-dev \
  libceres-dev \
  libabsl-dev \
  ffmpeg