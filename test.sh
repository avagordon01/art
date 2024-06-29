#!/usr/bin/env bash

set -o errexit

export CC=clang
export CXX=clang++
if [ ! -d builddir ]; then
    cmake -G Ninja -S . -B builddir
fi
cmake --build builddir
cmake --install builddir --prefix packaged

./packaged/bin/render
