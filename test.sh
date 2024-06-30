#!/usr/bin/env bash

set -o errexit

export CC=clang
export CXX=clang++
if [ ! -d builddir ]; then
    cmake -G Ninja -S . -B builddir \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
fi
cmake --build builddir
cmake --install builddir --prefix packaged

gdb -ex run --args \
    ./packaged/bin/render
