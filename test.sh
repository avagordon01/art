#!/usr/bin/env bash

set -o errexit

export CC=clang
export CXX=clang++
if [ ! -d builddir ]; then
    meson setup builddir
fi
meson install -C builddir

./packaged/bin/render

export DRI_PRIME=1
prime-run ./packaged/bin/render
