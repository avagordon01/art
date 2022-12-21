#!/usr/bin/env bash

set -o errexit

if [ ! -d out ]; then
    CXX=clang++ \
    meson setup out
fi
meson install -C out

prime-run ./packaged/bin/render
