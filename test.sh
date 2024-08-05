#!/usr/bin/env bash

set -o errexit

export PATH=$PATH:/opt/intel/oneapi/compiler/2024.1/bin/
export CC=icx
export CXX=icpx
if [ ! -d builddir ]; then
    cmake -G Ninja -S . -B builddir \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
fi
cmake --build builddir
cmake --install builddir --prefix packaged

./packaged/bin/test-sycl
