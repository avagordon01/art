#!/usr/bin/env bash

set -o errexit

#aria2c -j8 -Z \
    #ftp://ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/

wget -r \
    ftp://ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/
