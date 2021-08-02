#!/usr/bin/env bash

set -o errexit

aria2c -j8 -Z \
    https://netix.dl.sourceforge.net/project/artwizaleczapka/iso-8859-1/1.3/artwiz-aleczapka-en-1.3.tar.bz2 \
    http://www.fial.com/~scott/tamsyn-font/download/tamsyn-font-1.11.tar.gz \
    https://font.gohu.org/gohufont-2.1.tar.gz \
    https://gnu.mirror.constant.com/unifont/unifont-13.0.06/unifont-13.0.06.pcf.gz
