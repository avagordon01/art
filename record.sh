#!/usr/bin/env bash

set -o errexit

prime-run apitrace trace -o out.trace ./packaged/bin/render
apitrace dump-images out.trace
ffmpeg -framerate 60 -pattern_type glob -i '*.png' -c:v libx264 out.mp4
