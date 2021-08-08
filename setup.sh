#!/usr/bin/env bash

set -o errexit

sudo pacman -S glfw freetype2 glm assimp mesa-demos nlohmann-json

glxinfo | grep OpenGL
if [ $? -ne 0 ]; then
    echo "error no opengl driver found"
fi

./fonts/download.sh
./data/download.sh
