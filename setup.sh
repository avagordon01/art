#!/usr/bin/env bash

set -o errexit

sudo pacman -S glfw freetype2 glm mesa-demos

glxinfo | grep OpenGL
if [ $? -ne 0 ]; then
    echo "error no opengl driver found"
fi
