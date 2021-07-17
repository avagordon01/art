#pragma once
#include <string>

GLuint create_program(GLenum type, std::string shader_text) {
    shader_text = std::string("#version 450\n") + shader_text;
    const char* s[] = {
        shader_text.c_str(),
    };
    return glCreateShaderProgramv(type, 1, s);
}
