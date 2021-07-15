#pragma once

template<typename ...S>
GLuint create_program(GLenum type, S... program_texts) {
    const char* s[] = {
        "#version 450\n",
        program_texts.c_str()...
    };
    GLuint program = glCreateShaderProgramv(type, 1 + sizeof...(program_texts), s);
    std::string error_log(4096, '\0');
    glGetProgramInfoLog(program, 4096, NULL, error_log.data());
    if (error_log[0] != 0) {
        throw std::runtime_error("shader error:\n" + error_log);
    }
    return program;
}
