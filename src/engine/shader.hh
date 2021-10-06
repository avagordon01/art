#pragma once

#include <optional>
#include <string>
#include "util/misc.hh"

GLuint create_program(GLenum type, std::string shader_text) {
    shader_text = std::string("#version 450\n") + shader_text;
    const char* s[] = {
        shader_text.c_str(),
    };

    GLuint program = glCreateShaderProgramv(type, 1, s);

    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status == GL_TRUE) {
        glValidateProgram(program);
        GLint validate_status;
        glGetProgramiv(program, GL_VALIDATE_STATUS, &validate_status);
        if (validate_status == GL_TRUE) {
            return program;
        }
    }

    GLint info_log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
    std::string info_log(info_log_length, '\0');
    glGetProgramInfoLog(program, info_log.size(), NULL, info_log.data());
    std::cerr << info_log << std::endl;
    throw std::runtime_error("shader stuff");
}

struct shader {
    GLuint pipeline, program_vertex, program_fragment;
    shader(std::string vertex_source, std::string fragment_source) {
        glGenProgramPipelines(1, &pipeline);
        program_vertex = create_program(GL_VERTEX_SHADER, vertex_source);
        program_fragment = create_program(GL_FRAGMENT_SHADER, fragment_source);
        glUseProgramStages(pipeline, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline, GL_FRAGMENT_SHADER_BIT, program_fragment);
    }
    void draw() {
        glBindProgramPipeline(pipeline);
    }
};

struct compute_shader {
    GLuint pipeline, program;
    std::array<size_t, 3> dimensions;
    std::array<GLint, 3> workgroup_dimensions;
    compute_shader(std::string compute_source, std::array<size_t, 3> dimensions_):
        dimensions(dimensions_)
    {
        glGenProgramPipelines(1, &pipeline);
        program = create_program(GL_COMPUTE_SHADER, compute_source);
        glUseProgramStages(pipeline, GL_COMPUTE_SHADER_BIT, program);
        std::array<GLint, 3> workgroup_dimensions;
        glGetProgramiv(program, GL_COMPUTE_WORK_GROUP_SIZE, workgroup_dimensions.data());
        dimensions[0] = div_ceil(dimensions[0], workgroup_dimensions[0]);
        dimensions[1] = div_ceil(dimensions[1], workgroup_dimensions[1]);
        dimensions[2] = div_ceil(dimensions[2], workgroup_dimensions[2]);
    }

    void draw() {
        glBindProgramPipeline(pipeline);
        glDispatchCompute(dimensions[0], dimensions[1], dimensions[2]);
    }

    void draw(std::array<size_t, 3> dimensions_) {
        dimensions = dimensions_;
        dimensions[0] = div_ceil(dimensions[0], workgroup_dimensions[0]);
        dimensions[1] = div_ceil(dimensions[1], workgroup_dimensions[1]);
        dimensions[2] = div_ceil(dimensions[2], workgroup_dimensions[2]);
        glBindProgramPipeline(pipeline);
        glDispatchCompute(dimensions[0], dimensions[1], dimensions[2]);
    }
};
