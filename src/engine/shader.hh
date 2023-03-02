#pragma once

#include <optional>
#include <string>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "buffers.hh"
#include "util/misc.hh"

std::string shared_header_shader_text = R"foo(
layout(std140) uniform inputs {
    mat4 view, projection;
    vec2 mouse;
    vec2 resolution;
    uint framerate;
    float time;
};
)foo";
std::string shared_passthrough_vertex = R"foo(
in vec3 vertex;

out gl_PerVertex {
    vec4 gl_Position;
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
}
)foo";
std::string shared_passthrough_fragment = R"foo(
in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    colour = vec4(1);
}
)foo";

struct shared_uniforms {
    struct inputs {
        glm::mat4 view, projection;
        float mouse_x, mouse_y;
        float resolution_x, resolution_y;
        GLuint framerate;
        float time;
    };

    glfw_t& glfw;
    uniform_buffer<inputs> ubo;
    inputs& inputs;
    size_t frame = 0;
    shared_uniforms(glfw_t &glfw_):
        glfw(glfw_),
        ubo({{
            glm::identity<glm::mat4>(),
            glm::identity<glm::mat4>(),
            0, 0,
            0, 0,
            0,
            0,
        }}, GL_DYNAMIC_DRAW),
        inputs(ubo.data.front())
    {
        int interval = 1;
        glfwSwapInterval(interval);
        const GLFWvidmode *vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        inputs.framerate = vidmode->refreshRate / interval;
        glfwSetTime(0);

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        inputs.resolution_x = w;
        inputs.resolution_y = h;
    }
    void bind(GLuint program) {
        ubo.bind(program, "inputs");
    }
    void draw(bool perspective = true) {
        double mx, my;
        glfwGetCursorPos(glfw.window, &mx, &my);
        inputs.mouse_x = mx;
        inputs.mouse_y = my;

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        inputs.resolution_x = w;
        inputs.resolution_y = h;
        if (perspective) {
            inputs.projection = glm::perspective(glm::radians(75.0f), static_cast<float>(w) / h, 0.1f, 200.f);
        } else {
            inputs.projection = glm::ortho(0.0f, static_cast<float>(w), 0.0f, static_cast<float>(h), 0.0f, 200.0f);
        }

        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(inputs), &inputs);
    }
    void tick() {
        frame++;
        inputs.time = static_cast<float>(frame) / inputs.framerate;
    }
    ~shared_uniforms() {
        std::cout << frame / glfwGetTime() << " average fps" << std::endl;
    }
};

glfw_t glfw;
shared_uniforms shared{glfw};

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
    shader(
        std::string vertex_source = shared_passthrough_vertex,
        std::string fragment_source = shared_passthrough_fragment
    ) {
        glCreateProgramPipelines(1, &pipeline);
        program_vertex = create_program(GL_VERTEX_SHADER, shared_header_shader_text + vertex_source);
        program_fragment = create_program(GL_FRAGMENT_SHADER, shared_header_shader_text + fragment_source);
        glUseProgramStages(pipeline, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline, GL_FRAGMENT_SHADER_BIT, program_fragment);
        shared.bind(program_vertex);
        shared.bind(program_fragment);
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
        glCreateProgramPipelines(1, &pipeline);
        program = create_program(GL_COMPUTE_SHADER, shared_header_shader_text + compute_source);
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
