#include <string>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "buffers.hh"

struct shared_uniforms {
    struct inputs {
        glm::mat4 view, projection;
        float mouse_x, mouse_y;
        float resolution_x, resolution_y;
        GLuint frame;
    };
    std::string header_shader_text = R"foo(
layout(std140) uniform inputs {
    mat4 view, projection;
    vec2 mouse;
    vec2 resolution;
    uint frame;
};
)foo";
    glfw_t& glfw;
    uniform_buffer<inputs> ubo;
    inputs& inputs;
    shared_uniforms(glfw_t &glfw_):
        glfw(glfw_),
        ubo({{
            glm::identity<glm::mat4>(),
            glm::identity<glm::mat4>(),
            0, 0,
            0, 0,
            0
        }}, GL_STATIC_DRAW),
        inputs(ubo.data.front())
    {}
    void bind(GLuint program) {
        ubo.bind(program, "inputs");
    }
    void draw() {
        double mx, my;
        glfwGetCursorPos(glfw.window, &mx, &my);
        inputs.mouse_x = mx;
        inputs.mouse_y = my;

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        inputs.resolution_x = w;
        inputs.resolution_y = h;

        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(inputs), &inputs);

        inputs.frame++;
    }
};
