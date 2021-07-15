#include "layer.hh"
#include <string>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

struct shared_uniforms : layer_t {
    struct inputs {
        glm::mat4 view, projection;
        float mouse_x, mouse_y;
        float resolution_x, resolution_y;
        GLuint frame;
    } inputs;
    GLuint binding_id = 0;
    std::string header_shader_text = R"foo(
layout(std140, binding=)foo" + std::to_string(binding_id) + R"foo() uniform inputs {
    mat4 view, projection;
    vec2 mouse;
    vec2 resolution;
    uint frame;
};
)foo";
    glfw_t& glfw;
    shared_uniforms(glfw_t &glfw_):
        glfw(glfw_)
    {
        inputs.frame = 0;

        GLuint ubo;
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(inputs), &inputs, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, binding_id, ubo);
    }
    void draw() override {
        double mx, my;
        glfwGetCursorPos(glfw.window, &mx, &my);
        inputs.mouse_x = mx;
        inputs.mouse_y = my;

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        inputs.resolution_x = w;
        inputs.resolution_y = h;

        inputs.projection = glm::perspective(glm::radians(75.0f), inputs.resolution_x / inputs.resolution_y, 0.1f, 200.f);
        inputs.view = glm::identity<glm::mat4>();
        float angle = 2 * M_PI * inputs.frame / 60 / 60;
        float distance = -100.0f;
        inputs.view = glm::lookAt(
            glm::vec3(distance * sin(angle), 0.0f, distance * cos(angle)),
            glm::vec3(),
            glm::vec3(0, 1, 0)
        );

        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(inputs), &inputs);

        inputs.frame++;
    }
};
