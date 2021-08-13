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

    std::string passthrough_vertex = R"foo(
in vec3 vertex;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
}
)foo";
    std::string passthrough_fragment = R"foo(
in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    colour = vec4(1);
}
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
        }}, GL_DYNAMIC_DRAW),
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
