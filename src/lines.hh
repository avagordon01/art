#include <vector>
#include <array>
#include <utility>
#include "buffers.hh"
#include "shader.hh"

struct lines_renderer {
    vertex_array_object vao;
    vertex_buffer<std::array<float, 3>> vbo;

    shader shader;

    lines_renderer(std::vector<std::array<float, 3>> lines_, shared_uniforms& shared_uniforms):
        vbo(lines_),
        shader(shared_uniforms.header_shader_text + R"foo(
in vec3 vertex;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
}
)foo",
        R"foo(
in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    colour = vec4(1);
}
)foo")
    {
        vbo.bind(shader.program_vertex, "vertex");
        shared_uniforms.bind(shader.program_vertex);
    }
    void draw() {
        vao.draw();
        vbo.draw();
        shader.draw();
        glLineWidth(1);
        glDrawArrays(GL_LINES, 0, vbo.data.size());
    }
};
