#include <vector>
#include <array>
#include <utility>
#include "buffers.hh"
#include "shader.hh"

struct lines_renderer {
    GLuint vertex_attrib_index = 0;

    using line_type = std::pair<
        std::array<float, 3>,
        std::array<float, 3>
    >;

    vertex_array_object vao;
    vertex_buffer<line_type> vbo;

    shader shader;

    lines_renderer(std::vector<line_type> lines_, std::string shared_uniforms):
        vbo(lines_, vertex_attrib_index),
        shader(shared_uniforms + R"foo(
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
)foo", vertex_attrib_index)
    {}
    void draw() {
        vao.draw();
        vbo.draw();
        shader.draw();
        glLineWidth(1);
        glDrawArrays(GL_LINES, 0, vbo.data.size() * 2);
    }
};
