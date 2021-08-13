#include <vector>
#include <array>
#include <utility>
#include "buffers.hh"
#include "shader.hh"
#include "drawable.hh"

struct lines_renderer {
    drawable<> drawable;
    shader shader;

    lines_renderer(std::vector<std::array<float, 3>> lines_, shared_uniforms& shared_uniforms):
        drawable(GL_LINES),
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
        drawable.vbo.data = lines_;
        drawable.vbo.bind(shader.program_vertex, "vertex");
        shared_uniforms.bind(shader.program_vertex);
    }
    void draw() {
        shader.draw();
        drawable.draw();
    }
};
