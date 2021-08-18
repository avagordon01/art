#include <vector>
#include <array>
#include "engine/shader.hh"
#include "engine/drawable.hh"

struct lines_renderer {
    drawable<> drawable;
    shader shader;

    lines_renderer(std::vector<std::array<float, 3>> lines_, shared_uniforms& shared_uniforms):
        drawable(GL_LINES),
        shader(shared_uniforms.header_shader_text + shared_uniforms.passthrough_vertex, shared_uniforms.passthrough_fragment)
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
