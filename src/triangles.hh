#include <vector>
#include <array>
#include "engine/shader.hh"
#include "engine/drawable.hh"

struct instanced_triangles_renderer {
    drawable<> drawable;
    shared_uniforms& shared_uniforms;
    shader shader;

    instanced_triangles_renderer(std::vector<std::array<float, 3>> vertices_, std::vector<unsigned> indices_, struct shared_uniforms& shared_uniforms_):
        drawable(GL_TRIANGLES, true),
        shared_uniforms(shared_uniforms_),
        shader(shared_uniforms.header_shader_text + R"foo(
in vec3 vertex;
out vec4 vertex_position;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
    vertex_position = projection * view * vec4(vertex, 1.0f);
}
)foo",
        shared_uniforms.header_shader_text + R"foo(
in vec4 vertex_position;
in vec4 gl_FragCoord;
out vec4 colour;

void main() {)foo" +
    ((scene == scenes::dot_brain) ? R"foo(
    colour = vec4(1);
}
    )foo" : R"foo(
    if (mod(time, 10) <= 5) {
        colour = vec4(1) * int(gl_PrimitiveID < time * 6000);
    } else {
        colour = vec4(1) * float(int(vertex_position.y - time * 6) % 16 == 0);
    }
}
)foo"))
    {
        drawable.vbo.data = vertices_;
        drawable.ibo.data = indices_;
        drawable.vbo.bind(shader.program_vertex, "vertex");
        shared_uniforms.bind(shader.program_vertex);
        shared_uniforms.bind(shader.program_fragment);
    }
    void draw() {
        shader.draw();
        drawable.draw([](){
            glLineWidth(1);
            if (scene == scenes::dot_brain) {
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
            } else {
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            }
        });
    }
};
