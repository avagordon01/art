#include <string>
#include <vector>
#include <array>
#include "buffers.hh"
#include "shader.hh"

struct instanced_triangles_renderer {
    vertex_array_object vao;
    vertex_buffer<std::array<float, 3>> vbo;
    index_buffer ibo;

    shared_uniforms& shared_uniforms;

    shader shader;

    instanced_triangles_renderer(std::vector<std::array<float, 3>> vertices_, std::vector<unsigned> indices_, struct shared_uniforms& shared_uniforms_):
        vbo(vertices_),
        ibo(indices_),
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
    if (frame % (60 * 10) <= 60 * 5) {
        colour = vec4(1) * int(gl_PrimitiveID < frame * 100);
    } else {
        colour = vec4(1) * float(int(vertex_position.y - float(frame) * 0.1) % 16 == 0);
    }
}
)foo"))
    {
        vbo.bind(shader.program_vertex, "vertex");
        shared_uniforms.bind(shader.program_vertex);
        shared_uniforms.bind(shader.program_fragment);
    }
    void draw() {
        vao.draw();
        vbo.draw();
        ibo.draw();
        shader.draw();
        glLineWidth(1);
        if (scene == scenes::dot_brain) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        glDrawElements(GL_TRIANGLES, ibo.data.size(), GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
};
