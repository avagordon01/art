#include <vector>
#include <array>
#include <utility>
#include <random>
#include "buffers.hh"
#include "shader.hh"

struct dust {
    struct point {
        std::array<float, 4> position;
        std::array<float, 4> velocity;
    };
    vertex_array_object vao;
    vertex_buffer<point> vbo;
    storage_buffer<point> sbo;

    shader shader;
    compute_shader compute_shader;

    static constexpr size_t num_points = 1000;

    std::vector<point> gen_points() {
        std::vector<point> points_data(num_points);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> u(-200, 200);
        std::normal_distribution<float> n(0, 1);
        for (auto& point: points_data) {
            point.position = {
                static_cast<float>(u(gen)),
                static_cast<float>(u(gen)),
                static_cast<float>(u(gen)),
            };
            point.velocity = {
                static_cast<float>(n(gen)),
                static_cast<float>(n(gen)),
                static_cast<float>(n(gen)),
            };
        }
        return points_data;
    }

    dust(std::string shared_uniforms):
        vbo{std::vector<point>{num_points}, GL_DYNAMIC_COPY},
        sbo{gen_points(), GL_DYNAMIC_COPY},
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
)foo"),
        compute_shader(R"foo(
struct point {
    vec4 position;
    vec4 velocity;
};
layout(std430) buffer vertices_buffer {
    point vertices[];
};

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;
    vertices[index].position += vertices[index].velocity / 60.0f;
}
)foo", {num_points, 1, 1})
    {
        vbo.bind(shader.program_vertex, "vertex", 3, GL_FLOAT, GL_FALSE, sizeof(point), nullptr);
        sbo.bind(compute_shader.program, "vertices_buffer");
    }
    void draw() {
        vao.draw();
        vbo.draw();
        shader.draw();
        glPointSize(1);
        glDrawArrays(GL_POINTS, 0, vbo.data.size());
        compute_shader.draw();
        glCopyBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_ARRAY_BUFFER, 0, 0, sizeof(point) * num_points);
    }
};
