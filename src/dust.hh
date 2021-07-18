#include <vector>
#include <array>
#include <utility>
#include <random>
#include "buffers.hh"
#include "shader.hh"

struct dust {
    struct point {
        std::array<float, 3> position;
        std::array<float, 3> velocity;
    };
    vertex_array_object vao;
    vertex_buffer<point> vbo;

    shader shader;

    std::vector<point> gen_points() {
        std::vector<point> points_data(1000);
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
        vbo{gen_points(), false},
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
)foo")
    {
        vbo.bind(shader.program_vertex, "vertex");
    }
    void draw() {
        vao.draw();
        vbo.draw();
        shader.draw();
        glPointSize(1);
        glDrawArrays(GL_POINTS, 0, vbo.data.size());
        float dt = 1.0f / 60;
        for (auto& p: vbo.data) {
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
        }
    }
};
