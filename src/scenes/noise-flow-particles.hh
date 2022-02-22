#include <vector>
#include <array>
#include <utility>
#include <random>
#include "engine/drawable.hh"
#include "engine/shader.hh"
#include "noise.hh"

struct noise_flow_particles {
    shared_uniforms& shared;

    struct point {
        std::array<float, 4> position;
        std::array<float, 4> velocity;
    };
    drawable<point> drawable;
    storage_buffer<point> sbo;

    shader shader;
    compute_shader compute_shader;

    static constexpr size_t num_points = 1000000;

    std::vector<point> gen_points() {
        std::vector<point> points_data(num_points);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> u(-200, 200);
        std::normal_distribution<float> n(0, 1);
        for (auto& point: points_data) {
            point.position = {
                0.0f,
                0.0f,
                0.0f,
            };
            point.velocity = {
                static_cast<float>(n(gen)) * 0.004f,
                static_cast<float>(n(gen)) * 0.004f,
                static_cast<float>(n(gen)) * 0.004f,
            };
        }
        return points_data;
    }
    noise_flow_particles(shared_uniforms& shared_):
        shared{shared_},
        drawable(),
        sbo{gen_points(), GL_DYNAMIC_COPY},
        shader(),
        compute_shader(noise_header_text + R"foo(
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
    vec2 dir = vec2(
        noise(vec4(vertices[index].position.xyz / vec3(2, 3, 4), time)),
        noise(vec4(vertices[index].position.xyz / vec3(4, 3, 2), time))
    ) - vec2(0.5f);
    float dt = 1.0f / 60.0f;
    vec4 acceleration = vec4(dir * 20, 0.0f, 0.0f);
    vertices[index].velocity += acceleration * dt;
    float speed = length(vertices[index].velocity);
    if (speed > 10) {
        float new_speed = 10;
        vertices[index].velocity *= new_speed / speed;
    }
    vertices[index].position += vertices[index].velocity * dt;
}
)foo", {num_points, 1, 1})
    {
        shared.bind(shader.program_vertex);
        shared.bind(compute_shader.program);
        drawable.vbo = {std::vector<point>{num_points}, GL_DYNAMIC_COPY};
        drawable.vbo.bind(shader.program_vertex, "vertex", &point::position);
        sbo.bind(compute_shader.program, "vertices_buffer");
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shared.inputs.view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, -100.0f),
            glm::vec3(),
            glm::vec3(0, 1, 0)
        );
        shared.draw();

        shader.draw();
        drawable.draw(GL_POINTS);
        compute_shader.draw();
        glCopyBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_ARRAY_BUFFER, 0, 0, sizeof(point) * num_points);
    }
};
