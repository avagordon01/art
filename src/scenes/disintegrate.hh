#include <vector>
#include <array>
#include <utility>
#include <random>
#include "engine/drawable.hh"
#include "engine/shader.hh"
#include "noise.hh"

struct disintegrate {
    shared_uniforms& shared;

    struct point {
        std::array<float, 4> position;
        std::array<float, 4> velocity;
    };
    std::vector<point> points_data;
    drawable<point> drawable;
    storage_buffer<point> sbo;

    shader shader;
    compute_shader compute_shader;

    std::vector<point> gen_points() {
        size_t w = shared.inputs.resolution_x, h = shared.inputs.resolution_y;
        std::vector<std::array<uint8_t, 3>> buffer(w * h);
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
        std::vector<point> points_data;
        for (size_t x = 0; x < w; x++) {
            for (size_t y = 0; y < h; y++) {
                size_t i = y * w + x;
                if (buffer[i][0] != 0 || buffer[i][1] != 0 || buffer[i][2] != 0) {
                    point p;
                    p.position = {
                        static_cast<float>(x) + 1,
                        static_cast<float>(y) + 1,
                        -1.0f
                    };
                    p.velocity = {};
                    points_data.push_back(p);
                }
            }
        }

        return points_data;
    }
    disintegrate(shared_uniforms& shared_):
        shared{shared_},
        points_data{gen_points()},
        drawable(),
        sbo{points_data, GL_DYNAMIC_COPY},
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
    vec4 acceleration = vec4(dir * 8, 0.0f, 0.0f);
    vertices[index].velocity += acceleration * dt;
    float speed = length(vertices[index].velocity);
    if (speed > 10) {
        float new_speed = 10;
        vertices[index].velocity *= new_speed / speed;
    }
    vertices[index].position += vertices[index].velocity * dt;
}
)foo", {points_data.size(), 1, 1})
    {
        shared.bind(shader.program_vertex);
        shared.bind(compute_shader.program);
        drawable.vbo = {std::vector<point>{points_data.size()}, GL_DYNAMIC_COPY};
        drawable.vbo.bind(shader.program_vertex, "vertex", &point::position);
        sbo.bind(compute_shader.program, "vertices_buffer");
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shared.inputs.view = glm::identity<glm::mat4>();
        shared.draw(false);

        shader.draw();
        drawable.draw(GL_POINTS);
        compute_shader.draw();
        glCopyBufferSubData(GL_SHADER_STORAGE_BUFFER, GL_ARRAY_BUFFER, 0, 0, sizeof(point) * drawable.vbo.data.size());
    }
};
