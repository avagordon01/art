#pragma once

#include <vector>
#include <array>
#include <utility>
#include <random>
#include "buffers.hh"
#include "shader.hh"
#include "noise.hh"
#include "drawable.hh"

struct wanikani_review_time_grid {
    struct point {
        std::array<float, 4> position;
        std::array<float, 4> velocity;
        float timestamp;
        uint32_t stage;
    };
    drawable<point> drawable;

    shader shader;

    wanikani_review_time_grid(shared_uniforms& shared_uniforms):
        drawable(GL_POINTS),
        //extra_buffer({}, GL_DYNAMIC_DRAW),
        shader(
            shared_uniforms.header_shader_text + R"foo(
in vec2 vertex;
in float timestamp;
in uint stage;
out vec4 colour_f;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
float exp_decay(float value, float cutoff, float decay) {
    if (value >= cutoff) {
        return exp(-(value - cutoff) * decay);
    } else {
        return 0.0f;
    }
}
void main() {
    gl_Position = projection * view * vec4(vertex, 0.0f, 1.0f);
    float decay = 0;
    if (stage < 5) {
        decay = 0.1 * pow(0.6, stage);
    } else {
        decay = 0.1 * pow(0.5, stage);
    }

    colour_f = vec4(exp_decay(frame, floor(timestamp), decay));
    if (stage == 9) {
        colour_f = vec4(1, 0, 0, 1);
    }
}
        )foo",
        R"foo(
in vec4 colour_f;
out vec4 colour;

void main() {
    colour = colour_f;
}
        )foo")
    {
        shared_uniforms.bind(shader.program_vertex);
        drawable.vbo.bind(shader.program_vertex, "vertex");
        drawable.vbo.bind(shader.program_vertex, "timestamp", 1, offsetof(point, timestamp));
        drawable.vbo.bind(shader.program_vertex, "stage", 1, offsetof(point, stage), GL_UNSIGNED_INT);
    }
    void draw() {
        shader.draw();
        //extra_buffer.draw();
        drawable.draw();
    }
};
