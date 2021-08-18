#pragma once

#include <string>
#include "engine/buffers.hh"
#include "engine/shader.hh"
#include "engine/font-atlas.hh"
#include "engine/drawable.hh"

struct text_wanikani {
    struct char_vertex {
        std::array<float, 2> vertex;
        std::array<float, 3> texcoords;
    };
    monospace_unicode_font_atlas& atlas;
    drawable<char_vertex> drawable;
    struct extra_data {
        float timestamp;
        uint32_t stage;
    };
    vertex_buffer<extra_data> extra_buffer;

    void gen_text(std::wstring text, std::array<float, 2> position) {
        auto[x, y] = position;
        for (uint32_t c: text) {
            float char_index = c;
            float w = atlas.width();
            float h = atlas.height();
            drawable.vbo.data.push_back({{x,     y + h}, {0.0f, 0.0f, char_index}});
            drawable.vbo.data.push_back({{x,     y    }, {0.0f, h,    char_index}});
            drawable.vbo.data.push_back({{x + w, y    }, {w,    h,    char_index}});

            drawable.vbo.data.push_back({{x,     y + h}, {0.0f, 0.0f, char_index}});
            drawable.vbo.data.push_back({{x + w, y    }, {w,    h,    char_index}});
            drawable.vbo.data.push_back({{x + w, y + h}, {w,    0.0f, char_index}});
            x += w;
        }
    }
    shader shader;
    text_wanikani(shared_uniforms& shared_uniforms, monospace_unicode_font_atlas& atlas_):
    atlas(atlas_),
    drawable(GL_TRIANGLES),
    extra_buffer({}, GL_DYNAMIC_DRAW),
    shader(
        shared_uniforms.header_shader_text + R"foo(
in vec2 vertex;
in vec3 texcoords;
out vec3 texcoords_f;
in float timestamp;
in uint stage;
out vec4 colour_f;

out gl_PerVertex {
    vec4 gl_Position;
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
    texcoords_f = texcoords;
    float decay = 0;
    if (stage < 5) {
        decay = 6 * pow(0.6, stage);
    } else {
        decay = 6 * pow(0.5, stage);
    }

    colour_f = vec4(exp_decay(time, timestamp, decay));
    if (stage == 9) {
        colour_f = vec4(1, 0, 0, 1);
    }
}
)foo",
        shared_uniforms.header_shader_text + atlas.header_shader_text + R"foo(
in vec3 texcoords_f;
in vec4 colour_f;
out vec4 colour;

void main() {
    colour = colour_f * vec4(bitmap_font_atlas_fetch(ivec3(texcoords_f)));
}
)foo")
    {
        shared_uniforms.bind(shader.program_fragment);
        drawable.vbo.bind(shader.program_vertex, "vertex", 2, offsetof(char_vertex, vertex));
        drawable.vbo.bind(shader.program_vertex, "texcoords", 3, offsetof(char_vertex, texcoords));
        extra_buffer.bind(shader.program_vertex, "timestamp", 1, offsetof(extra_data, timestamp));
        extra_buffer.bind(shader.program_vertex, "stage", 1, offsetof(extra_data, stage), GL_UNSIGNED_INT);
        atlas.bind(shader.program_fragment);
    }
    void draw() {
        shader.draw();
        extra_buffer.draw();
        drawable.draw();
    }
};
