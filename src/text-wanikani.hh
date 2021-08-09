#pragma once

#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"
#include "font-atlas.hh"

struct text_wanikani {
    fullscreen_quad quad;
    vertex_array_object vao;
    struct char_vertex {
        std::array<float, 2> vertex;
        std::array<float, 3> texcoords;
    };
    monospace_unicode_font_atlas& atlas;
    vertex_buffer<char_vertex> vbo;
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
            vbo.data.push_back({{x,     y + h}, {0.0f, 0.0f, char_index}});
            vbo.data.push_back({{x,     y    }, {0.0f, h,    char_index}});
            vbo.data.push_back({{x + w, y    }, {w,    h,    char_index}});

            vbo.data.push_back({{x,     y + h}, {0.0f, 0.0f, char_index}});
            vbo.data.push_back({{x + w, y    }, {w,    h,    char_index}});
            vbo.data.push_back({{x + w, y + h}, {w,    0.0f, char_index}});
            x += w;
        }
    }
    shader shader;
    text_wanikani(shared_uniforms& shared_uniforms, monospace_unicode_font_atlas& atlas_):
    atlas(atlas_),
    vbo({}, GL_DYNAMIC_DRAW),
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
    texcoords_f = texcoords;
    colour_f = vec4(exp_decay(frame, timestamp, 0.001));
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
        vbo.bind(shader.program_vertex, "vertex", 2, GL_FLOAT, GL_FALSE, sizeof(char_vertex), (void*)offsetof(char_vertex, vertex));
        vbo.bind(shader.program_vertex, "texcoords", 3, GL_FLOAT, GL_FALSE, sizeof(char_vertex), (void*)offsetof(char_vertex, texcoords));
        extra_buffer.bind(shader.program_vertex, "timestamp", 1, GL_FLOAT, GL_FALSE, sizeof(extra_data), (void*)offsetof(extra_data, timestamp));
        //extra_buffer.bind(shader.program_vertex, "stage", 1, GL_UNSIGNED_INT, GL_FALSE, sizeof(extra_data), (void*)offsetof(extra_data, stage));
        {
            glBindBuffer(GL_ARRAY_BUFFER, extra_buffer.buffer_id);
            GLint attrib_index = glGetAttribLocation(shader.program_vertex, "stage");
            glVertexAttribIPointer(attrib_index, 1, GL_UNSIGNED_INT, sizeof(extra_data), (void*)offsetof(extra_data, stage));
            glEnableVertexAttribArray(attrib_index);
        }
        atlas.bind(shader.program_fragment);
    }
    void draw() {
        vao.draw();
        vbo.draw();
        extra_buffer.draw();
        shader.draw();
        glDrawArrays(GL_TRIANGLES, 0, vbo.data.size());
    }
};
