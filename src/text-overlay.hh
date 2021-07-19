#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"
#include "font-atlas.hh"

struct text_overlay {
    fullscreen_quad quad;
    vertex_array_object vao;
    struct char_vertex {
        std::array<float, 2> vertex;
        std::array<float, 3> texcoords;
    };
    monospace_printable_ascii_font_atlas& atlas;
    vertex_buffer<char_vertex> vbo;

    std::vector<char_vertex> gen_text(std::string text) {
        std::vector<char_vertex> char_data;
        float xpos = 0.0f;
        float ypos = 0.0f;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            float char_index = c - ' ';
            float w = atlas.width();
            float h = atlas.height();
            std::vector<char_vertex> vertices = {
                {{xpos,     ypos + h}, {0.0f, 0.0f, char_index}},
                {{xpos,     ypos    }, {0.0f, h,    char_index}},
                {{xpos + w, ypos    }, {w,    h,    char_index}},

                {{xpos,     ypos + h}, {0.0f, 0.0f, char_index}},
                {{xpos + w, ypos    }, {w,    h,    char_index}},
                {{xpos + w, ypos + h}, {w,    0.0f, char_index}}
            };
            xpos += w;
            char_data.insert(char_data.end(), vertices.begin(), vertices.end());
        }
        return char_data;
    }
    shader shader;
    text_overlay(shared_uniforms& shared_uniforms, monospace_printable_ascii_font_atlas& atlas_):
    atlas(atlas_),
    vbo(gen_text("XA__test__AX"), GL_DYNAMIC_DRAW),
    shader(
        shared_uniforms.header_shader_text + R"foo(
in vec2 vertex;
in vec3 texcoords;
out vec3 texcoords_f;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = vec4(vertex.xy / resolution * 2 - 1, 0.0, 1.0);
    texcoords_f = texcoords;
}
)foo",
        shared_uniforms.header_shader_text + atlas.header_shader_text + R"foo(
in vec3 texcoords_f;
out vec4 colour;

void main() {
    colour = texelFetch(font_atlas, ivec3(texcoords_f), 0);
}
)foo")
    {
        shared_uniforms.bind(shader.program_fragment);
        vbo.bind(shader.program_vertex, "vertex", 2, GL_FLOAT, GL_FALSE, sizeof(char_vertex), (void*)offsetof(char_vertex, vertex));
        vbo.bind(shader.program_vertex, "texcoords", 3, GL_FLOAT, GL_FALSE, sizeof(char_vertex), (void*)offsetof(char_vertex, texcoords));
    }
    void draw() {
        vao.draw();
        vbo.draw();
        shader.draw();
        glDrawArrays(GL_TRIANGLES, 0, vbo.data.size());
    }
};
