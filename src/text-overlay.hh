#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"
#include "font-atlas.hh"

struct text_overlay {
    fullscreen_quad quad;
    vertex_array_object vao;
    shader shader;
    text_overlay(shared_uniforms& shared_uniforms, monospace_printable_ascii_font_atlas& atlas):
    shader(quad.vertex_shader,
        shared_uniforms.header_shader_text + atlas.header_shader_text + R"foo(
layout(origin_upper_left) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy) / textureSize(font_atlas, 0).xy;
    ivec3 texcoord = ivec3(gl_FragCoord.xy, (fc.x + frame) % textureSize(font_atlas, 0).z);
    texcoord = texcoord % textureSize(font_atlas, 0);

    colour = texelFetch(font_atlas, texcoord, 0);
    colour.a = 0.25f;
}
)foo")
    {
        shared_uniforms.bind(shader.program_fragment);
    }
    void draw() {
        vao.draw();
        shader.draw();
        quad.draw();
    }
};
