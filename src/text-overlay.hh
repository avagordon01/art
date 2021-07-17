#include "layer.hh"
#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"

struct text_overlay : layer_t {
    fullscreen_quad quad;
    vertex_array_object vao;
    shader shader;
    text_overlay(std::string shared_uniforms, std::string atlas):
    shader(quad.vertex_shader,
        shared_uniforms + atlas + R"foo(
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
    {}
    void draw() override {
        vao.draw();
        shader.draw();
        quad.draw();
    }
};
