#include "layer.hh"
#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"

struct ticks : layer_t {
    fullscreen_quad quad;
    vertex_array_object vao;
    shader shader;
    ticks(std::string shared_uniforms):
    shader(
        quad.vertex_shader,
        shared_uniforms + R"foo(
layout(pixel_center_integer) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    ivec2 fc_new = fc + ivec2(frame) * ivec2(2, 1);

    bool x_tick = (fc_new.x & 0xf) == 0 && fc.y < (2 << clamp(findLSB(fc_new.x >> 4), 1, 4));
    bool y_tick = (fc_new.y & 0xf) == 0 && fc.x < (2 << clamp(findLSB(fc_new.y >> 4), 1, 4));
    colour = vec4(1) * float(x_tick || y_tick);
}
)foo")
    {}
    void draw() override {
        vao.draw();
        shader.draw();
        quad.draw();
    }
};
