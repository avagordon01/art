#include "engine/shader.hh"
#include "engine/drawable.hh"

struct ticks {
    drawable<> drawable;
    shader shader;
    ticks(shared_uniforms& shared_uniforms):
    drawable(),
    shader(
        drawable.quad.vertex_shader,
        shared_uniforms.header_shader_text + R"foo(
layout(pixel_center_integer) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    ivec2 fc_new = fc + ivec2(time * 60) * ivec2(2, 1);

    bool x_tick = (fc_new.x & 0xf) == 0 && fc.y < (2 << clamp(findLSB(fc_new.x >> 4), 1, 4));
    bool y_tick = (fc_new.y & 0xf) == 0 && fc.x < (2 << clamp(findLSB(fc_new.y >> 4), 1, 4));
    colour = vec4(1) * float(x_tick || y_tick);
}
)foo")
    {
        shared_uniforms.bind(shader.program_fragment);
    }
    void draw() {
        shader.draw();
        drawable.draw(GL_QUADS);
    }
};
