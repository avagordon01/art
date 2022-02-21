#include "engine/shader.hh"
#include "engine/shared-uniforms.hh"
#include "engine/drawable.hh"

struct vsync_test {
    shared_uniforms& shared;

    drawable<> drawable;
    shader shader;
    vsync_test(shared_uniforms& shared_):
    shared{shared_},
    drawable{},
    shader{
        drawable.quad.vertex_shader,
        shared.header_shader_text + R"foo(
layout(pixel_center_integer) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    ivec2 fc_new = fc + ivec2(time * framerate);

    if (fc.y > 1080 / 2) {
        colour = vec4(1) * float((fc_new.x & 0xff) == 0);
    } else {
        colour = vec4(1) * float((fc.x & 0xff) == 0);
    }
}
)foo"}
    {
        shared.bind(shader.program_fragment);
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shared.draw();
        shader.draw();
        drawable.draw(GL_QUADS);
    }
};
