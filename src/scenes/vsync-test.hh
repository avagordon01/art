#include "engine/drawable.hh"

struct vsync_test {
    drawable<> drawable;
    vsync_test():
    drawable{
        fullscreen_quad{}.vertex_shader,
        R"foo(
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
    }
    void draw() {
        shared.draw();
        drawable.draw(GL_QUADS);
    }
};
