#include "engine/drawable.hh"

struct inverter {
    drawable<> drawable;
    inverter():
        drawable(drawable.quad.vertex_shader,
        R"foo(
out vec4 colour;

void main() {
    colour = vec4(1);
}
)foo")
    {}
    void draw() {
        drawable.draw(GL_QUADS, false, [](){
            glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZERO, GL_ONE, GL_ZERO);
        });
    }
};
