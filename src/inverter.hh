#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"
#include "drawable.hh"

struct inverter {
    drawable<> drawable;
    shader shader;
    inverter():
        drawable(GL_QUADS),
        shader(drawable.quad.vertex_shader,
        R"foo(
out vec4 colour;

void main() {
    colour = vec4(1);
}
)foo")
    {}
    void draw() {
        shader.draw();
        drawable.draw([](){
            glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZERO, GL_ONE, GL_ZERO);
        });
    }
};
