#include "layer.hh"
#include <string>
#include "buffers.hh"
#include "shader.hh"
#include "fullscreen-quad.hh"

struct grid : layer_t {
    fullscreen_quad quad;
    vertex_array_object vao;
    shader shader;
    grid(std::string shared_uniforms):
        shader(quad.vertex_shader,
        shared_uniforms + R"foo(
layout(pixel_center_integer) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    //ivec2 fc_new = fc + ivec2(frame) * ivec2(2, 1);

    ivec2 fc_mod = fc % ivec2(64);

    bool x = (fc_mod.x == 0);
    bool y = (fc_mod.y == 0);
    colour = vec4(vec3(1), 0.25) * float(x || y);

    bool x_plus = x && (fc_mod.y < 8 || fc_mod.y > 56);
    bool y_plus = y && (fc_mod.x < 8 || fc_mod.x > 56);
    colour += vec4(1) * float(x_plus || y_plus);
}
)foo")
    {}
    void draw() override {
        vao.draw();
        shader.draw();
        quad.draw();
    }
};
