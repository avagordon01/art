#include "engine/buffers.hh"
#include "engine/shader.hh"
#include "engine/shared-uniforms.hh"

struct template_scene {
    glfw_t& glfw;
    shared_uniforms shared;

    template_scene(glfw_t& glfw_):
        glfw{glfw_},
        shared{glfw},
    {
    }
    void draw() {
    }
};
