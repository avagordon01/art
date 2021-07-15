#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdbool>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <utility>
#include <random>
#include <chrono>

#include "glfw.hh"
#include "shared-uniforms.hh"
#include "julia.hh"
#include "text-overlay.hh"
#include "lines.hh"

#include "font-atlas.hh"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

int main() {
    glfw_t glfw;

    glfwSwapInterval(0);
    glfwSetInputMode(glfw.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(glfw.window, GLFW_STICKY_KEYS, 1);
    glfwSetKeyCallback(glfw.window, key_callback);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shared_uniforms s{glfw};
    juliaset j{s.header_shader_text};
    monospace_printable_ascii_font_atlas atlas{
        "fonts/gohufont-2.1/gohufont-11.pcf.gz"
        //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
        //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
    };
    text_overlay t{s.header_shader_text, atlas.header_shader_text};
    lines_renderer l{};

    std::vector<layer_t*> layers {&s, &j, &t, &l};

    glfwSetTime(0);
    while (!glfwWindowShouldClose(glfw.window)) {
        for (auto& layer: layers) {
            layer->draw();
        }
        glfw.tick();
    }
    std::cout << s.inputs.frame / glfwGetTime() << " average fps" << std::endl;
}
