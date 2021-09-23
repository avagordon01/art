#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <array>
#include <utility>
#include <random>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "util/profiler.hh"

#include "engine/glerror.hh"
#include "engine/glfw.hh"
#include "engine/shared-uniforms.hh"
#include "engine/assimp.hh"
#include "engine/font-atlas.hh"

enum class scenes {
    dot_brain,
    scan_brain,
    noise_flow_particles,
    wanikani_subject_grid,
    wanikani_review_time_grid,
    grid_bezier,
    scratch_tmp_new,
};
static constexpr scenes scene = scenes::scratch_tmp_new;
#include "drawables/ticks.hh"
#include "drawables/grid.hh"
#include "drawables/text-overlay.hh"
#include "drawables/text-wanikani.hh"
#include "drawables/lines.hh"
#include "drawables/grid-bezier-evaluator.hh"

#include "scenes/noise-flow-particles.hh"
#include "scenes/brain.hh"
#include "scenes/wanikani-subject-grid.hh"
#include "scenes/wanikani-review-time-grid.hh"
#include "scenes/grid-bezier.hh"
#include "scenes/scratch.hh"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

int main() {
    glfw_t glfw;

    glfwSetInputMode(glfw.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(glfw.window, GLFW_STICKY_KEYS, 1);
    glfwSetKeyCallback(glfw.window, key_callback);

    gl_setup_errors();

    shared_uniforms shared{glfw};

    if (scene == scenes::dot_brain || scene == scenes::scan_brain) {
        brain scene{glfw};
        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.tick();
        }
    } else if (scene == scenes::noise_flow_particles) {
        noise_flow_particles scene{glfw};
        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.tick();
        }
    } else if (scene == scenes::wanikani_subject_grid) {
        wanikani_subject_grid scene{glfw};

        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.tick();
        }

    } else if (scene == scenes::wanikani_review_time_grid) {
        wanikani_review_time_grid scene{glfw};
        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.tick();
        }
    } else if (scene == scenes::grid_bezier) {
        grid_bezier scene{glfw};
        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.tick();
        }

    } else if (scene == scenes::scratch_tmp_new) {
        scratch scene{glfw};
        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.tick();
        }

    }
    std::cout << shared.inputs.time * shared.inputs.framerate / glfwGetTime() << " average fps" << std::endl;
}
