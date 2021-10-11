#include <iostream>

#include "engine/glfw.hh"
#include "engine/glerror.hh"

#include "scenes/noise-flow-particles.hh"
#include "scenes/brain.hh"
#include "scenes/wanikani-subject-grid.hh"
#include "scenes/wanikani-review-time-grid.hh"
#include "scenes/grid-bezier.hh"
#include "scenes/scratch.hh"
#include "scenes/vsync-test.hh"
#include "scenes/icosphere.hh"
#include "scenes/disintegrate.hh"

#include "engine/render-to-file.hh"

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

    brain scene{glfw, true};
    //noise_flow_particles scene{glfw};
    //wanikani_subject_grid scene{glfw};
    //wanikani_review_time_grid scene{glfw};
    //grid_bezier scene{glfw};
    //scratch scene{glfw};
    //vsync_test scene{glfw};
    //icosphere scene{glfw};

    //render_to_file rtf(glfw, "test.mp4");

    for (size_t tick = 0; tick < 60 * 3 && !glfwWindowShouldClose(glfw.window); tick++) {
        scene.draw();
        glfw.draw();
        //rtf.draw();
    }
    scene.~brain();
    disintegrate scene2{glfw};
    while (!glfwWindowShouldClose(glfw.window)) {
        scene2.draw();
        glfw.draw();
    }
}
