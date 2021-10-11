#include <iostream>

#include "engine/glfw.hh"

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

int main() {
    glfw_t glfw;
    {
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
    }
    {
        disintegrate scene{glfw};
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.draw();
        }
    }
}
