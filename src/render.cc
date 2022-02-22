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
    {
        brain scene{shared};
        //noise_flow_particles scene{shared};
        //wanikani_subject_grid scene{shared};
        //wanikani_review_time_grid scene{shared};
        //grid_bezier scene{shared};
        //scratch scene{shared};
        //vsync_test scene{shared};
        //icosphere scene{shared};

        //render_to_file rtf(glfw, "test.mp4");

        for (size_t tick = 0; tick < 60 * 3 && !glfwWindowShouldClose(glfw.window); tick++) {
            scene.draw();
            glfw.draw();
            //rtf.draw();
        }
    }
    {
        disintegrate scene{shared};
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.draw();
        }
    }
}
