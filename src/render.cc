#include <iostream>

#include "engine/glfw.hh"
#include "engine/render-to-file.hh"
#include "engine/webgpu.hh"
#include "scenes/brain.hh"
#include "scenes/disintegrate.hh"
#include "scenes/grid-3d.hh"
#include "scenes/grid-bezier.hh"
#include "scenes/icosphere.hh"
#include "scenes/noise-flow-particles.hh"
#include "scenes/orbiter.hh"
#include "scenes/scratch.hh"
#include "scenes/vsync-test.hh"
#include "scenes/wanikani-review-time-grid.hh"
#include "scenes/wanikani-subject-grid.hh"

int main() {
    {
        webgpu_t webgpu{};
        while (!glfwWindowShouldClose(glfw.window)) {
            webgpu.draw();
            glfwPollEvents();
        }
    }
    {
        //brain scene{};
        //noise_flow_particles scene{};
        //wanikani_subject_grid scene{};
        //wanikani_review_time_grid scene{};
        //grid_bezier scene{};
        //scratch scene{};
        //vsync_test scene{};
        //icosphere scene{};
        //grid_3d scene{};
        orbiter scene{};

        //render_to_file rtf("test.mp4");

        while (!glfwWindowShouldClose(glfw.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            scene.draw();
            glfw.draw();
            //rtf.draw();
            shared.tick();
            shared.draw();
        }
    }
    {
        disintegrate scene{};
        while (!glfwWindowShouldClose(glfw.window)) {
            scene.draw();
            glfw.draw();
        }
    }
}
