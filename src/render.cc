#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <array>
#include <utility>
#include <random>

#include "profiler.hh"

#include "glerror.hh"
#include "glfw.hh"
#include "shared-uniforms.hh"
#include "assimp.hh"
#include "font-atlas.hh"

enum class scenes {
    dot_brain,
    scan_brain,
    noise_flow_particles,
    scratch_tmp_new,
};
static constexpr scenes scene = scenes::scratch_tmp_new;
#include "ticks.hh"
#include "grid.hh"
#include "text-overlay.hh"
#include "lines.hh"
#include "triangles.hh"
#include "dust.hh"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

int main() {
    glfw_t glfw;

    glfwSwapInterval(1);
    glfwSetInputMode(glfw.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(glfw.window, GLFW_STICKY_KEYS, 1);
    glfwSetKeyCallback(glfw.window, key_callback);

    gl_setup_errors();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shared_uniforms shared{glfw};

    if (scene == scenes::dot_brain || scene == scenes::scan_brain) {

        ticks ticks{shared};
        grid grid{shared};
        mesh mesh{{
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_left_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_left_rsl_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_right_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/gray_right_rsl_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_left_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_left_rsl_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_right_327680.obj",
            "data/ftp.bigbrainproject.org/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/wavefront-obj/white_right_rsl_327680.obj",
        }, "brain"};
        instanced_triangles_renderer triangles{mesh.vertices, mesh.indices, shared};

        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int w, h;
            glfwGetWindowSize(glfw.window, &w, &h);
            shared.inputs.projection = glm::perspective(glm::radians(75.0f), static_cast<float>(w) / h, 0.1f, 200.f);
            float angle = 2 * M_PI * shared.inputs.frame / 60 / 60;
            float distance = -100.0f;
            shared.inputs.view = glm::lookAt(
                glm::vec3(distance * sin(angle), 0.0f, distance * cos(angle)),
                glm::vec3(),
                glm::vec3(0, 1, 0)
            );

            shared.draw();
            grid.draw();
            ticks.draw();
            triangles.draw();

            glfw.tick();
        }

    } else if (scene == scenes::noise_flow_particles) {

        dust dust{shared};

        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int w, h;
            glfwGetWindowSize(glfw.window, &w, &h);
            shared.inputs.projection = glm::perspective(glm::radians(75.0f), static_cast<float>(w) / h, 0.1f, 200.f);
            shared.inputs.view = glm::lookAt(
                glm::vec3(0.0f, 0.0f, -100.0f),
                glm::vec3(),
                glm::vec3(0, 1, 0)
            );

            shared.draw();
            dust.draw();

            glfw.tick();
        }

    } else if (scene == scenes::scratch_tmp_new) {

        monospace_printable_ascii_font_atlas atlas{
            "fonts/gohufont-2.1/gohufont-11.pcf.gz"
            //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
            //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
        };
        text_overlay text{shared, atlas};

        lines_renderer lines{{}, shared};

        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int w, h;
            glfwGetWindowSize(glfw.window, &w, &h);
            float w_ = w;
            float h_ = h;
            shared.inputs.projection = glm::ortho(0.0f, w_, 0.0f, h_, 0.0f, 200.0f);
            shared.inputs.view = glm::identity<glm::mat4>();
            shared.draw();

            {
                text.vbo.data.clear();
                lines.vbo.data.clear();
                for (size_t i = 0; i < 8; i++) {
                    float x = w / 2 + h / 2;
                    float y = h * i / 8 + shared.inputs.frame % (h / 8);
                    text.gen_text("x = " + std::to_string(x / (w_ / 2)) + ", y = " + std::to_string(y / (h_ / 2)), {x, y});
                    lines.vbo.data.push_back({w_ / 2, h_ / 2, 0.0f});
                    lines.vbo.data.push_back({x, y, 0.0f});
                }
            }
            text.draw();
            lines.draw();

            glfw.tick();
        }

    }
    std::cout << shared.inputs.frame / glfwGetTime() << " average fps" << std::endl;
}
