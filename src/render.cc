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
#include "ticks.hh"
#include "grid.hh"
#include "text-overlay.hh"
#include "lines.hh"
#include "triangles.hh"

#include "assimp.hh"

#include "font-atlas.hh"

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
    ticks ticks{shared};
    grid grid{shared};
    monospace_printable_ascii_font_atlas atlas{
        "fonts/gohufont-2.1/gohufont-11.pcf.gz"
        //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
        //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
    };
    text_overlay text{shared, atlas.header_shader_text};

    std::vector<
        std::pair<
            std::array<float, 3>,
            std::array<float, 3>
        >
    > ls(1000);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> d(-1, 1);
    for (auto& line: ls) {
        line.first = {
            static_cast<float>(d(gen)),
            static_cast<float>(d(gen)),
            static_cast<float>(d(gen)),
        };
        line.second = {
            line.first[0] + 0.1f * static_cast<float>(d(gen)),
            line.first[1] + 0.1f * static_cast<float>(d(gen)),
            line.first[2] + 0.1f * static_cast<float>(d(gen)),
        };
    }
    lines_renderer lines{ls, shared};
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
        shared.inputs.view = glm::identity<glm::mat4>();
        float angle = 2 * M_PI * shared.inputs.frame / 60 / 60;
        float distance = -100.0f;
        shared.inputs.view = glm::lookAt(
            glm::vec3(distance * sin(angle), 0.0f, distance * cos(angle)),
            glm::vec3(),
            glm::vec3(0, 1, 0)
        );

        shared.draw();
        text.draw();
        grid.draw();
        ticks.draw();
        lines.draw();
        triangles.draw();

        glfw.tick();
    }
    std::cout << shared.inputs.frame / glfwGetTime() << " average fps" << std::endl;
}
