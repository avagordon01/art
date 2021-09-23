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
static constexpr scenes scene = scenes::scan_brain;
#include "drawables/ticks.hh"
#include "drawables/grid.hh"
#include "drawables/text-overlay.hh"
#include "drawables/text-wanikani.hh"
#include "drawables/lines.hh"
#include "drawables/wanikani-review-time-grid.hh"
#include "drawables/grid-bezier-evaluator.hh"

#include "scenes/noise-flow-particles.hh"
#include "scenes/brain.hh"

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

        monospace_unicode_font_atlas atlas{
            "fonts/unifont-13.0.06.pcf"
        };
        text_wanikani text{shared, atlas};

        std::ifstream i("data/wanikani/data.json");
        json j;
        i >> j;

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
                text.drawable.vbo.data.clear();
                text.extra_buffer.data.clear();
                size_t s = 24;
                size_t rows = h / s;
                size_t i = 0;
                float start_timestamp = j["start_timestamp"];
                float end_timestamp = j["end_timestamp"];
                for (json::iterator it = j["subjects"].begin(); it != j["subjects"].end(); ++it, i++) {
                    float x = floor(i / rows) * s;
                    float y = floor(i % rows) * s;
                    std::string character = (*it)["character"];
                    auto& updates = (*it)["updates"];
                    float timestamp = std::numeric_limits<float>::infinity();
                    uint32_t stage = 1;
                    for (json::iterator jt = updates.begin(); jt != updates.end(); ++jt) {
                        float next_timestamp = (*jt)["data_updated_at"];
                        next_timestamp = (next_timestamp - start_timestamp) / (end_timestamp - start_timestamp) * 100;
                        if (shared.inputs.time > next_timestamp) {
                            timestamp = next_timestamp;
                            stage = (*jt)["ending_srs_stage"];
                        } else {
                            break;
                        }
                    }
                    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                    std::wstring wide = converter.from_bytes(character);
                    text.gen_text(wide, {x, y});
                    for (size_t i = 0; i < 6; i++) {
                        text.extra_buffer.data.push_back({timestamp, stage});
                    }
                }
            }
            text.draw();

            glfw.tick();
        }

    } else if (scene == scenes::wanikani_review_time_grid) {

        wanikani_review_time_grid wanikani_review_time_grid{shared};

        std::ifstream i("data/wanikani/data2.json");
        json j;
        i >> j;

        glfwSetTime(0);

        int w, h;
        glfwGetWindowSize(glfw.window, &w, &h);
        {
            float seconds_per_day = 24 * 60 * 60;
            float start_timestamp = j["start_timestamp"];
            float end_timestamp = j["end_timestamp"];
            start_timestamp = floor(start_timestamp / seconds_per_day) * seconds_per_day;
            for (json::iterator it = j["reviews"].begin(); it != j["reviews"].end(); ++it) {
                float timestamp = (*it)["data_updated_at"];
                timestamp -= start_timestamp;
                float x = 4 * floor(timestamp / seconds_per_day);
                float y = h * (timestamp - floor(timestamp / seconds_per_day) * seconds_per_day) / seconds_per_day;
                uint32_t stage = (*it)["ending_srs_stage"];
                timestamp /= (end_timestamp - start_timestamp);
                timestamp *= 10 * 2;
                wanikani_review_time_grid.drawable.vbo.data.push_back({{x, y}, {}, timestamp, stage});
            }
        }
        while (!glfwWindowShouldClose(glfw.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int w, h;
            glfwGetWindowSize(glfw.window, &w, &h);
            float w_ = w;
            float h_ = h;
            shared.inputs.projection = glm::ortho(0.0f, w_, 0.0f, h_, 0.0f, 200.0f);
            shared.inputs.view = glm::identity<glm::mat4>();
            shared.draw();

            wanikani_review_time_grid.draw();

            glfw.tick();
        }

    } else if (scene == scenes::grid_bezier) {

        grid_bezier_evaluator<2, glm::vec3> gbe{shared, GL_POINT, 32, 32};
        gbe.controls = {
            glm::vec3{0, 0, -10},
            glm::vec3{5, 5, -5},
            glm::vec3{10, 0, -10},

            glm::vec3{0, 5, -5},
            glm::vec3{5, 5, -5},
            glm::vec3{10, 5, -5},

            glm::vec3{0, 10, -10},
            glm::vec3{5, 5, -5},
            glm::vec3{10, 10, -10},
        };

        glfwSetTime(0);
        while (!glfwWindowShouldClose(glfw.window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int w, h;
            glfwGetWindowSize(glfw.window, &w, &h);
            shared.inputs.projection = glm::perspective(glm::radians(75.0f), static_cast<float>(w) / h, 0.1f, 200.f);
            shared.inputs.view = glm::lookAt(
                glm::vec3(0, 0, 20),
                glm::vec3(0, 0, -10),
                glm::vec3(0, 1, 0)
            );
            shared.draw();

            gbe.draw();

            glfw.tick();
        }

    } else if (scene == scenes::scratch_tmp_new) {

        monospace_unicode_font_atlas atlas{
            "fonts/unifont-13.0.06.pcf"
            //"fonts/gohufont-2.1/gohufont-11.pcf.gz"
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
                text.drawable.vbo.data.clear();
                lines.drawable.vbo.data.clear();
                for (size_t i = 0; i < 8; i++) {
                    float x = w / 2 + h / 2;
                    float y = h * i / 8 + static_cast<int>(shared.inputs.time * 60) % (h / 8);
                    text.gen_text(L"匚 x = " + std::to_wstring(x / (w_ / 2)) + L", y = " + std::to_wstring(y / (h_ / 2)), {x, y});
                    lines.drawable.vbo.data.push_back({w_ / 2, h_ / 2, 0.0f});
                    lines.drawable.vbo.data.push_back({x, y, 0.0f});
                }
            }
            text.draw();
            lines.draw();

            glfw.tick();
        }

    }
    std::cout << shared.inputs.time * shared.inputs.framerate / glfwGetTime() << " average fps" << std::endl;
}
