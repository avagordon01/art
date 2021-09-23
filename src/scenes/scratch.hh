#include "engine/buffers.hh"
#include "engine/shader.hh"
#include "engine/shared-uniforms.hh"

struct scratch {
    glfw_t& glfw;
    shared_uniforms shared;

    monospace_unicode_font_atlas atlas;
    text_overlay text;
    lines_renderer lines;

    scratch(glfw_t& glfw_):
        glfw{glfw_},
        shared{glfw},

        atlas{
            "fonts/unifont-13.0.06.pcf"
            //"fonts/gohufont-2.1/gohufont-11.pcf.gz"
            //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
            //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
        },
        text{shared, atlas},
        lines{{}, shared}
    {}
    void draw() {
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
    }
};
