#include "engine/drawable.hh"
#include "drawables/text-overlay.hh"

struct scratch {
    monospace_unicode_font_atlas atlas;
    text_overlay text;

    drawable<> drawable;

    scratch():
        atlas{
            "fonts/unifont-13.0.06.pcf"
            //"fonts/gohufont-2.1/gohufont-11.pcf.gz"
            //"fonts/artwiz-aleczapka-en-1.3/cure.pcf"
            //"fonts/tamsyn-font-1.11/Tamsyn6x12r.pcf"
        },
        text{atlas},
        drawable()
    {
        drawable.vbo.data = {};
        drawable.vbo.bind(drawable.shader.program_vertex, "vertex");
    }
    void draw() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shared.inputs.view = glm::identity<glm::mat4>();
        shared.draw(false);

        {
            text.drawable.vbo.data.clear();
            drawable.vbo.data.clear();
            float w = shared.inputs.resolution_x;
            float h = shared.inputs.resolution_y;
            for (size_t i = 0; i < 8; i++) {
                float x = w / 2 + h / 2;
                float y = h * i / 8 + static_cast<int>(shared.inputs.time * 60) % static_cast<int>(h / 8);
                text.gen_text(L"匚 x = " + std::to_wstring(x / (w / 2)) + L", y = " + std::to_wstring(y / (h / 2)), {x, y});
                drawable.vbo.data.push_back({w / 2, h / 2, 0.0f});
                drawable.vbo.data.push_back({x, y, 0.0f});
            }
        }
        text.draw();
        drawable.draw(GL_LINES);
    }
};
