#include "engine/drawable.hh"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct wanikani_review_time_grid {
    struct point {
        std::array<float, 4> position;
        std::array<float, 4> velocity;
        float timestamp;
        uint32_t stage;
    };
    drawable<point> drawable;

    json j;

    wanikani_review_time_grid():
        //extra_buffer({}, GL_DYNAMIC_DRAW),
        drawable(
            R"foo(
in vec2 vertex;
in float timestamp;
in uint stage;
out vec4 colour_f;

out gl_PerVertex {
    vec4 gl_Position;
};
float exp_decay(float value, float cutoff, float decay) {
    if (value >= cutoff) {
        return exp(-(value - cutoff) * decay);
    } else {
        return 0.0f;
    }
}
void main() {
    gl_Position = projection * view * vec4(vertex, 0.0f, 1.0f);
    float decay = 0;
    if (stage < 5) {
        decay = 6 * pow(0.6, stage);
    } else {
        decay = 6 * pow(0.5, stage);
    }

    colour_f = vec4(exp_decay(time, timestamp, decay));
    if (stage == 9) {
        colour_f = vec4(1, 0, 0, 1);
    }
}
        )foo",
        R"foo(
in vec4 colour_f;
out vec4 colour;

void main() {
    colour = colour_f;
}
        )foo")
    {
        std::ifstream i("data/wanikani/data2.json");
        i >> j;

        {
            float seconds_per_day = 24 * 60 * 60;
            float start_timestamp = j["start_timestamp"];
            float end_timestamp = j["end_timestamp"];
            start_timestamp = floor(start_timestamp / seconds_per_day) * seconds_per_day;
            for (json::iterator it = j["reviews"].begin(); it != j["reviews"].end(); ++it) {
                float timestamp = (*it)["data_updated_at"];
                timestamp -= start_timestamp;
                float x = 4 * floor(timestamp / seconds_per_day);
                float y = shared.inputs.resolution_y * (timestamp - floor(timestamp / seconds_per_day) * seconds_per_day) / seconds_per_day;
                uint32_t stage = (*it)["ending_srs_stage"];
                timestamp /= (end_timestamp - start_timestamp);
                timestamp *= 10 * 2;
                drawable.vbo.data.push_back({{x, y}, {}, timestamp, stage});
            }
        }

        drawable.vbo.bind(drawable.shader.program_vertex, "vertex", &point::position);
        drawable.vbo.bind(drawable.shader.program_vertex, "timestamp", &point::timestamp);
        drawable.vbo.bind(drawable.shader.program_vertex, "stage", &point::stage);
    }
    void draw() {
        shared.inputs.view = glm::identity<glm::mat4>();
        shared.draw(projection_mode::orthogonal);

        //extra_buffer.draw();
        drawable.draw(GL_POINTS);
    }
};
