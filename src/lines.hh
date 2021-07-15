#include "layer.hh"
#include "util.hh"

struct lines_renderer : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    std::vector<
        std::pair<
            std::array<float, 3>,
            std::array<float, 3>
        >
    > lines;

    template<typename ...S>
    lines_renderer(S... program_texts) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        lines.resize(1000000);
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<float> d(-1, 1);
        for (auto& line: lines) {
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

        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(*lines.data()), lines.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        std::string source_vertex = R"foo(
in vec3 vertex;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
}
)foo";
        std::string source_fragment = R"foo(
in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    colour = vec4(1);
}
)foo";

        program_vertex = create_program(GL_VERTEX_SHADER, program_texts..., source_vertex);
        program_fragment = create_program(GL_FRAGMENT_SHADER, source_fragment);

        glGenProgramPipelines(1, &pipeline_render);
        glUseProgramStages(pipeline_render, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline_render, GL_FRAGMENT_SHADER_BIT, program_fragment);
        glBindProgramPipeline(pipeline_render);

        glBindAttribLocation(pipeline_render, 0, "vertex");

        glBindVertexArray(0);
    }
    void draw() override {
        glBindVertexArray(vao);
        glBindProgramPipeline(pipeline_render);
        glLineWidth(1);
        glDrawArrays(GL_LINES, 0, lines.size() * 2);
    }
};
