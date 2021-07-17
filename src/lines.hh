#include "layer.hh"
#include "util.hh"
#include <vector>
#include <array>
#include <utility>

struct lines_renderer : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    using lines_type = std::vector<
        std::pair<
            std::array<float, 3>,
            std::array<float, 3>
        >
    >;
    lines_type lines;

    lines_renderer(lines_type lines_, std::string shared_uniforms):
        lines(lines_)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(*lines.data()), lines.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        std::string source_vertex = shared_uniforms + R"foo(
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

        program_vertex = create_program(GL_VERTEX_SHADER, source_vertex);
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
