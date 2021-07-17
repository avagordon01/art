#include "layer.hh"
#include "util.hh"
#include <string>
#include <vector>
#include <array>

struct instanced_triangles_renderer : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    std::vector<std::array<float, 3>> vertices;
    std::vector<unsigned> indices;

    template<typename ...S>
    instanced_triangles_renderer(std::vector<std::array<float, 3>> vertices_, std::vector<unsigned> indices_, S... program_texts):
        vertices(vertices_),
        indices(indices_)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(*vertices.data()), vertices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);

        GLuint index_buffer;
        glGenBuffers(1, &index_buffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(*indices.data()), indices.data(), GL_STATIC_DRAW);

        std::string source_vertex = R"foo(
in vec3 vertex;
out vec4 vertex_position;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = projection * view * vec4(vertex, 1.0f);
    vertex_position = projection * view * vec4(vertex, 1.0f);
}
)foo";
        std::string source_fragment = R"foo(
in vec4 vertex_position;
in vec4 gl_FragCoord;
in int gl_PrimitiveID;
out vec4 colour;

void main() {
    if (frame % (60 * 10) <= 60 * 5) {
        colour = vec4(1) * int(gl_PrimitiveID < frame * 100);
    } else {
        colour = vec4(1) * float(int(vertex_position.y - float(frame) * 0.1) % 16 == 0);
    }
}
)foo";

        program_vertex = create_program(GL_VERTEX_SHADER, program_texts..., source_vertex);
        program_fragment = create_program(GL_FRAGMENT_SHADER, program_texts..., source_fragment);

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
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
};
