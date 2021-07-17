#include "layer.hh"
#include "util.hh"
#include <string>

struct grid : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    grid(std::string shared_uniforms) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        std::string source_vertex = R"foo(
const vec2 vertices[] = {
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
};

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};
void main() {
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)foo";

        std::string source_fragment = shared_uniforms + R"foo(
layout(pixel_center_integer) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    //ivec2 fc_new = fc + ivec2(frame) * ivec2(2, 1);

    ivec2 fc_mod = fc % ivec2(64);

    bool x = (fc_mod.x == 0);
    bool y = (fc_mod.y == 0);
    colour = vec4(vec3(1), 0.25) * float(x || y);

    bool x_plus = x && (fc_mod.y < 8 || fc_mod.y > 56);
    bool y_plus = y && (fc_mod.x < 8 || fc_mod.x > 56);
    colour += vec4(1) * float(x_plus || y_plus);
}
)foo";

        program_vertex = create_program(GL_VERTEX_SHADER, source_vertex);
        program_fragment = create_program(GL_FRAGMENT_SHADER, source_fragment);

        glGenProgramPipelines(1, &pipeline_render);
        glUseProgramStages(pipeline_render, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline_render, GL_FRAGMENT_SHADER_BIT, program_fragment);

        glBindVertexArray(0);
    }
    void draw() override {
        glBindVertexArray(vao);
        glBindProgramPipeline(pipeline_render);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};
