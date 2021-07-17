#include "layer.hh"
#include "util.hh"
#include <string>

struct text_overlay : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    template<typename ...S>
    text_overlay(S... program_texts) {
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
        std::string source_fragment = R"foo(
layout(origin_upper_left) in vec4 gl_FragCoord;
out vec4 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy) / textureSize(font_atlas, 0).xy;
    ivec3 texcoord = ivec3(gl_FragCoord.xy, (fc.x + frame) % textureSize(font_atlas, 0).z);
    texcoord = texcoord % textureSize(font_atlas, 0);

    colour = texelFetch(font_atlas, texcoord, 0);
    colour.a = 0.25f;
}
)foo";

        program_vertex = create_program(GL_VERTEX_SHADER, source_vertex);
        program_fragment = create_program(GL_FRAGMENT_SHADER, program_texts..., source_fragment);

        glGenProgramPipelines(1, &pipeline_render);
        glUseProgramStages(pipeline_render, GL_VERTEX_SHADER_BIT, program_vertex);
        glUseProgramStages(pipeline_render, GL_FRAGMENT_SHADER_BIT, program_fragment);
        glBindProgramPipeline(pipeline_render);

        glBindVertexArray(0);
    }
    void draw() override {
        glBindVertexArray(vao);
        glBindProgramPipeline(pipeline_render);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};
