#include "layer.hh"
#include "util.hh"
#include <string>

struct juliaset : layer_t {
    GLuint program_vertex, program_fragment, pipeline_render, vao;
    juliaset(std::string shared_uniforms) {
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
layout(origin_upper_left) in vec4 gl_FragCoord;
out vec4 colour;

vec2 cmul(in vec2 a, in vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

vec3 colormap(float x) {
    float r = clamp((1.0 + 1.0 / 63.0) * x - 1.0 / 63.0, 0.0, 1.0);
    float g = clamp(-(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0), 0.0, 1.0);
    float b = 1.0;
    return vec3(r, g, b);
}

vec3 julia(vec2 z, vec2 c) {
    int i;
    int n = 512;
    for (i = 0; i < n; i++) {
        z = cmul(z, z) + c;
        if (dot(z, z) > 4.0f)
            break;
    }
    if (i == n) {
        return vec3(0.0);
    } else {
        return colormap(sqrt(float(i) / float(n)));
    }
}

void main() {
    vec2 t = (gl_FragCoord.xy - (resolution.xy * vec2(0.5f))) / resolution.xx;
    t = t * vec2(4);

    vec2 m = (mouse.xy / resolution.xy) - vec2(0.5, 0.5);
    m = (m - vec2(0.25, 0.0)) * vec2(2);

    colour.rgb = julia(t, m);
    colour.a = 1.0f;
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
