#pragma once

struct fullscreen_quad {
    std::string vertex_shader = R"foo(
const vec2 vertices[] = {
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
};

out gl_PerVertex {
    vec4 gl_Position;
};
void main() {
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)foo";
    void draw() {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
};
