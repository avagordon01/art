#include <string>
#include "engine/drawable.hh"

struct juliaset {
    drawable<> drawable;
    juliaset(std::string shared_uniforms):
        drawable(
        drawable.quad.vertex_shader,
        shared_uniforms + R"foo(
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
)foo")
    {}
    void draw() {
        drawable.draw(GL_QUADS);
    }
};
