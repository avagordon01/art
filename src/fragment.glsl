#version 450

layout(std140) uniform inputs {
    vec2 mouse;
    vec2 resolution;
    uint frame;
};
layout(binding=0) uniform usampler2DArray font_atlas;

layout(origin_upper_left) in vec4 gl_FragCoord;
out vec3 colour;

vec2 cmul(in vec2 a, in vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

vec2 csqr(in vec2 a) {
    return cmul(a, a);
}

float colormap_red(float x) {
    return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
}

float colormap_green(float x) {
    return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
}

vec3 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = 1.0;
    return vec3(r, g, b);
}

float julia(vec2 z, vec2 c) {
    int i;
    int n = 512;
    for (i = 0; i < n; i++) {
        z = csqr(z) + c;
        if (dot(z, z) > 4.0f)
            break;
    }
    if (i == n) {
        return 0.0;
    } else {
        return sqrt(float(i) / float(n));
    }
}

void main() {
    vec2 t = (gl_FragCoord.xy - (resolution.xy * vec2(0.5f))) / resolution.xx;
    t = t * vec2(4);

    vec2 m = (mouse.xy / resolution.xy) - vec2(0.5, 0.5);
    m = (m - vec2(0.25, 0.0)) * vec2(2);

    float colour_julia = julia(t, m);

    ivec2 fc = ivec2(gl_FragCoord.xy) / textureSize(font_atlas, 0).xy;
    ivec3 texcoord = ivec3(gl_FragCoord.xy, (fc.x) % textureSize(font_atlas, 0).z);
    texcoord = texcoord % textureSize(font_atlas, 0);
    float colour_text = texelFetch(font_atlas, texcoord, 0).a;

    //colour = vec3(colour_text);
    colour = colormap(colour_julia);
}
