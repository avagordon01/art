layout(std140) uniform inputs {
    vec2 mouse;
    vec2 resolution;
    uint frame;
};
layout(binding=0) uniform usampler2DArray font_atlas;
