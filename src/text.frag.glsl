layout(origin_upper_left) in vec4 gl_FragCoord;
out vec3 colour;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy) / textureSize(font_atlas, 0).xy;
    ivec3 texcoord = ivec3(gl_FragCoord.xy, (fc.x) % textureSize(font_atlas, 0).z);
    texcoord = texcoord % textureSize(font_atlas, 0);

    colour = texelFetch(font_atlas, texcoord, 0).rgb;
}
