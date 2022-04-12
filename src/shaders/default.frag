#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler tex_sampler;
layout(set = 0, binding = 2) uniform Uniforms {
    float width;
    float height;
};

void main() {
    vec3 color = texture(sampler2D(tex, tex_sampler), tex_coords).rgb;

    frag_color = vec4(color, 1.0);
}
