#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;
uniform float x_factor;
uniform float x_radius;
uniform float y_factor;
uniform float y_radius;
uniform bool enableBF;

float compareColor(vec4 col1, vec4 col2, float factor)
{
    float l1 = col1.x * 0.2125 + col1.y * 0.7154 + col1.z * 0.0721;
    float l2 = col2.x * 0.2125 + col2.y * 0.7154 + col2.z * 0.0721;
    return smoothstep(factor, 1.0f, 1.0f - abs(l1 - l2));
}

void main()
{
    if(!enableBF)
    {
        vec4 res = vec4(texture(screenTexture, TexCoords).rgb, 1.0f);
        FragColor = vec4(pow(res.x, 1.0f / 2.2f), pow(res.y, 1.0f / 2.2f), pow(res.z, 1.0f / 2.2f), 1.0);
    }
    else
    {
        vec2 delta = vec2(0.001f, 0.001f) * vec2(0, y_radius);
        vec2 delta2 = vec2(0.001f, 0.001f) * vec2(x_radius, 0);
        vec4 col = texture(screenTexture, TexCoords);
        vec4 col0a = texture(screenTexture, TexCoords - delta);
        vec4 col0b = texture(screenTexture, TexCoords + delta);
        vec4 col1a = texture(screenTexture, TexCoords - 2.0f * delta);
        vec4 col1b = texture(screenTexture, TexCoords + 2.0f * delta);
        vec4 col2a = texture(screenTexture, TexCoords - 3.0f * delta);
        vec4 col2b = texture(screenTexture, TexCoords + 3.0f * delta);

        vec4 col2 = texture(screenTexture, TexCoords);
        vec4 col20a = texture(screenTexture, TexCoords - delta2);
        vec4 col20b = texture(screenTexture, TexCoords + delta2);
        vec4 col21a = texture(screenTexture, TexCoords - 2.0f * delta2);
        vec4 col21b = texture(screenTexture, TexCoords + 2.0f * delta2);
        vec4 col22a = texture(screenTexture, TexCoords - 3.0f * delta2);
        vec4 col22b = texture(screenTexture, TexCoords + 3.0f * delta2);

        float w = 0.37004405286;
        float w0a = compareColor(col, col0a, y_factor) * 0.31718061674;
        float w0b = compareColor(col, col0b, y_factor) * 0.31718061674;
        float w1a = compareColor(col, col1a, y_factor) * 0.19823788546;
        float w1b = compareColor(col, col1b, y_factor) * 0.19823788546;
        float w2a = compareColor(col, col2a, y_factor) * 0.11453744493;
        float w2b = compareColor(col, col2b, y_factor) * 0.11453744493;

        float w2 = 0.37004405286;
        float w20a = compareColor(col2, col20a, x_factor) * 0.31718061674;
        float w20b = compareColor(col2, col20b, x_factor) * 0.31718061674;
        float w21a = compareColor(col2, col21a, x_factor) * 0.19823788546;
        float w21b = compareColor(col2, col21b, x_factor) * 0.19823788546;
        float w22a = compareColor(col2, col22a, x_factor) * 0.11453744493;
        float w22b = compareColor(col2, col22b, x_factor) * 0.11453744493;

        vec3 res;
        res = w * col.xyz;
        res += w0a * col0a.xyz;
        res += w0b * col0b.xyz;
        res += w1a * col1a.xyz;
        res += w1b * col1b.xyz;
        res += w2a * col2a.xyz;
        res += w2b * col2b.xyz;

        res += w2 * col2.xyz;
        res += w20a * col20a.xyz;
        res += w20b * col20b.xyz;
        res += w21a * col21a.xyz;
        res += w21b * col21b.xyz;
        res += w22a * col22a.xyz;
        res += w22b * col22b.xyz;

        res /= (w + w0a + w0b + w1a + w1b + w2a + w2b + w2 + w20a + w20b + w21a + w21b + w22a + w22b);

        vec3 colout = texture(screenTexture, TexCoords).rgb;
        FragColor = vec4(pow(res.x, 1.0f / 2.2f), pow(res.y, 1.0f / 2.2f), pow(res.z, 1.0f / 2.2f), 1.0);
    }
}