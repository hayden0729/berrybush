void main() {
    clipSpace = modelViewProjectionMtx * vec4(position, 1.0);
    gl_Position = clipSpace;
    fragPosition = position;
    fragNormal = normal;
    fragColor0 = color0;
    fragColor1 = color1;
    fragUV0 = uv0;
    fragUV1 = uv1;
    fragUV2 = uv2;
    fragUV3 = uv3;
    fragUV4 = uv4;
    fragUV5 = uv5;
    fragUV6 = uv6;
    fragUV7 = uv7;
}
