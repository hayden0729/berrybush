void main() {
    fragOutput = texture(tex, fragPosition / 2 + .5);
    if (doAlpha) {
        // convert to straight alpha before gamma correction
        // (since transparent rendering is done on top of a black background,
        // it's essentially premultiplied up to this point)
        fragOutput.rgb /= fragOutput.a;
    }
    else {
        fragOutput.a = 1.0;
    }
    // convert to linear color space, expected by blender
    fragOutput = pow(fragOutput, vec4(2.2, 2.2, 2.2, 1.0));
    if (doAlpha) {
        // convert back to premultiplied alpha, expected by blender
        fragOutput.rgb *= fragOutput.a;
    }
}
