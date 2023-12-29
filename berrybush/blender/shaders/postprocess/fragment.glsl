void main() {
    fragOutput = texture(tex, fragPosition / 2 + .5);
    if (doAlpha) {
        if (undoPremul) {
            // undo premultiplied alpha BEFORE gamma correction!
            fragOutput.rgb /= fragOutput.a;
        }
    }
    else {
        fragOutput.a = 1.0;
    }
    // gamma correction
    fragOutput = pow(fragOutput, vec4(2.2, 2.2, 2.2, 1.0));
}
