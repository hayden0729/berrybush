vec4 fragColor[2];
vec2 fragUV[8];

void initializeAttrArrs() {
    fragColor = vec4[](fragColor0, fragColor1);
    fragUV = vec2[](fragUV0, fragUV1, fragUV2, fragUV3, fragUV4, fragUV5, fragUV6, fragUV7);
}

vec4 swapColors(vec4 color, int swapIdx) {
    // run a color through through the color swap at swapIdx (0-3) in the swap table
    uvec4 swap = material.colorSwaps[swapIdx];
    vec4 swappedColor = color;
    for (int chan = 0; chan < 4; chan++) {
        swappedColor[chan] = color[swap[chan]];
    }
    return swappedColor;
}

vec3 getConstColor(int sel) {
    // get constant color from a const sel enum
    if (sel < 8) { return vec3(1.0 - float(sel) / 8.0); }
    else if (sel < 16) { return material.constColors[sel - 12].rgb; }
    else if (sel < 20) { return material.constColors[sel - 16].rrr; }
    else if (sel < 24) { return material.constColors[sel - 20].ggg; }
    else if (sel < 28) { return material.constColors[sel - 24].bbb; }
    else if (sel < 32) { return material.constColors[sel - 28].aaa; }
}

vec4 getLightChanColor(int chanIdx) {
    // get color output for a light channel
    GLSLLightChan lc = material.lightChans[chanIdx];
    vec4 color;
    int constVertColor = mesh.colors[chanIdx];
    vec4 vertColor = (constVertColor == -1) ? fragColor[chanIdx] : vec4(constVertColor);
    color.rgb = (lc.colorSettings.difFromReg) ? lc.difReg.rgb : vertColor.rgb;
    if (lc.colorSettings.difMode >= 0) {
        vec3 ambColor = (lc.colorSettings.ambFromReg) ? lc.ambReg.rgb : vertColor.rgb;
        color.rgb *= ambColor;
    }
    // alpha
    color.a = (lc.alphaSettings.difFromReg) ? lc.difReg.a : vertColor.a;
    if (lc.alphaSettings.difMode >= 0) {
        float ambAlpha = (lc.alphaSettings.ambFromReg) ? lc.ambReg.a : vertColor.a;
        color.a *= ambAlpha;
    }
    return color;
}

vec4 getRasColor(GLSLTevStage stage) {
    // get raster color for a stage
    int sel = stage.sels.ras;
    vec4 rasterColor = vec4(0.0);
    if (sel < 2) { // light channel output
        rasterColor = getLightChanColor(sel);
    }
    else if (sel < 4) {
        // TODO: bump alpha
    }
    return swapColors(rasterColor, stage.sels.rasSwap);
}

vec2 getTexCoord(GLSLTexture tex) {
    // get the coordinate to sample for a texture
    vec3 coord = vec3(0.0);
    int mapMode = tex.mapMode;
    switch (mapMode) {
        case 0: // uvs
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            int constUV = mesh.uvs[tex.mapMode];
            coord = vec3((constUV == -1) ? fragUV[tex.mapMode] : vec2(constUV), 1);
            break;
        case 8: coord.st = (clipSpace.xy / clipSpace.w / 2 + .5); break; // projection mapping
        case 9: // environment mapping
        case 10:
        case 11: coord = fragNormal; break;
        case 12: // colors
        case 13: coord = getLightChanColor(tex.mapMode - 12).stp; break;
    }
    if (8 <= tex.mapMode && tex.mapMode < 12) { // TODO: env light & env spec
        if (9 <= tex.mapMode) {
            coord = normalize(normalMtx * coord) / 2 + .5;
        }
        coord.t = 1 - coord.t; // switch to brres convention before texture matrix multiplication
    }
    coord.st = vec3(coord.st, 1) * tex.mtx;
    coord.t = 1 - coord.t; // switch to blender convention
    return coord.st;
}

vec4 sampleTexture(int texIdx, vec2 coord) {
    // sad function to sample a texture based on a coord & texture index
    // wish i could have an array of samplers and just use texture(image[texIdx], coord),
    // but blender's gpu module doesn't seem to support this
    // (supports array textures, but not sampler arrays)
    switch (texIdx) {
        case 0: return texture(image0, coord);
        case 1: return texture(image1, coord);
        case 2: return texture(image2, coord);
        case 3: return texture(image3, coord);
        case 4: return texture(image4, coord);
        case 5: return texture(image5, coord);
        case 6: return texture(image6, coord);
        case 7: return texture(image7, coord);
    }
}

vec4 sampleStageTexture(GLSLTevStage stage) {
    // sample a stage's texture w/ color swap applied
    int texIdx = stage.sels.tex;
    vec4 texColor = vec4(0.0);
    if (texIdx < material.numTextures) {
        GLSLTexture tex = material.textures[texIdx];
        if (tex.hasImg) {
            vec2 texCoord = getTexCoord(tex);
            if (0 <= stage.ind.mtxIdx && stage.ind.mtxIdx < material.numIndMtcs) {
                // apply indirect texturing, as long as a valid indirect texture is selected
                GLSLIndTex ind = material.inds[stage.ind.texIdx];
                if (ind.texIdx < material.numTextures) {
                    int mode = ind.mode;
                    switch (mode) {
                        case 0: // warp
                            vec2 indTexCoord = getTexCoord(material.textures[ind.texIdx]) / ind.coordScale;
                            vec3 indColor = sampleTexture(ind.texIdx, indTexCoord).abg * 255 + stage.ind.bias;
                            vec2 indOffset = indColor * material.indMtcs[stage.ind.mtxIdx] / tex.dims;
                            indOffset.y *= -1; // switch from wii to blender convention (y *= -1 rather than y = 1 - y bc it's an offset)
                            texCoord += indOffset.st;
                            break;
                        case 1: break; // TODO: normal mapping
                        case 2: break; // TODO: normal mapping (specular)
                        case 3: // custom
                        case 4: break;
                    }
                }
            }
            texColor = sampleTexture(texIdx, texCoord);
        }
    }
    return swapColors(texColor, stage.sels.texSwap);
}


vec4 getTevOutput() {
    // calculate the main output of the tev stages
    mat4 outputColors = material.outputColors;
    for (int stageIdx = 0; stageIdx < material.numStages; stageIdx += 1) {
        GLSLTevStage stage = material.stages[stageIdx];
        mat4x3 colorArgs;
        for (int calcIdx = 0; calcIdx < 2; calcIdx += 1) { // one calculation for color, one for alpha
            bool isAlpha = (calcIdx == 1);
            GLSLTevStageCalcParams params = (isAlpha) ? stage.alphaParams : stage.colorParams;
            mat4x3 args;
            for (int argIdx = 0; argIdx < 4; argIdx++) {
                int arg = params.args[argIdx];
                if (isAlpha) {
                    switch (arg) {
                        case 0: args[argIdx] = outputColors[0].aaa; break;
                        case 1: args[argIdx] = outputColors[1].aaa; break;
                        case 2: args[argIdx] = outputColors[2].aaa; break;
                        case 3: args[argIdx] = outputColors[3].aaa; break;
                        case 4: args[argIdx] = sampleStageTexture(stage).aaa; break;
                        case 5: args[argIdx] = getRasColor(stage).aaa; break;
                        case 6: args[argIdx] = getConstColor(params.constSel); break;
                        case 7: args[argIdx] = vec3(0.0); break;
                    }
                }
                else {
                    switch (arg) {
                        case 0: args[argIdx] = outputColors[0].rgb; break;
                        case 1: args[argIdx] = outputColors[0].aaa; break;
                        case 2: args[argIdx] = outputColors[1].rgb; break;
                        case 3: args[argIdx] = outputColors[1].aaa; break;
                        case 4: args[argIdx] = outputColors[2].rgb; break;
                        case 5: args[argIdx] = outputColors[2].aaa; break;
                        case 6: args[argIdx] = outputColors[3].rgb; break;
                        case 7: args[argIdx] = outputColors[3].aaa; break;
                        case 8: args[argIdx] = sampleStageTexture(stage).rgb; break;
                        case 9: args[argIdx] = sampleStageTexture(stage).aaa; break;
                        case 10: args[argIdx] = getRasColor(stage).rgb; break;
                        case 11: args[argIdx] = getRasColor(stage).aaa; break;
                        case 12: args[argIdx] = vec3(1.0); break;
                        case 13: args[argIdx] = vec3(0.5); break;
                        case 14: args[argIdx] = getConstColor(params.constSel); break;
                        case 15: args[argIdx] = vec3(0.0); break;
                    }
                }
            }
            if (!isAlpha) {
                // color args are stored across color and alpha calculation since they may be used for alpha comparison if enabled
                colorArgs = args;
            }
            vec3 calcOutput;
            if (params.compMode) { // compare mode
                calcOutput = args[3];
                if (params.compChan < 3) {
                    // compare combined color bits
                    vec2 compArgs;
                    for (int i = 0; i < 2; i++) {
                        int compChan = params.compChan;
                        switch (compChan) {
                            case 0: compArgs[i] = colorArgs[i].r; break;
                            case 1: compArgs[i] = colorArgs[i].r + colorArgs[i].g * 255; break;
                            case 2: compArgs[i] = colorArgs[i].r + colorArgs[i].g * 255 + colorArgs[i].b * 255 * 255; break;
                        }
                    }
                    if ((params.op == 0 && compArgs[0] > compArgs[1]) || (params.op == 1 && compArgs[0] == compArgs[1])) {
                        calcOutput += args[2];
                    }
                }
                else {
                    // compare by channel
                    if (params.op == 0) {
                        if (args[0].r > args[1].r) { calcOutput.r += args[2].r; }
                        if (args[0].g > args[1].g) { calcOutput.g += args[2].g; }
                        if (args[0].b > args[1].b) { calcOutput.b += args[2].b; }
                    }
                    else {
                        if (round(args[0].r * 255) == round(args[1].r * 255)) { calcOutput.r += args[2].r; }
                        if (round(args[0].g * 255) == round(args[1].g * 255)) { calcOutput.g += args[2].g; }
                        if (round(args[0].b * 255) == round(args[1].b * 255)) { calcOutput.b += args[2].b; }
                    }
                }
            }
            else { // standard calcluation
                calcOutput = mix(args[0], args[1], args[2]);
                if (params.op == 1) { // operation is subtract (otherwise add)
                    calcOutput *= -1;
                }
                calcOutput = params.scale * (params.bias + args[3] + calcOutput);
            }
            if (params.clamp) {
                calcOutput = clamp(calcOutput, vec3(0.0), vec3(1.0));
            }
            if (isAlpha) {
                outputColors[params.outputIdx] = vec4(outputColors[params.outputIdx].rgb, calcOutput[0]);
            }
            else {
                outputColors[params.outputIdx] = vec4(calcOutput, outputColors[params.outputIdx].a);
            }
        }
    }
    // return outputColors[0];
    GLSLTevStage lastStage = material.stages[material.numStages - 1];
    return vec4(outputColors[lastStage.colorParams.outputIdx].rgb, outputColors[lastStage.alphaParams.outputIdx].a);
}


bool alphaTest(float val) {
    // perform the alpha test for some alpha value (if false, should be discarded)
    bvec2 alphaTestsPassed = bvec2(false, false);
    for (int i = 0; i < 2; i++) {
        int compMode = material.alphaTestComps[i];
        switch (compMode) {
            case 0: alphaTestsPassed[i] = false; break;
            case 1: alphaTestsPassed[i] = (val < material.alphaTestVals[i]); break;
            case 2: alphaTestsPassed[i] = (val == material.alphaTestVals[i]); break;
            case 3: alphaTestsPassed[i] = (val <= material.alphaTestVals[i]); break;
            case 4: alphaTestsPassed[i] = (val > material.alphaTestVals[i]); break;
            case 5: alphaTestsPassed[i] = (val != material.alphaTestVals[i]); break;
            case 6: alphaTestsPassed[i] = (val >= material.alphaTestVals[i]); break;
            case 7: alphaTestsPassed[i] = true; break;
        }
    }
    int logicMode = material.alphaTestLogic;
    switch (logicMode) {
        case 0: return (alphaTestsPassed[0] && alphaTestsPassed[1]);
        case 1: return (alphaTestsPassed[0] || alphaTestsPassed[1]);
        case 2: return (alphaTestsPassed[0] ^^ alphaTestsPassed[1]);
        case 3: return !(alphaTestsPassed[0] ^^ alphaTestsPassed[1]);
    }
}


void main() {
    if (isConstAlphaWrite) {
        fragOutput.a = material.constAlpha;
    }
    else {
        initializeAttrArrs();
        fragOutput = getTevOutput();
        // alpha test
        // note: alpha test can't actually be disabled in brres materials
        // enable check just exists so that everything isn't discarded when ubo is empty (i.e., an object has no material set)
        if (material.alphaTestEnable && !alphaTest(fragOutput.a)) {
            discard;
        }
        // little hack for some spots in blender where alpha gets weird (one of the many material preview issues)
        if (forceOpaque) {
            fragOutput.a = 1.0;
        }
        // else {
        //     fragOutput.rgb = fragNormal / 2 + .5;
        // }
    }
}
