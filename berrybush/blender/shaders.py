# standard imports
import pathlib
# 3rd party imports
import bpy
import bgl # this is deprecated, but has a lot of functionality that gpu still lacks
import gpu
import numpy as np
# internal imports
from .common import enumVal
from .shaderstruct import (
    ShaderBool, ShaderInt, ShaderFloat, ShaderVec, ShaderArr, ShaderMat, ShaderStruct
)
from .material import ColorRegSettings, IndTransform, LightChannelSettings
from .texture import TexSettings, TextureTransform
from .tev import TevStageSettings
from ..wii import gx, transform as tf


# CURRENT SHADER APPROACH (ubershader vs dynamic):
# one vertex shader & fragment shader compiled for everything, taking material info through a ubo
# this has good info & links about this issue:
# https://community.khronos.org/t/ubershader-and-branching-cost/108571


TEX_WRAPS = {
    'CLAMP': bgl.GL_CLAMP_TO_EDGE,
    'REPEAT': bgl.GL_REPEAT,
    'MIRROR': bgl.GL_MIRRORED_REPEAT
}


TEX_FILTERS = {
    'NEAREST': bgl.GL_NEAREST,
    'LINEAR': bgl.GL_LINEAR,
    'NEAREST_MIPMAP_NEAREST': bgl.GL_NEAREST_MIPMAP_NEAREST,
    'LINEAR_MIPMAP_NEAREST': bgl.GL_LINEAR_MIPMAP_NEAREST,
    'NEAREST_MIPMAP_LINEAR': bgl.GL_NEAREST_MIPMAP_LINEAR,
    'LINEAR_MIPMAP_LINEAR': bgl.GL_LINEAR_MIPMAP_LINEAR,
}


class ShaderTevStageSels(ShaderStruct):
    tex = ShaderInt
    texSwap = ShaderInt
    ras = ShaderInt
    rasSwap = ShaderInt


class ShaderTevStageIndSettings(ShaderStruct):
    texIdx = ShaderInt
    fmt = ShaderInt
    bias = ShaderVec(ShaderInt, 3)
    bumpAlphaComp = ShaderInt
    mtxType = ShaderInt
    mtxIdx = ShaderInt
    wrap = ShaderVec(ShaderInt, 2)
    utcLOD = ShaderBool
    addPrev = ShaderBool


class ShaderTevStageCalcParams(ShaderStruct):
    constSel = ShaderInt
    args = ShaderVec(ShaderInt, 4)
    compMode = ShaderBool
    op = ShaderInt
    scale = ShaderFloat
    bias = ShaderFloat
    compChan = ShaderInt
    clamp = ShaderBool
    outputIdx = ShaderInt


class ShaderTevStage(ShaderStruct):
    sels = ShaderTevStageSels
    ind = ShaderTevStageIndSettings
    colorParams = ShaderTevStageCalcParams
    alphaParams = ShaderTevStageCalcParams

    @classmethod
    def fromStageSettings(cls, stage: TevStageSettings):
        rStage = ShaderTevStage()
        # selections
        sels = stage.sels
        rSels = rStage.sels
        rSels.tex = sels.texSlot - 1
        rSels.texSwap = sels.texSwapSlot - 1
        rSels.ras = enumVal(sels, "rasterSel", callback=type(sels).rasterSelItems)
        rSels.rasSwap = sels.rasSwapSlot - 1
        # indirect settings
        ind = stage.indSettings
        rInd = rStage.ind
        rInd.texIdx = ind.slot - 1
        rInd.fmt = int(ind.fmt[-1])
        rInd.bias = tuple((-128 if rInd.fmt == 8 else 1) if b else 0 for b in ind.enableBias)
        rInd.bumpAlphaComp = enumVal(ind, "bumpAlphaComp")
        rInd.mtxType = enumVal(ind, "mtxType")
        rInd.mtxIdx = ind.mtxSlot - 1 if ind.enable else -1
        rInd.wrap = tuple(-1 if w == 'OFF' else int(w[3:]) for w in (ind.wrapU, ind.wrapV))
        rInd.utcLOD = ind.utcLOD
        rInd.addPrev = ind.addPrev
        # color & alpha params
        rStage.colorParams.constSel = gx.TEVConstSel[stage.sels.constColor].value
        rStage.alphaParams.constSel = gx.TEVConstSel[stage.sels.constAlpha].value
        rCalcParams = (rStage.colorParams, rStage.alphaParams)
        calcParams = (stage.colorParams, stage.alphaParams)
        for rParams, params in zip(rCalcParams, calcParams):
            argItemsCallback = type(params).argItems
            rParams.args = tuple(enumVal(params, arg, callback=argItemsCallback) for arg in "abcd")
            rParams.compMode = params.compMode
            if rParams.compMode:
                rParams.op = enumVal(params, "compOp")
                rParams.compChan = enumVal(params, "compChan")
            else:
                rParams.op = enumVal(params, "op")
                rParams.scale = 2 ** (enumVal(params, "scale") - 1)
                rParams.bias = (0, .5, -.5)[enumVal(params, "bias")]
            rParams.outputIdx = enumVal(params, "output")
            rParams.clamp = params.clamp
        return rStage


class ShaderTexture(ShaderStruct):
    mtx = ShaderMat(ShaderFloat, 2, 3)
    dims = ShaderVec(ShaderInt, 2)
    mapMode = ShaderInt
    hasImg = ShaderBool

    def __init__(self):
        super().__init__()
        self.wrap: tuple[int, int] = ()
        self.filter: tuple[int, int] = ()
        self.lodBias: float = 0
        self.imgName: str = ""
        self.imgSlot: int = 0
        self.transform = tf.Transformation(2)
        self.mtx = ((1, 0, 0), (0, 1, 0))
        self._s = (1, 1)
        self._r = 0
        self._t = (0, 0)

    def setMtx(self, texTf: TextureTransform, tfGen: tf.MtxGenerator):
        """Set this texture's transformation matrix."""
        # TODO: implement similar caching for indirect matrices
        s = texTf.scale.to_tuple()
        r = texTf.rotation
        t = texTf.translation.to_tuple()
        if s != self._s or r != self._r or t != self._t:
            stf = self.transform
            stf.set(s, np.rad2deg((r, )), t)
            self.mtx = tuple(tuple(v) for v in tfGen.genMtx(stf)[:2])
            self._s = s
            self._r = r
            self._t = t

    @classmethod
    def fromTexSettings(cls, tex: TexSettings, tfGen: tf.MtxGenerator):
        rTex = cls()
        # image
        img: bpy.types.Image = tex.activeImg
        rTex.hasImg = img is not None
        rTex.imgName = img.name if rTex.hasImg else ""
        rTex.imgSlot = tex.activeImgSlot
        # transform
        rTex.setMtx(tex.transform, tfGen)
        # settings
        rTex.dims = tuple(img.size) if rTex.hasImg else (0, 0)
        rTex.mapMode = enumVal(tex, "mapMode", callback=type(tex).coordSrcItems)
        rTex.wrap = (TEX_WRAPS[tex.wrapModeU], TEX_WRAPS[tex.wrapModeV])
        mmLevels = len(img.brres.mipmaps) if rTex.hasImg else 0
        minFilter = f'{tex.minFilter}_MIPMAP_{tex.mipFilter}' if mmLevels > 0 else tex.minFilter
        rTex.filter = (TEX_FILTERS[minFilter], TEX_FILTERS[tex.magFilter])
        rTex.lodBias = tex.lodBias
        return rTex


class ShaderIndTex(ShaderStruct):
    texIdx = ShaderInt
    mode = ShaderInt
    lightIdx = ShaderInt
    coordScale = ShaderVec(ShaderInt, 2)


class ShaderLightChanSettings(ShaderStruct):
    difFromReg = ShaderBool
    ambFromReg = ShaderBool
    difMode = ShaderInt
    atnMode = ShaderInt
    enabledLights = ShaderArr(ShaderBool, 8)


class ShaderLightChan(ShaderStruct):
    difReg = ShaderVec(ShaderFloat, 4)
    ambReg = ShaderVec(ShaderFloat, 4)
    colorSettings = ShaderLightChanSettings
    alphaSettings = ShaderLightChanSettings

    def setRegs(self, lc: LightChannelSettings):
        """Update this light channel's diffuse/ambient registers from BRRES settings."""
        self.difReg = tuple(lc.difColor)
        self.ambReg = tuple(lc.ambColor)

    @classmethod
    def fromLightChanSettings(cls, lc: LightChannelSettings):
        rlc = ShaderLightChan()
        rlc.setRegs(lc)
        rlcCA = (rlc.colorSettings, rlc.alphaSettings)
        lcCA = (lc.colorSettings, lc.alphaSettings)
        for rOps, ops in zip(rlcCA, lcCA): # (options for color/alpha)
            rOps.difFromReg = ops.difFromReg
            rOps.ambFromReg = ops.ambFromReg
            rOps.difMode = enumVal(ops, "diffuseMode") if ops.enableDiffuse else -1
            rOps.atnMode = enumVal(ops, "attenuationMode") if ops.enableAttenuation else -1
            rOps.enabledLights = tuple(ops.enabledLights)
        return rlc


class ShaderMaterial(ShaderStruct):
    colorSwaps = ShaderArr(ShaderVec(ShaderInt, 4), gx.MAX_COLOR_SWAPS)
    stages = ShaderArr(ShaderTevStage, gx.MAX_TEV_STAGES)
    textures = ShaderArr(ShaderTexture, gx.MAX_TEXTURES)
    inds = ShaderArr(ShaderIndTex, gx.MAX_INDIRECTS)
    indMtcs = ShaderArr(ShaderMat(ShaderFloat, 2, 3), gx.MAX_INDIRECT_MTCS)
    constColors = ShaderArr(ShaderVec(ShaderFloat, 4), gx.MAX_TEV_CONST_COLORS)
    outputColors = ShaderArr(ShaderVec(ShaderFloat, 4), gx.MAX_TEV_STAND_COLORS + 1)
    lightChans = ShaderArr(ShaderLightChan, gx.MAX_CLR_ATTRS)
    enableBlend = ShaderBool
    alphaTestVals = ShaderVec(ShaderFloat, 2)
    alphaTestComps = ShaderVec(ShaderInt, 2)
    alphaTestLogic = ShaderInt
    alphaTestEnable = ShaderBool
    constAlpha = ShaderFloat
    numStages = ShaderInt
    numTextures = ShaderInt
    numIndMtcs = ShaderInt

    blendSubtract: bool = False
    blendSrcFac: int = 0
    blendDstFac: int = 0
    enableBlendLogic: bool = False
    blendLogicOp: int = 0
    enableDither: bool = False
    blendUpdateColorBuffer: bool = False
    blendUpdateAlphaBuffer: bool = False
    enableConstAlpha: bool = False
    enableCulling: bool = False
    cullMode: int = 0
    isXlu: bool = False

    enableDepthTest: bool = False
    depthFunc: int = 0
    enableDepthUpdate: bool = False

    name: str = ""

    def setColorRegs(self, regs: ColorRegSettings):
        """Set this material's color registers from BRRES settings."""
        self.constColors = tuple(tuple(c) for c in regs.constant)
        self.outputColors = tuple(tuple(c) for c in regs.standard)

    def setIndMtcs(self, tfs: list[IndTransform], tfGen: tf.MtxGenerator):
        """Set this material's indirect matrices from BRRES settings."""
        self.indMtcs = ()
        for itf in tfs:
            itf = itf.transform
            s, r, t = itf.scale, [np.rad2deg(itf.rotation)], itf.translation
            mtx = tf.IndMtxGen2D.genMtx(tf.Transformation(2, s, r, t))[:2]
            self.indMtcs += (tuple(tuple(v) for v in mtx), )
        self.numIndMtcs = len(tfs)


class ShaderMesh(ShaderStruct):
    colors = ShaderArr(ShaderInt, gx.MAX_CLR_ATTRS)
    uvs = ShaderArr(ShaderInt, gx.MAX_UV_ATTRS)


RENDER_STRUCTS = (
    ShaderTevStageSels,
    ShaderTevStageIndSettings,
    ShaderTevStageCalcParams,
    ShaderTevStage,
    ShaderTexture,
    ShaderIndTex,
    ShaderLightChanSettings,
    ShaderLightChan,
    ShaderMaterial,
    ShaderMesh
)

def compileBrresShader() -> gpu.types.GPUShader:
    """Compile & return the main BRRES shader."""
    shaderInfo = gpu.types.GPUShaderCreateInfo()
    # uniforms
    shaderInfo.typedef_source("".join(s.getSource() for s in RENDER_STRUCTS))
    shaderInfo.uniform_buf(1, ShaderMaterial.getName(), "material")
    shaderInfo.uniform_buf(2, ShaderMesh.getName(), "mesh")
    shaderInfo.push_constant('MAT4', "modelViewProjectionMtx")
    shaderInfo.push_constant('MAT3', "normalMtx")
    shaderInfo.push_constant('BOOL', "forceOpaque")
    shaderInfo.push_constant('BOOL', "isConstAlphaWrite")
    for i in range(gx.MAX_TEXTURES):
        shaderInfo.sampler(i, 'FLOAT_2D', f"image{i}")
    # vertex inputs
    shaderInfo.vertex_in(0, 'VEC3', "position")
    shaderInfo.vertex_in(1, 'VEC3', "normal")
    for clr in range(gx.MAX_CLR_ATTRS):
        shaderInfo.vertex_in(2 + clr, 'VEC4', f"color{clr}")
    for uv in range(gx.MAX_UV_ATTRS):
        shaderInfo.vertex_in(4 + uv, 'VEC2', f"uv{uv}")
    # interfaces (vertex outputs/fragment inputs)
    interfaceInfo = gpu.types.GPUStageInterfaceInfo("shader_interface")
    interfaceInfo.smooth('VEC4', "clipSpace")
    interfaceInfo.smooth('VEC3', "fragPosition")
    interfaceInfo.smooth('VEC3', "fragNormal")
    for clr in range(gx.MAX_CLR_ATTRS):
        interfaceInfo.smooth('VEC4', f"fragColor{clr}")
    for uv in range(gx.MAX_UV_ATTRS):
        interfaceInfo.smooth('VEC2', f"fragUV{uv}")
    shaderInfo.vertex_out(interfaceInfo)
    # fragment outputs
    shaderInfo.fragment_out(0, 'VEC4', "fragOutput")
    # compile
    shaderPath = (pathlib.Path(__file__).parent / "shaders").resolve()
    with open(shaderPath / "vertex.glsl", "r", encoding="utf-8") as f:
        shaderInfo.vertex_source(f.read())
    with open(shaderPath / "fragment.glsl", "r", encoding="utf-8") as f:
        shaderInfo.fragment_source(f.read())
    return gpu.shader.create_from_info(shaderInfo)


BRRES_SHADER = compileBrresShader()


def compilePostprocessShader() -> gpu.types.GPUShader:
    """Compile & return the post-processing shader."""
    shaderInfo = gpu.types.GPUShaderCreateInfo()
    shaderInfo.sampler(0, 'FLOAT_2D', "tex")
    shaderInfo.push_constant('BOOL', "doAlpha")
    shaderInfo.vertex_in(0, 'VEC2', "position")
    interfaceInfo = gpu.types.GPUStageInterfaceInfo("shader_interface")
    interfaceInfo.smooth('VEC2', "fragPosition")
    shaderInfo.vertex_out(interfaceInfo)
    shaderInfo.fragment_out(0, 'VEC4', "fragOutput")
    shaderPath = (pathlib.Path(__file__).parent / "shaders" / "postprocess").resolve()
    with open(shaderPath / "vertex.glsl", "r", encoding="utf-8") as f:
        shaderInfo.vertex_source(f.read())
    with open(shaderPath / "fragment.glsl", "r", encoding="utf-8") as f:
        shaderInfo.fragment_source(f.read())
    return gpu.shader.create_from_info(shaderInfo)


POSTPROCESS_SHADER = compilePostprocessShader()
