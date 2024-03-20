# standard imports
from abc import ABC, abstractmethod
import pathlib
from typing import Generic, TypeVar
# 3rd party imports
import bpy
import bgl # this is deprecated, but has a lot of functionality that gpu still lacks
from bpy.types import Image
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
import numpy as np
# internal imports
from .common import ( # pylint: disable=unused-import
    LOG_PATH, PropertyPanel,
    drawColumnSeparator, enumVal, foreachGet,
    getLoopVertIdcs, getLoopFaceIdcs, getFaceMatIdcs, getLayerData
)
from .glslstruct import GLSLBool, GLSLInt, GLSLFloat, GLSLVec, GLSLArr, GLSLMat, GLSLStruct
from .material import ColorRegSettings, IndTransform, LightChannelSettings
from .texture import TexSettings, TextureTransform
from .tev import TevStageSettings
from ..wii import gx, tex0, transform as tf


# CURRENT SHADER APPROACH (ubershader vs dynamic):
# one vertex shader & fragment shader compiled for everything, taking material info through a ubo
# this has good info & links about this issue:
# https://community.khronos.org/t/ubershader-and-branching-cost/108571


TextureManagerT = TypeVar("TextureManagerT", bound="TextureManager")


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


BLEND_FACS = {
    'ZERO': bgl.GL_ZERO,
    'ONE': bgl.GL_ONE,
    'SRC_COLOR': bgl.GL_SRC_COLOR,
    'INV_SRC_COLOR': bgl.GL_ONE_MINUS_SRC_COLOR,
    'DST_COLOR': bgl.GL_DST_COLOR,
    'INV_DST_COLOR': bgl.GL_ONE_MINUS_DST_COLOR,
    'SRC_ALPHA': bgl.GL_SRC_ALPHA,
    'INV_SRC_ALPHA': bgl.GL_ONE_MINUS_SRC_ALPHA,
    'DST_ALPHA': bgl.GL_DST_ALPHA,
    'INV_DST_ALPHA': bgl.GL_ONE_MINUS_DST_ALPHA
}


BLEND_LOGIC_OPS = {
    'CLEAR': bgl.GL_CLEAR,
    'AND': bgl.GL_AND,
    'REVAND': bgl.GL_AND_REVERSE,
    'COPY': bgl.GL_COPY,
    'INVAND': bgl.GL_AND_INVERTED,
    'NOOP': bgl.GL_NOOP,
    'XOR': bgl.GL_XOR,
    'OR': bgl.GL_OR,
    'NOR': bgl.GL_NOR,
    'EQUIV': bgl.GL_EQUIV,
    'INV': bgl.GL_INVERT,
    'REVOR': bgl.GL_OR_REVERSE,
    'INVCOPY': bgl.GL_COPY_INVERTED,
    'INVOR': bgl.GL_OR_INVERTED,
    'NAND': bgl.GL_NAND,
    'SET': bgl.GL_SET
}


CULL_MODES = {
    'FRONT': bgl.GL_FRONT,
    'BACK': bgl.GL_BACK,
    'BOTH': bgl.GL_FRONT_AND_BACK,
}


DEPTH_FUNCS = {
    'NEVER': bgl.GL_NEVER,
    'LESS': bgl.GL_LESS,
    'EQUAL': bgl.GL_EQUAL,
    'LEQUAL': bgl.GL_LEQUAL,
    'GREATER': bgl.GL_GREATER,
    'NEQUAL': bgl.GL_NOTEQUAL,
    'GEQUAL': bgl.GL_GEQUAL,
    'ALWAYS': bgl.GL_ALWAYS
}


def printDepsgraphUpdateInfo(update: bpy.types.DepsgraphUpdate):
    """Print info about a depsgraph update."""
    print("-------- DEPSGRAPH UPDATE --------\n"
        f"Updated ID:        {update.id},\n"
        f"Updated ID Name:   {update.id.name},\n"
        f"Updated Geometry:  {update.is_updated_geometry},\n"
        f"Updated Shading:   {update.is_updated_shading},\n"
        f"Updated Transform: {update.is_updated_transform}"
    )


def debugDraw():
    """Draw a simple quad with a blue gradient to the active framebuffer."""
    shaderInfo = gpu.types.GPUShaderCreateInfo()
    shaderInfo.vertex_in(0, 'VEC2', "position")
    interfaceInfo = gpu.types.GPUStageInterfaceInfo("shader_interface")
    interfaceInfo.smooth('VEC2', "fragPosition")
    shaderInfo.vertex_out(interfaceInfo)
    shaderInfo.fragment_out(0, 'VEC4', "fragOutput")
    vertSrc = "gl_Position = vec4(position, 1, 1); fragPosition = position;"
    shaderInfo.vertex_source(f"void main() {{{vertSrc}}}")
    fragSrc = "fragOutput = vec4(fragPosition, 1.0, 1.0);"
    shaderInfo.fragment_source(f"void main() {{{fragSrc}}}")
    shader: gpu.types.GPUShader = gpu.shader.create_from_info(shaderInfo)
    verts = {"position": [[-1, -1], [1, -1], [-1, 1], [1, 1]]}
    idcs = [[0, 1, 2], [3, 2, 1]]
    batch: gpu.types.GPUBatch = batch_for_shader(shader, 'TRIS', verts, indices=idcs)
    batch.draw(shader)


def deleteBglTextures(bindcodes: list[int]):
    if bindcodes:
        n = len(bindcodes)
        bgl.glDeleteTextures(n, bgl.Buffer(bgl.GL_INT, n, bindcodes))


def getLoopAttributeData(mesh: bpy.types.Mesh,
                         colorLayerNames: list[str] = None, uvLayerNames: list[str] = None):
    """Get a dict mapping a mesh's BRRES attribute layers' names to their loop data."""
    loopAttrs: dict[str, np.ndarray] = {}
    loopVertIdcs = getLoopVertIdcs(mesh)
    loopFaceIdcs: np.ndarray = None # only calculated when necessary
    # positions
    loopAttrs["position"] = foreachGet(mesh.vertices, "co", 3)[loopVertIdcs]
    # normals
    if not mesh.has_custom_normals:
        mesh.calc_normals_split()
    loopAttrs["normal"] = foreachGet(mesh.loops, "normal", 3)
    # colors & uvs
    colorLayerNames = colorLayerNames if colorLayerNames else [""] * gx.MAX_CLR_ATTRS
    uvLayerNames = uvLayerNames if uvLayerNames else [""] * gx.MAX_UV_ATTRS
    clrData = getLayerData(mesh, colorLayerNames, unique=False)
    uvData = getLayerData(mesh, uvLayerNames, isUV=True, unique=False)
    attrTypeInfo = (("color", "uv"), (gx.ClrAttr, gx.UVAttr), (clrData, uvData))
    for aTypeName, aType, aLayerData in zip(*attrTypeInfo):
        for i, (layer, layerData, layerIdcs) in enumerate(aLayerData):
            # format data & add to dict
            if layer:
                # get data in per-loop domain, regardless of original domain
                try:
                    if layer.domain == 'POINT':
                        if loopVertIdcs is None:
                            loopVertIdcs = getLoopVertIdcs(mesh)
                        layerData = layerData[loopVertIdcs]
                    elif layer.domain == 'FACE':
                        if loopFaceIdcs is None:
                            loopFaceIdcs = getLoopFaceIdcs(mesh)
                        layerData = layerData[loopFaceIdcs]
                except AttributeError:
                    pass # this is a uv layer, which implicity has per-loop (corner) domain
                loopAttrs[f"{aTypeName}{i}"] = aType.pad(layerData)
    return loopAttrs


class BlendImageExtractor:

    _IMG_FMTS: dict[str, type[tex0.ImageFormat]] = {
        'I4': tex0.I4,
        'I8': tex0.I8,
        'IA4': tex0.IA4,
        'IA8': tex0.IA8,
        'RGB565': tex0.RGB565,
        'RGB5A3': tex0.RGB5A3,
        'RGBA8': tex0.RGBA8,
        'CMPR': tex0.CMPR
    }

    @classmethod
    def getFormat(cls, img: bpy.types.Image):
        return cls._IMG_FMTS[img.brres.fmt]

    @classmethod
    def getDims(cls, img: bpy.types.Image, setLargeToBlack = False):
        """Get the dimensions (for BRRES conversion) of a Blender image."""
        dims = np.array(img.size, dtype=np.integer)
        dims[dims == 0] = 1
        if setLargeToBlack and dims.max() > gx.MAX_TEXTURE_SIZE:
            dims[:] = 1
        return dims

    @classmethod
    def getRgba(cls, img: bpy.types.Image, setLargeToBlack = False):
        """Extract RGBA image data (guaranteed valid for BRRES conversion) from a Blender image.
        
        Optionally, set an image to 1x1 black if it exceeds the maximum Wii texture size.
        """
        if not img:
            return np.zeros((1, 1, 4), dtype=np.float32)
        dims = img.size
        px = np.array(img.pixels, dtype=np.float32).reshape(dims[1], dims[0], img.channels)
        # pad all image dimensions to at least 1 (render result is 0x0 if unset) & channels to 4
        px = np.pad(px, ((0, dims[1] == 0), (0, dims[0] == 0), (0, 4 - img.channels)))
        if setLargeToBlack and max(dims) > gx.MAX_TEXTURE_SIZE:
            px = px[:1, :1] * 0
        return px

    @classmethod
    def getRgbaWithDims(cls, img: bpy.types.Image, dims: tuple[int, int]):
        """Extract pixels from a Blender image, cropped or padded to the specified dimensions."""
        output = np.zeros((dims[1], dims[0], 4), dtype=np.float32)
        if not img:
            return output
        # crop & pad, all at once
        output[:img.size[1], :img.size[0]] = cls.getRgba(img)[:dims[1], :dims[0]]
        return output


class GLSLTevStageSels(GLSLStruct):
    tex = GLSLInt
    texSwap = GLSLInt
    ras = GLSLInt
    rasSwap = GLSLInt


class GLSLTevStageIndSettings(GLSLStruct):
    texIdx = GLSLInt
    fmt = GLSLInt
    bias = GLSLVec(GLSLInt, 3)
    bumpAlphaComp = GLSLInt
    mtxType = GLSLInt
    mtxIdx = GLSLInt
    wrap = GLSLVec(GLSLInt, 2)
    utcLOD = GLSLBool
    addPrev = GLSLBool


class GLSLTevStageCalcParams(GLSLStruct):
    constSel = GLSLInt
    args = GLSLVec(GLSLInt, 4)
    compMode = GLSLBool
    op = GLSLInt
    scale = GLSLFloat
    bias = GLSLFloat
    compChan = GLSLInt
    clamp = GLSLBool
    outputIdx = GLSLInt


class GLSLTevStage(GLSLStruct):
    sels = GLSLTevStageSels
    ind = GLSLTevStageIndSettings
    colorParams = GLSLTevStageCalcParams
    alphaParams = GLSLTevStageCalcParams

    @classmethod
    def fromStageSettings(cls, stage: TevStageSettings):
        rStage = GLSLTevStage()
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


class GLSLTexture(GLSLStruct):
    mtx = GLSLMat(GLSLFloat, 2, 3)
    dims = GLSLVec(GLSLInt, 2)
    mapMode = GLSLInt
    hasImg = GLSLBool

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
        mipmapLevels = len(img.brres.mipmaps) if rTex.hasImg else 0
        minFilter = f'{tex.minFilter}_MIPMAP_{tex.mipFilter}' if mipmapLevels > 0 else tex.minFilter
        rTex.filter = (TEX_FILTERS[minFilter], TEX_FILTERS[tex.magFilter])
        rTex.lodBias = tex.lodBias
        return rTex


class GLSLIndTex(GLSLStruct):
    texIdx = GLSLInt
    mode = GLSLInt
    lightIdx = GLSLInt
    coordScale = GLSLVec(GLSLInt, 2)


class GLSLLightChanSettings(GLSLStruct):
    difFromReg = GLSLBool
    ambFromReg = GLSLBool
    difMode = GLSLInt
    atnMode = GLSLInt
    enabledLights = GLSLArr(GLSLBool, 8)


class GLSLLightChan(GLSLStruct):
    difReg = GLSLVec(GLSLFloat, 4)
    ambReg = GLSLVec(GLSLFloat, 4)
    colorSettings = GLSLLightChanSettings
    alphaSettings = GLSLLightChanSettings

    def setRegs(self, lc: LightChannelSettings):
        """Update this light channel's diffuse/ambient registers from BRRES settings."""
        self.difReg = tuple(lc.difColor)
        self.ambReg = tuple(lc.ambColor)

    @classmethod
    def fromLightChanSettings(cls, lc: LightChannelSettings):
        rlc = GLSLLightChan()
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


class GLSLMaterial(GLSLStruct):
    colorSwaps = GLSLArr(GLSLVec(GLSLInt, 4), gx.MAX_COLOR_SWAPS)
    stages = GLSLArr(GLSLTevStage, gx.MAX_TEV_STAGES)
    textures = GLSLArr(GLSLTexture, gx.MAX_TEXTURES)
    inds = GLSLArr(GLSLIndTex, gx.MAX_INDIRECTS)
    indMtcs = GLSLArr(GLSLMat(GLSLFloat, 2, 3), gx.MAX_INDIRECT_MTCS)
    constColors = GLSLMat(GLSLFloat, gx.MAX_TEV_CONST_COLORS)
    outputColors = GLSLMat(GLSLFloat, gx.MAX_TEV_STAND_COLORS + 1)
    lightChans = GLSLArr(GLSLLightChan, gx.MAX_CLR_ATTRS)
    enableBlend = GLSLBool
    alphaTestVals = GLSLVec(GLSLFloat, 2)
    alphaTestComps = GLSLVec(GLSLInt, 2)
    alphaTestLogic = GLSLInt
    alphaTestEnable = GLSLBool
    constAlpha = GLSLFloat
    numStages = GLSLInt
    numTextures = GLSLInt
    numIndMtcs = GLSLInt

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


class GLSLMesh(GLSLStruct):
    colors = GLSLArr(GLSLInt, gx.MAX_CLR_ATTRS)
    uvs = GLSLArr(GLSLInt, gx.MAX_UV_ATTRS)


GLSL_STRUCTS = (
    GLSLTevStageSels,
    GLSLTevStageIndSettings,
    GLSLTevStageCalcParams,
    GLSLTevStage,
    GLSLTexture,
    GLSLIndTex,
    GLSLLightChanSettings,
    GLSLLightChan,
    GLSLMaterial,
    GLSLMesh
)


class GLSLMaterialUpdater:

    _EMPTY_UBO_BYTES = b"\x00" * GLSLMaterial.getSize()
    EMPTY_UBO = gpu.types.GPUUniformBuf(_EMPTY_UBO_BYTES)

    def __init__(self):
        self.mat = GLSLMaterial()

    def updateAnimation(self, mat: bpy.types.Material):
        """Update this material's animatable settings based on a Blender material."""
        brres = mat.brres
        rMat = self.mat
        # color registers
        rMat.setColorRegs(brres.colorRegs)
        for lc, rlc in zip(brres.lightChans, rMat.lightChans):
            rlc.setRegs(lc)
        # texture matrices & active texture images
        tfGen = brres.miscSettings.getTexTransformGen()
        for tex, rTex in zip(brres.textures, rMat.textures):
            rTex.setMtx(tex.transform, tfGen)
            if rTex.imgSlot != tex.activeImgSlot:
                rTex.imgSlot = tex.activeImgSlot
                img = tex.activeImg
                if img is not None:
                    rTex.hasImg = True
                    rTex.imgName = img.name
                else:
                    rTex.hasImg = False
        rMat.setIndMtcs(brres.indSettings.transforms, tfGen)

    def update(self, mat: bpy.types.Material):
        """Update this material based on a Blender material."""
        rMat = self.mat
        brres = mat.brres
        rMat.name = mat.name
        # tev settings
        try:
            tev = bpy.context.scene.brres.tevConfigs[brres.tevID]
            rMat.colorSwaps = tuple(tuple(enumVal(s, c) for c in "rgba") for s in tev.colorSwaps)
            enabledStages = tuple(stage for stage in tev.stages if not stage.hide)
            rMat.numStages = len(enabledStages)
            rMat.stages = tuple(GLSLTevStage.fromStageSettings(stage) for stage in enabledStages)
            indTexSlots = tev.indTexSlots
        except KeyError:
            rMat.numStages = 0
            indTexSlots = (1, ) * gx.MAX_INDIRECTS
        # textures
        tfGen = brres.miscSettings.getTexTransformGen()
        rMat.numTextures = len(brres.textures)
        rMat.textures = tuple(GLSLTexture.fromTexSettings(tex, tfGen) for tex in brres.textures)
        # indirect textures
        rInds: list[GLSLIndTex] = []
        for texSlot, ind in zip(indTexSlots, brres.indSettings.texConfigs):
            coordScale = tuple(int(s[4:]) for s in (ind.scaleU, ind.scaleV))
            rInd = GLSLIndTex()
            rInd.texIdx = texSlot - 1
            rInd.mode = enumVal(ind, "mode")
            rInd.lightIdx = ind.lightSlot - 1
            rInd.coordScale = coordScale
            rInds.append(rInd)
        rMat.inds = tuple(rInds)
        rMat.setIndMtcs(brres.indSettings.transforms, tfGen)
        # color regs
        rMat.setColorRegs(brres.colorRegs)
        # light channels
        rMat.lightChans = tuple(GLSLLightChan.fromLightChanSettings(lc) for lc in brres.lightChans)
        # alpha settings
        alphaSettings = brres.alphaSettings
        rMat.enableBlend = alphaSettings.enableBlendOp
        rMat.blendSubtract = alphaSettings.blendOp == 'SUBTRACT'
        rMat.blendSrcFac = BLEND_FACS[alphaSettings.blendSrcFactor]
        rMat.blendDstFac = BLEND_FACS[alphaSettings.blendDstFactor]
        rMat.enableBlendLogic = alphaSettings.enableLogicOp
        rMat.blendLogicOp = BLEND_LOGIC_OPS[alphaSettings.logicOp]
        rMat.enableDither = alphaSettings.enableDither
        rMat.blendUpdateColorBuffer = alphaSettings.enableColorUpdate
        rMat.blendUpdateAlphaBuffer = alphaSettings.enableAlphaUpdate
        rMat.enableCulling = alphaSettings.cullMode != 'NONE'
        if rMat.enableCulling:
            rMat.cullMode = CULL_MODES[alphaSettings.cullMode]
        rMat.isXlu = alphaSettings.isXlu
        rMat.alphaTestVals = tuple(alphaSettings.testVals)
        rMat.alphaTestComps = tuple(enumVal(alphaSettings, f"testComp{i + 1}") for i in range(2))
        rMat.alphaTestLogic = enumVal(alphaSettings, "testLogic")
        rMat.alphaTestEnable = True
        rMat.enableConstAlpha = alphaSettings.enableConstVal
        rMat.constAlpha = alphaSettings.constVal
        # depth settings
        depthSettings = brres.depthSettings
        rMat.enableDepthTest = depthSettings.enableDepthTest
        rMat.depthFunc = DEPTH_FUNCS[depthSettings.depthFunc]
        rMat.enableDepthUpdate = depthSettings.enableDepthUpdate


class GLSLMaterialUpdaterWithUBO(GLSLMaterialUpdater):
    """Updates a GLSLMaterial and keeps track of a UBO for it."""

    _EMPTY_UBO_BYTES = b"\x00" * GLSLMaterial.getSize()
    EMPTY_UBO = gpu.types.GPUUniformBuf(_EMPTY_UBO_BYTES)

    def __init__(self):
        self.ubo = gpu.types.GPUUniformBuf(self._EMPTY_UBO_BYTES)
        super().__init__()

    def updateAnimation(self, mat: bpy.types.Material):
        """Update this material info's animatable settings based on a Blender material."""
        super().updateAnimation(mat)
        self.ubo.update(self.mat.pack())

    def update(self, mat: bpy.types.Material):
        """Update this material info based on a Blender material."""
        super().update(mat)
        self.ubo.update(self.mat.pack())


class MaterialManager:
    pass


class ObjectInfo:

    _EMPTY_UBO_BYTES = b"\x00" * GLSLMesh.getSize()

    def __init__(self):
        self.batches: dict[GLSLMaterialUpdaterWithUBO, gpu.types.GPUBatch] = {}
        self.drawPrio = 0
        self.matrix: Matrix = Matrix.Identity(4)
        self.usedAttrs: set[str] = set()
        self.mesh = GLSLMesh()
        self.ubo = gpu.types.GPUUniformBuf(self._EMPTY_UBO_BYTES)


class TextureManager(ABC):

    @abstractmethod
    def updateTexture(self, tex: GLSLTexture):
        """Update the data corresponding to a texture, creating if nonexistent."""

    @abstractmethod
    def updateTexturesUsingImage(self, img: bpy.types.Image):
        """Update the data corresponding to all textures that use some image."""

    @abstractmethod
    def removeUnused(self):
        """Free texture resources not used since the last call to this method."""

    @abstractmethod
    def delete(self):
        """Clean up resources used by this TextureManager that must be freed."""


class BglTextureManager(TextureManager):

    def __init__(self):
        super().__init__()
        self._textures: dict[GLSLTexture, tuple[int, int]] = {}
        """OpenGL bindcode & mipmap count for each texture."""
        self._images: dict[str, list[bgl.Buffer]] = {}
        """Data buffer for each mipmap of each image (original included)."""
        self._usedTextures: set[GLSLTexture] = set()
        """Set of textures bound since the last removeUnused() call."""

    def _getImage(self, img: bpy.types.Image):
        """Get the data buffers corresponding to an image, updating if nonexistent."""
        try:
            return self._images[img.name]
        except KeyError:
            self._updateImage(img)
            return self._images[img.name]

    def _updateImage(self, img: bpy.types.Image):
        """Update the data buffers corresponding to an image."""
        imgFmt = BlendImageExtractor.getFormat(img)
        px = BlendImageExtractor.getRgba(img, setLargeToBlack=True)
        adjusted = imgFmt.adjustImg(px).astype(np.float32) # can't be float64 for bgl
        flattened = adjusted.flatten()
        mainPxBuffer = bgl.Buffer(bgl.GL_FLOAT, len(flattened), flattened)
        pxBuffers = [mainPxBuffer]
        self._images[img.name] = pxBuffers
        dims = np.array(px.shape[:2][::-1], dtype=np.integer)
        if not np.all(dims == BlendImageExtractor.getDims(img, setLargeToBlack=True)):
            print(img.name, dims, BlendImageExtractor.getDims(img, setLargeToBlack=True))
        # load mipmaps if provided
        for mm in img.brres.mipmaps:
            dims //= 2
            mmPx = BlendImageExtractor.getRgbaWithDims(mm.img, dims)
            mmPx = imgFmt.adjustImg(mmPx).astype(np.float32).flatten()
            mmPxBuffer = bgl.Buffer(bgl.GL_FLOAT, len(mmPx), mmPx)
            pxBuffers.append(mmPxBuffer)

    def _getTexture(self, tex: GLSLTexture):
        """Get the bindcode and mipmap count for a texture, updating if nonexistent."""
        try:
            return self._textures[tex]
        except KeyError:
            self.updateTexture(tex)
            return self._textures[tex]

    def updateTexture(self, tex: GLSLTexture):
        if not tex.hasImg:
            return
        img = bpy.data.images[tex.imgName]
        bindcodeBuf = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, bindcodeBuf)
        bindcode = bindcodeBuf[0] # pylint: disable=unsubscriptable-object
        self._textures[tex] = (bindcode, len(img.brres.mipmaps))
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, bindcode)
        imgBuffers = self._getImage(img)
        fmt = bgl.GL_RGBA
        dims = BlendImageExtractor.getDims(img, setLargeToBlack=True)
        for i, b in enumerate(imgBuffers):
            bgl.glTexImage2D(bgl.GL_TEXTURE_2D, i, fmt, *dims, 0, fmt, bgl.GL_FLOAT, b)
            dims //= 2

    def updateTexturesUsingImage(self, img: Image):
        if img.name in self._images:
            self._updateImage(img)
            name = img.name
            for tex in self._textures:
                if tex.imgName == name:
                    self.updateTexture(tex)

    def removeUnused(self):
        unusedBindcodes: list[int] = []
        for texture in tuple(self._textures): # tuple() so removal doesn't mess w/ iteration
            if texture not in self._usedTextures:
                bindcode, numMipmaps = self._textures.pop(texture)
                unusedBindcodes.append(bindcode)
        deleteBglTextures(unusedBindcodes)
        self._usedTextures.clear()

    def delete(self):
        bindcodes = [bindcode for (bindcode, numMipmaps) in self._textures.values()]
        deleteBglTextures(bindcodes)

    def bindTexture(self, texture: GLSLTexture):
        """Bind a texture in the OpenGL state."""
        self._usedTextures.add(texture)
        bindcode, mipmapLevels = self._getTexture(texture)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, bindcode)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, texture.wrap[0])
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, texture.wrap[1])
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, texture.filter[0])
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, texture.filter[1])
        bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_LOD_BIAS, texture.lodBias)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAX_LEVEL, mipmapLevels)


class PreviewTextureManager(TextureManager):

    def __init__(self):
        super().__init__()
        self._textures: dict[GLSLTexture, gpu.types.GPUTexture] = {}
        """GPUTexture for each texture."""
        self._images: dict[str, gpu.types.Buffer] = {}
        """GPU data buffer for each image."""
        self._usedTextures: set[GLSLTexture] = set()
        """Set of textures bound since the last removeUnused() call."""

    def _getImage(self, img: bpy.types.Image):
        """Get the GPU data buffer corresponding to an image, updating if nonexistent."""
        try:
            return self._images[img.name]
        except KeyError:
            self._updateImage(img)
            return self._images[img.name]

    def _updateImage(self, img: bpy.types.Image):
        """Update the GPU data buffer corresponding to an image."""
        px = BlendImageExtractor.getRgba(img, setLargeToBlack=True)
        # convert to float32 since float64 (default) is not allowed for 32-bit float buffer
        adjusted = BlendImageExtractor.getFormat(img).adjustImg(px).astype(np.float32)
        flattened = adjusted.flatten()
        self._images[img.name] = gpu.types.Buffer('FLOAT', len(flattened), flattened)

    def getTexture(self, tex: GLSLTexture):
        """Get the GPUTexture corresponding to a texture, updating if nonexistent."""
        self._usedTextures.add(tex)
        try:
            return self._textures[tex]
        except KeyError:
            self.updateTexture(tex)
            return self._textures[tex]

    def updateTexture(self, tex: GLSLTexture):
        if not tex.hasImg:
            return
        img: bpy.types.Image = bpy.data.images[tex.imgName]
        pxBuf = self._getImage(img)
        fmt = 'RGBA32F'
        dims = BlendImageExtractor.getDims(img, setLargeToBlack=True)
        self._textures[tex] = gpu.types.GPUTexture(dims, format=fmt, data=pxBuf)

    def updateTexturesUsingImage(self, img: Image):
        if img.name in self._images:
            self._updateImage(img)
            name = img.name
            for tex in self._textures:
                if tex.imgName == name:
                    self.updateTexture(tex)

    def removeUnused(self):
        self._textures = {
            glslTex: gpuTex
            for glslTex, gpuTex in self._textures.items()
            if glslTex in self._usedTextures
        }
        self._usedTextures.clear()

    def delete(self):
        pass


class BrresRenderer(ABC, Generic[TextureManagerT]):
    """Renders a Blender BRRES scene."""

    def __init__(self):
        self.shader: gpu.types.GPUShader = None
        self.materials: dict[str, GLSLMaterialUpdaterWithUBO] = {}
        self.objects: dict[str, ObjectInfo] = {}
        self._textureManager: TextureManagerT = None

    @classmethod
    def _compileShader(cls) -> gpu.types.GPUShader:
        """Compile & return the main BRRES shader."""
        shaderInfo = gpu.types.GPUShaderCreateInfo()
        # uniforms
        shaderInfo.typedef_source("".join(s.getSource() for s in GLSL_STRUCTS))
        shaderInfo.uniform_buf(1, GLSLMaterial.getName(), "material")
        shaderInfo.uniform_buf(2, GLSLMesh.getName(), "mesh")
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

    def _genBatch(self, batchType: str, content: dict[str, np.ndarray], indices: np.ndarray):
        """Generate a GPUBatch for this renderer's shader. (Faster than batch_for_shader())"""
        # vboFormat = gpu.types.GPUVertFormat()
        # attrInfo = {name: int(attrType[-1]) for name, attrType in self.shader.attrs_info_get()}
        # for name, attrLen in attrInfo.items():
        #     vboFormat.attr_add(id=name, comp_type='F32', len=attrLen, fetch_mode='FLOAT')
        vboFormat = self.shader.format_calc()
        vboLen = len(next(iter(content.values())))
        vbo = gpu.types.GPUVertBuf(vboFormat, vboLen)
        # for name, attrLen in attrInfo.items():
        #     try:
        #         data = content[name]
        #     except KeyError:
        #         data = np.empty((vboLen, attrLen))
        #     vbo.attr_fill(name, data)
        for name, data in content.items():
            vbo.attr_fill(name, data)
        ibo = gpu.types.GPUIndexBuf(type=batchType, seq=indices)
        return gpu.types.GPUBatch(type=batchType, buf=vbo, elem=ibo)

    def _getDrawCalls(self):
        """List of things drawn by this renderer.
        
        Each item has an object info, material, and batch for drawing, sorted as sorted on hardware.
        """
        objects = reversed(self.objects.values())
        drawCalls = [(o, m, b) for o in objects for m, b in o.batches.items()]
        drawCalls.sort(key=lambda v: (
            v[1] and v[1].mat.isXlu,
            v[0].drawPrio,
            v[1].mat.name if v[1] else ""
        ))
        return drawCalls

    def _getMaterial(self, mat: bpy.types.Material):
        """Get the GLSLMaterial corresponding to a Blender material, updating if nonexistent."""
        try:
            return self.materials[mat.name]
        except KeyError:
            self._updateMaterial(mat)
            return self.materials[mat.name]

    def _updateMaterial(self, mat: bpy.types.Material):
        """Update the GLSLMaterial corresponding to a Blender material, creating if nonexistent."""
        if mat.name not in self.materials:
            self.materials[mat.name] = GLSLMaterialUpdaterWithUBO()
        renderMat = self.materials[mat.name]
        renderMat.update(mat)
        for tex in renderMat.mat.textures:
            self._textureManager.updateTexture(tex)

    def _getBrresLayerNames(self, mesh: bpy.types.Mesh) -> tuple[list[str], list[str]]:
        """Get the BRRES color & UV attribute names for a mesh."""
        return (mesh.brres.meshAttrs.clrs, mesh.brres.meshAttrs.uvs)

    def delete(self):
        """Clean up resources managed by this renderer."""
        self._textureManager.delete()

    def _updateMeshCache(self, obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph):
        """Update an object's mesh in the rendering cache, or add it if the object's not there yet.

        If the object has no mesh, delete it (or don't add it).

        Also, if the object uses any materials that aren't in the cache yet, add them to it (but
        don't update existing ones).
        """
        # get mesh (and delete if object has none and there's one in the cache)
        try:
            mesh: bpy.types.Mesh = obj.to_mesh()
        except RuntimeError: # object doesn't have geometry (it's a camera, light, etc)
            if obj.name in self.objects:
                del self.objects[obj.name]
            return
        brres = obj.original.data.brres if obj.original.type == 'MESH' else None
        # get object info/create if none exists
        try:
            objInfo = self.objects[obj.name]
        except KeyError: # add object to cache if not in it yet
            self.objects[obj.name] = objInfo = ObjectInfo()
            self._updateObjMtxCache(obj)
        # set draw priority
        objInfo.drawPrio = brres.drawPrio if brres and brres.enableDrawPrio else 0
        # calculate triangles & normals
        mesh.calc_loop_triangles()
        mesh.calc_normals_split()
        # generate a batch for each material used
        loopIdcs = foreachGet(mesh.loop_triangles, "loops", 3, np.uint32)
        matLoopIdcs = np.zeros(loopIdcs.shape)
        if len(mesh.materials) > 1: # lookup is wasteful if there's only one material
            matLoopIdcs = getFaceMatIdcs(mesh)[getLoopFaceIdcs(mesh)][loopIdcs]
        layerNames = self._getBrresLayerNames(mesh)
        attrs = getLoopAttributeData(mesh, *layerNames)
        objInfo.batches.clear()
        noMat = np.full(loopIdcs.shape, len(mesh.materials) == 0) # all loops w/ no material
        matSlotIdcs: dict[bpy.types.Material, list[int]] = {}
        for i, mat in enumerate(mesh.materials):
            # get matSlotIdcs to contain the indices used for each material
            # (may be multiple if the same material shows up in multiple slots)
            if mat in matSlotIdcs:
                matSlotIdcs[mat].append(i)
            else:
                matSlotIdcs[mat] = [i]
        if None in matSlotIdcs:
            noMat = np.logical_or(noMat, np.isin(noMat, matSlotIdcs.pop(None)))
        for mat, idcs in matSlotIdcs.items():
            matInfo = self._getMaterial(mat)
            idcs = loopIdcs[np.isin(matLoopIdcs, idcs)]
            objInfo.batches[matInfo] = self._genBatch('TRIS', attrs, idcs)
        if np.any(noMat):
            idcs = loopIdcs[noMat]
            objInfo.batches[None] = self._genBatch('TRIS', attrs, idcs)
        obj.to_mesh_clear()
        # set constant vals for unprovided attributes (or -1 if provided)
        # constant val is usually 0, except for previews, where colors get 1
        usedAttrs = set(attrs)
        if objInfo.usedAttrs != usedAttrs:
            objInfo.usedAttrs = usedAttrs
            m = objInfo.mesh
            m.colors = tuple(-1 if f"color{i}" in attrs else 1 for i in range(gx.MAX_CLR_ATTRS))
            m.uvs = tuple(-1 if f"uv{i}" in attrs else 0 for i in range(gx.MAX_UV_ATTRS))
            objInfo.ubo.update(m.pack())
        return objInfo

    def _updateObjMtxCache(self, obj: bpy.types.Object):
        """Update an object's matrix in the rendering cache. Do nothing if object's not present."""
        objInfo = self.objects.get(obj.name)
        if objInfo:
            objInfo.matrix = obj.matrix_world.copy()

    def update(self, depsgraph: bpy.types.Depsgraph, context: bpy.types.Context = None):
        """Update this renderer's settings from a Blender depsgraph & optional context.

        If no context is provided, then the settings will be updated for new & deleted Blender
        objects, but changes to existing ones will be ignored.
        """
        # shader has to be compiled in drawing context or something so do that here instead of init
        # (otherwise there are issues w/ material previews, which are my archnemesis at this point)
        if self.shader is None:
            self.shader = self._compileShader()
        # remove deleted stuff
        visObjs = set(depsgraph.objects)
        if context:
            # determine which objects are visible
            vl = context.view_layer
            vp = context.space_data
            visObjs = {ob for ob in visObjs if ob.original.visible_get(view_layer=vl, viewport=vp)}
        names = {obj.name for obj in visObjs}
        self.objects = {n: info for n, info in self.objects.items() if n in names}
        isMatUpdate = depsgraph.id_type_updated('MATERIAL')
        if isMatUpdate:
            for matName, matInfo in tuple(self.materials.items()):
                # if material has been deleted (or renamed), remove it from renderer
                # additionally, regenerate any objects that used it
                if matName not in bpy.data.materials:
                    del self.materials[matName]
                    for objName, objInfo in self.objects.items():
                        if matInfo in objInfo.batches:
                            self._updateMeshCache(depsgraph.objects[objName], depsgraph)
        # add new stuff
        for obj in visObjs:
            if obj.name not in self.objects:
                self._updateMeshCache(obj, depsgraph)
        # update modified stuff
        tevConfigs = context.scene.brres.tevConfigs if context else ()
        tevIDs = {t.uuid for t in tevConfigs} | {""} # for tev config deletion detection
        for update in depsgraph.updates:
            updateId = update.id
            if isinstance(updateId, bpy.types.Object):
                if updateId.name in self.objects:
                    # check above ensures hidden things that are updated stay hidden
                    if update.is_updated_transform:
                        self._updateObjMtxCache(updateId)
                    if update.is_updated_geometry:
                        self._updateMeshCache(updateId, depsgraph)
            elif isinstance(updateId, bpy.types.Material):
                if update.is_updated_shading and update.is_updated_geometry:
                    # this indicates some material property was changed by the user (not animation)
                    self._updateMaterial(updateId)
                elif updateId.name in self.materials:
                    self.materials[updateId.name].updateAnimation(updateId)
            elif isinstance(updateId, bpy.types.Image):
                # material updates can sometimes trigger image updates for some reason,
                # so make sure this is an actual image update
                if not isMatUpdate or not update.is_updated_shading:
                    self._textureManager.updateTexturesUsingImage(updateId)
            elif isinstance(updateId, bpy.types.Scene) and update.is_updated_geometry:
                if update.is_updated_geometry:
                    # this implies a tev update, so update all materials that use the active tev
                    # it could also mean a tev was deleted, so also
                    # update materials w/ invalid tev ids
                    try:
                        activeTev = context.active_object.active_material.brres.tevID
                        for matName in self.materials:
                            mat = bpy.data.materials[matName]
                            if mat.brres.tevID == activeTev:
                                self._updateMaterial(mat)
                            elif mat.brres.tevID not in tevIDs:
                                # this means the material's tev was recently deleted, so reset uuid
                                # (proputils treats invalid id refs and empty refs the same, but
                                # this makes it easy to figure out which materials to update when
                                # configs are deleted, as otherwise all mats w/ no tev would update)
                                mat.brres.tevID = ""
                                self._updateMaterial(mat)
                    except AttributeError:
                        # no active material, so don't worry about it
                        # (or no context provided, which means this is a final render, and in that
                        # case tev deletion makes no sense as that's not animatable)
                        pass

    def draw(self, projectionMtx: Matrix, viewMtx: Matrix):
        """Draw the current BRRES scene to the active framebuffer."""
        self.preDraw()
        for (objInfo, matInfo, batch) in self._getDrawCalls():
            self.processDrawCall(viewMtx, projectionMtx, objInfo, matInfo, batch)
        self.postDraw()
        self._textureManager.removeUnused()

    @abstractmethod
    def processDrawCall(self, viewMtx: Matrix, projectionMtx: Matrix, objInfo: ObjectInfo,
                        matInfo: GLSLMaterialUpdaterWithUBO, batch: gpu.types.GPUBatch):
        """Draw something represented by a "draw call" (object/material/batch tuple)."""

    @abstractmethod
    def preDraw(self):
        """Initialize GPU state before draw calls are processed."""

    @abstractmethod
    def postDraw(self):
        """Clean up GPU state after draw calls are processed."""


class BrresBglRenderer(BrresRenderer[BglTextureManager]):
    """Renders a Blender BRRES scene using Blender's `bgl` module."""

    def __init__(self, assumeOpaqueMats: bool, noTransparentOverwrite: bool):
        super().__init__()
        self._assumeOpaqueMats = assumeOpaqueMats
        self._noTransparentOverwrite = noTransparentOverwrite
        self._textureManager = BglTextureManager()

    def _updateMaterial(self, mat: bpy.types.Material):
        super()._updateMaterial(mat)
        renderMat = self.materials[mat.name].mat
        if self._assumeOpaqueMats and not renderMat.enableBlend:
            renderMat.enableConstAlpha = True
            renderMat.constAlpha = 1

    def preDraw(self):
        self.shader.bind()
        self.shader.uniform_bool("forceOpaque", [False])
        # stencil buffer is used to determine which fragments have had values written to them
        # all 0 at first, and then set to 1 on writes
        # (used for "ignore background" functionality)
        bgl.glEnable(bgl.GL_STENCIL_TEST)
        bgl.glStencilFunc(bgl.GL_ALWAYS, True, 0xFF)
        bgl.glStencilOp(bgl.GL_REPLACE, bgl.GL_REPLACE, bgl.GL_REPLACE)

    def processDrawCall(self, viewMtx: Matrix, projectionMtx: Matrix, objInfo: ObjectInfo,
                        matInfo: GLSLMaterialUpdaterWithUBO, batch: gpu.types.GPUBatch):
        mvMtx = viewMtx @ objInfo.matrix
        self.shader.uniform_bool("isConstAlphaWrite", [False])
        self.shader.uniform_float("modelViewProjectionMtx", projectionMtx @ mvMtx)
        self.shader.uniform_float("normalMtx", mvMtx.to_3x3().inverted_safe().transposed())
        self.shader.uniform_block("mesh", objInfo.ubo)
        # load material data
        if matInfo is not None:
            self.shader.uniform_block("material", matInfo.ubo)
            shaderMat = matInfo.mat
            # textures
            for i, tex in enumerate(shaderMat.textures):
                if tex.hasImg:
                    bgl.glActiveTexture(bgl.GL_TEXTURE0 + i)
                    self._textureManager.bindTexture(tex)
            # dithering
            if shaderMat.enableDither:
                bgl.glEnable(bgl.GL_DITHER)
            else:
                bgl.glDisable(bgl.GL_DITHER)
            # culling
            if shaderMat.enableCulling:
                bgl.glEnable(bgl.GL_CULL_FACE)
                bgl.glCullFace(shaderMat.cullMode)
            else:
                bgl.glDisable(bgl.GL_CULL_FACE)
            # depth test
            if shaderMat.enableDepthTest:
                bgl.glEnable(bgl.GL_DEPTH_TEST)
            else:
                bgl.glDisable(bgl.GL_DEPTH_TEST)
            bgl.glDepthFunc(shaderMat.depthFunc)
            bgl.glDepthMask(shaderMat.enableDepthUpdate)
            # blending & logic op
            if shaderMat.enableBlend:
                bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
                if shaderMat.blendSubtract:
                    bgl.glEnable(bgl.GL_BLEND)
                    bgl.glBlendEquation(bgl.GL_FUNC_SUBTRACT)
                    bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE)
                else:
                    bgl.glEnable(bgl.GL_BLEND)
                    bgl.glBlendEquation(bgl.GL_FUNC_ADD)
                    bgl.glBlendFunc(shaderMat.blendSrcFac, shaderMat.blendDstFac)
            else:
                bgl.glDisable(bgl.GL_BLEND)
                if shaderMat.enableBlendLogic:
                    bgl.glDisable(bgl.GL_BLEND)
                    bgl.glEnable(bgl.GL_COLOR_LOGIC_OP)
                    bgl.glLogicOp(shaderMat.blendLogicOp)
                else:
                    bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
        else:
            self.shader.uniform_block("material", GLSLMaterialUpdaterWithUBO.EMPTY_UBO)
            bgl.glDisable(bgl.GL_STENCIL_TEST)
            bgl.glDisable(bgl.GL_BLEND)
            bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
            bgl.glDisable(bgl.GL_DITHER)
            bgl.glDisable(bgl.GL_CULL_FACE)
            bgl.glEnable(bgl.GL_DEPTH_TEST)
        # draw
        if self._noTransparentOverwrite:
            # draw color & alpha separately, w/ special attention to alpha blending
            # https://en.wikipedia.org/wiki/Alpha_compositing
            # (btw we can't just use glBlendFuncSeparate for this, bc that doesn't exist in bgl)

            # note: this is done regardless of whether blending is enabled for this material.
            # this is not required! it does mean that objects w/o blending will still have this
            # special blending applied, but objects w/o blending should really not have alpha
            # values anyway (and the "assume opaque materials" setting & constant alpha material
            # setting take care of that in most cases where it does happen), so i can't really
            # think of a use case either way
            # tldr: shaderMat.enableBlend isn't taken into acccount rn, but that's arbitrary

            # rgb
            bgl.glColorMask(True, True, True, False)
            batch.draw(self.shader)
            # alpha
            bgl.glColorMask(False, False, False, True)
            bgl.glEnable(bgl.GL_BLEND)
            bgl.glBlendEquation(bgl.GL_FUNC_ADD)
            bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
            batch.draw(self.shader)
            bgl.glColorMask(True, True, True, True)
        else:
            batch.draw(self.shader)
        # write constant alpha if enabled (must be done after blending, hence 2 draw calls)
        if matInfo and matInfo.mat.enableConstAlpha:
            bgl.glDisable(bgl.GL_BLEND)
            bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
            bgl.glDisable(bgl.GL_DITHER)
            self.shader.uniform_bool("isConstAlphaWrite", [True])
            bgl.glColorMask(False, False, False, True)
            batch.draw(self.shader)
            bgl.glColorMask(True, True, True, True)

    def postDraw(self):
        bgl.glDisable(bgl.GL_BLEND)
        bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
        bgl.glDisable(bgl.GL_DITHER)
        bgl.glDisable(bgl.GL_CULL_FACE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)


class BrresPreviewRenderer(BrresRenderer[PreviewTextureManager]):
    """Renders a Blender BRRES scene in "preview mode".
    
    Preview mode: Exclusively uses Blender's `gpu` module for rendering without touching the
    deprecated `bgl` module. This approach offers less control and therefore less accuracy
    (in general, it means rendering is significantly simplified), but it's required for material
    preview icon generation, which doesn't play nicely with `bgl`.
    """

    def __init__(self):
        super().__init__()
        self._textureManager = PreviewTextureManager()

    def _getBrresLayerNames(self, mesh: bpy.types.Mesh):
        return (["Col"] * gx.MAX_CLR_ATTRS, ["UVMap"] * gx.MAX_UV_ATTRS)

    def preDraw(self):
        self.shader.bind()
        # since bgl isn't allowed, keep things simple w/ a forced depth test
        gpu.state.depth_test_set('LESS_EQUAL')
        # also force drawn alpha values to be 1, since blending gets weird
        self.shader.uniform_bool("forceOpaque", [True])

    def processDrawCall(self, viewMtx: Matrix, projectionMtx: Matrix, objInfo: ObjectInfo,
                        matInfo: GLSLMaterialUpdaterWithUBO, batch: gpu.types.GPUBatch):
        mvMtx = viewMtx @ objInfo.matrix
        self.shader.uniform_bool("isConstAlphaWrite", [False])
        self.shader.uniform_float("modelViewProjectionMtx", projectionMtx @ mvMtx)
        self.shader.uniform_float("normalMtx", mvMtx.to_3x3().inverted_safe().transposed())
        self.shader.uniform_block("mesh", objInfo.ubo)
        # load material data
        if matInfo:
            self.shader.uniform_block("material", matInfo.ubo)
            for i, tex in enumerate(matInfo.mat.textures):
                if tex.hasImg:
                    self.shader.uniform_sampler(f"image{i}", self._textureManager.getTexture(tex))
        else:
            self.shader.uniform_block("material", GLSLMaterialUpdaterWithUBO.EMPTY_UBO)
        # draw
        batch.draw(self.shader)

    def postDraw(self):
        gpu.state.depth_test_set('NONE')


class BRRESRenderEngine(bpy.types.RenderEngine):
    bl_idname = "BERRYBUSH"
    bl_label = "BerryBush"
    bl_use_preview = True
    bl_use_gpu_context = True
    # not sure how to handle lookdev ("material preview") shading mode rn
    # workbench hides it as an option, and i wish i could do that, but that's hardcoded in blender
    # so for now, it just works exactly like rendered mode
    bl_use_eevee_viewport = False

    def __init__(self):
        self._sceneRenderers: dict[str, BrresRenderer] = {}
        """Maps scene names to their BRRES renderers."""
        self.isViewport = True
        """Whether a viewport render is being rendered."""
        self._backgroundColor = (0, 0, 0)
        # shader & batch
        verts = {"position": [[-1, -1], [1, -1], [-1, 1], [1, 1]]}
        idcs = [[0, 1, 2], [3, 2, 1]]
        self.shader = self._compileShader()
        self.batch: gpu.types.GPUBatch = batch_for_shader(self.shader, 'TRIS', verts, indices=idcs)
        # offscreen
        self.offscreen: gpu.types.GPUOffScreen = None
        # dimensions
        self._updateDims((1, 1))

    def _getSceneRenderer(self, scene: bpy.types.Scene):
        """Get the BRRES renderer for a scene, creating one if it doesn't exist yet."""
        try:
            return self._sceneRenderers[scene.name]
        except KeyError:
            renderer: BrresRenderer
            if self.is_preview:
                renderer = BrresPreviewRenderer()
            else:
                isTransparent = scene.render.film_transparent and not self.isViewport
                assumeOpaqueMats = isTransparent and scene.brres.renderAssumeOpaqueMats
                noTransparentOverwrite = isTransparent and scene.brres.renderNoTransparentOverwrite
                renderer = BrresBglRenderer(assumeOpaqueMats, noTransparentOverwrite)
            self._sceneRenderers[scene.name] = renderer
            return renderer

    @classmethod
    def _compileShader(cls) -> gpu.types.GPUShader:
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

    def _updateDims(self, dims: tuple[int, int]):
        self.dims = dims
        if self.offscreen is not None:
            self.offscreen.free()
        # when an offscreen is created, the previously-bound framebuffer is unbound, though this
        # change is seemingly invisible in the gpu module (can only be detected & fixed w/ bgl)
        # probably an api bug?
        if not self.is_preview: # no bgl allowed for previews, but fb is irrelevant there anyway
            activeFb = bgl.Buffer(bgl.GL_INT, 1)
            bgl.glGetIntegerv(bgl.GL_FRAMEBUFFER_BINDING, activeFb)
        self.offscreen = gpu.types.GPUOffScreen(*dims)
        if not self.is_preview:
            bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, activeFb[0]) # pylint: disable=unsubscriptable-object

    @classmethod
    def getSupportedBlenderPanels(cls):
        """Get the standard Blender panels supported by this render engine."""
        panels: set[type[bpy.types.Panel]] = set()
        # support panels w/ BLENDER_RENDER, plus anything in additional & minus anything in excluded
        additional = {
            # preview stuff disabled as it's bugged for now and idk the fix; see
            # https://blender.stackexchange.com/questions/285693/custom-render-engine-creating-material-previews-with-opengl
            "MATERIAL_PT_preview",
            "EEVEE_MATERIAL_PT_context_material"
        }
        excluded = {
            "EEVEE_MATERIAL_PT_viewport_settings"
        }
        for p in bpy.types.Panel.__subclasses__():
            if hasattr(p, 'COMPAT_ENGINES'):
                n = p.__name__
                if ('BLENDER_RENDER' in p.COMPAT_ENGINES and n not in excluded) or n in additional:
                    panels.add(p)
        return panels

    @classmethod
    def registerOnPanels(cls):
        """Add this render engine to all standard Blender panels it supports."""
        for p in cls.getSupportedBlenderPanels():
            p.COMPAT_ENGINES.add(cls.bl_idname)

    @classmethod
    def unregisterOnPanels(cls):
        """Remove this render engine from all standard Blender panels it supports."""
        for p in cls.getSupportedBlenderPanels():
            try:
                p.COMPAT_ENGINES.remove(cls.bl_idname)
            except KeyError:
                pass

    def __del__(self):
        try:
            for renderer in self._sceneRenderers.values():
                renderer.delete()
            self.offscreen.free()
        except AttributeError:
            pass

    def _drawScene(self, scene: bpy.types.Scene, projectionMtx: Matrix, viewMtx: Matrix):
        # get active framebuffer for re-bind to fix the problem described in _updateDims,
        # since it also gets triggered when the offscreen is bound immediately after creation
        # (seemingly not necessary after first frame, but i'm doing it every time anyway to be safe)
        activeFb = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetIntegerv(bgl.GL_FRAMEBUFFER_BINDING, activeFb)
        # pass 1: main rendering
        with self.offscreen.bind():
            fb: gpu.types.GPUFrameBuffer = gpu.state.active_framebuffer_get()
            # write mask must be enabled to clear
            bgl.glDepthMask(True)
            bgl.glStencilMask(0xFF)
            fb.clear(color=(*self._backgroundColor, 0), depth=1, stencil=0)
            self._getSceneRenderer(scene).draw(projectionMtx, viewMtx)
        # pass 2: post-processing
        bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, activeFb[0]) # pylint: disable=unsubscriptable-object
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.offscreen.color_texture)
        self.shader.bind()
        doAlpha = scene.render.film_transparent and not self.isViewport
        self.shader.uniform_bool("doAlpha", [doAlpha])
        self.batch.draw(self.shader)

    def _drawPreview(self, scene: bpy.types.Scene, projectionMtx: Matrix, viewMtx: Matrix):
        """Draw a BRRES scene in preview mode (no bgl) to the active framebuffer."""
        # pass 1: main rendering
        with self.offscreen.bind():
            fb: gpu.types.GPUFrameBuffer = gpu.state.active_framebuffer_get()
            fb.clear(color=(*self._backgroundColor, 0), depth=1)
            self._getSceneRenderer(scene).draw(projectionMtx, viewMtx)
        # pass 2: post-processing
        self.shader.bind()
        self.shader.uniform_sampler("tex", self.offscreen.texture_color)
        self.shader.uniform_bool("doAlpha", [True])
        self.batch.draw(self.shader)

    def render(self, depsgraph: bpy.types.Depsgraph):
        self.isViewport = False
        scene = depsgraph.scene
        render = scene.render
        self._getSceneRenderer(scene).update(depsgraph)
        scale = render.resolution_percentage / 100
        dims = (int(render.resolution_x * scale), int(render.resolution_y * scale))
        self._updateDims(dims)
        result = self.begin_result(0, 0, *dims, layer=depsgraph.view_layer.name)
        projectionMtx = scene.camera.calc_matrix_camera(depsgraph, x=dims[0], y=dims[1])
        viewMtx = scene.camera.matrix_world.inverted()
        # after way too much debugging, i've found that this is false by default :)
        if not self.is_preview:
            bgl.glDepthMask(True)
            bgl.glStencilMask(0xFF)
        offscreen = gpu.types.GPUOffScreen(*dims, format='RGBA32F')
        with offscreen.bind():
            fb: gpu.types.GPUFrameBuffer = gpu.state.active_framebuffer_get()
            if self.is_preview:
                self._drawPreview(scene, projectionMtx, viewMtx)
            else:
                self._drawScene(scene, projectionMtx, viewMtx)
            b = gpu.types.Buffer('FLOAT', (dims[0] * dims[1], 4))
            fb.read_color(0, 0, *dims, 4, 0, 'FLOAT', data=b)
            result.layers[0].passes["Combined"].rect.foreach_set(b)
        offscreen.free()
        self.end_result(result)

    def view_update(self, context, depsgraph):
        # import cProfile, pstats
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()

        self.isViewport = True
        scene = context.scene
        self._getSceneRenderer(scene).update(depsgraph, context)
        dims = (context.region.width, context.region.height)
        if dims != self.dims:
            self._updateDims(dims)
        self._backgroundColor = np.array(context.scene.world.color[:3]) ** .4545
        # this makes response immediate for some updates that otherwise result in a delayed draw
        # (for instance, using the proputils collection move operators)
        self.tag_redraw()

        # pr.disable()
        # with open(LOG_PATH, "w", encoding="utf-8") as logFile:
        #     ps = pstats.Stats(pr, stream=logFile).sort_stats(SortKey.CUMULATIVE)
        #     ps.print_stats()

    def view_draw(self, context, depsgraph):
        self.isViewport = True
        self._drawScene(
            scene=context.scene,
            projectionMtx=context.region_data.window_matrix,
            viewMtx=context.region_data.view_matrix
        )


class FilmPanel(PropertyPanel):
    bl_idname = "RENDER_PT_brres_film"
    bl_label = "Film"
    bl_context = "render"
    COMPAT_ENGINES = {'BERRYBUSH'}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        render = scene.render
        layout.prop(render, "film_transparent")
        col = layout.row().column()
        col.prop(scene.brres, "renderAssumeOpaqueMats")
        drawColumnSeparator(col)
        col.prop(scene.brres, "renderNoTransparentOverwrite")
        col.enabled = render.film_transparent

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return context.engine in cls.COMPAT_ENGINES
