# standard imports
import pathlib
# 3rd party imports
import bpy
import bgl # this is deprecated, but has a lot of functionality that gpu still lacks
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
from .brresexport import IMG_FMTS, padImgData
from .glslstruct import GLSLBool, GLSLInt, GLSLFloat, GLSLVec, GLSLArr, GLSLMat, GLSLStruct
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


def printUpdateInfo(update: bpy.types.DepsgraphUpdate):
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


def getLoopAttrs(mesh: bpy.types.Mesh, clrs: list[str] = None, uvs: list[str] = None):
    """Get a dict for a mesh with its loops' BRRES attributes based on attribute layer names."""
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
    clrs = clrs if clrs is not None else [""] * gx.MAX_CLR_ATTRS
    uvs = uvs if uvs is not None else [""] * gx.MAX_UV_ATTRS
    clrData = getLayerData(mesh, clrs, unique=False)
    uvData = getLayerData(mesh, uvs, isUV=True, unique=False)
    attrTypeInfo = (("color", "uv"), (gx.ClrAttr, gx.UVAttr), (clrData, uvData))
    for aTypeName, aType, aLayerData in zip(*attrTypeInfo):
        for i, (layer, layerData, layerIdcs) in enumerate(aLayerData):
            # format data & add to dict
            if layer is not None:
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

    def bind(self, bindcode: int, mipmapLevels: int):
        """Bind this texture in OpenGL."""
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, bindcode)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, self.wrap[0])
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, self.wrap[1])
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, self.filter[0])
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, self.filter[1])
        bgl.glTexParameterf(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_LOD_BIAS, self.lodBias)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAX_LEVEL, mipmapLevels)

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
        rTex.dims = img.size if rTex.hasImg else (0, 0)
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


class MaterialInfo:

    _EMPTY_UBO_BYTES = b"\x00" * GLSLMaterial.getSize()
    EMPTY_UBO = gpu.types.GPUUniformBuf(_EMPTY_UBO_BYTES)

    def __init__(self, mat: bpy.types.Material):
        self.ubo = gpu.types.GPUUniformBuf(self._EMPTY_UBO_BYTES)
        self.mat = GLSLMaterial()
        self.update(mat)

    def updateAnimation(self, mat: bpy.types.Material, renderer: "MainBRRESRenderer"):
        """Update this material info's animatable settings based on a Blender material."""
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
                    if img.name not in renderer.images:
                        renderer.updateImgCache(img)
                else:
                    rTex.hasImg = False
        rMat.setIndMtcs(brres.indSettings.transforms, tfGen)
        # update ubo
        self.ubo.update(rMat.pack())

    def update(self, mat: bpy.types.Material):
        """Update this material info based on a Blender material."""
        scene = bpy.context.scene
        rMat = self.mat
        brres = mat.brres
        rMat.name = mat.name
        # tev settings
        try:
            tev = scene.brres.tevConfigs[brres.tevID]
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
        assumeOpaqueMats = scene.render.film_transparent and scene.brres.renderAssumeOpaqueMats
        if assumeOpaqueMats and not rMat.enableBlend:
            rMat.enableConstAlpha = True
            rMat.constAlpha = 1
        else:
            rMat.enableConstAlpha = alphaSettings.enableConstVal
            rMat.constAlpha = alphaSettings.constVal
        # depth settings
        depthSettings = brres.depthSettings
        rMat.enableDepthTest = depthSettings.enableDepthTest
        rMat.depthFunc = DEPTH_FUNCS[depthSettings.depthFunc]
        rMat.enableDepthUpdate = depthSettings.enableDepthUpdate
        # update ubo
        self.ubo.update(rMat.pack())


class ObjectInfo:

    _EMPTY_UBO_BYTES = b"\x00" * GLSLMesh.getSize()

    def __init__(self):
        self.batches: dict[MaterialInfo, gpu.types.GPUBatch] = {}
        self.drawPrio = 0
        self.matrix: Matrix = Matrix.Identity(4)
        self.usedAttrs: set[str] = set()
        self.mesh = GLSLMesh()
        self.ubo = gpu.types.GPUUniformBuf(self._EMPTY_UBO_BYTES)


class MainBRRESRenderer():
    """Renders a Blender BRRES scene without any post-processing."""

    def __init__(self, previewMode = False):
        self.noTransparentOverwrite = False
        self.shader: gpu.types.GPUShader = None
        self.materials: dict[str, MaterialInfo] = {}
        self.objects: dict[str, ObjectInfo] = {}
        self.images: dict[str, tuple[int, int]] = {}
        """Bindcode and # of mipmap levels for each image."""
        # preview mode: used for material preview icons
        # https://blender.stackexchange.com/questions/285693/custom-render-engine-creating-material-previews-with-opengl
        # bgl calls during preview drawing cause glitches, so things have to be done differently
        # this means that custom mipmaps, blending, dithering, culling, and depth testing are
        # unsupported for previews
        self.previewMode = previewMode

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


    def _updateMatCache(self, mat: bpy.types.Material):
        """Update a material in the rendering cache, or add it if it's not there yet."""
        try:
            self.materials[mat.name].update(mat)
        except KeyError:
            self.materials[mat.name] = MaterialInfo(mat)
        for tex in mat.brres.textures:
            img = tex.activeImg
            if img is not None and img.name not in self.images:
                self.updateImgCache(img)

    def updateImgCache(self, img: bpy.types.Image):
        """Update an image in the rendering cache, or add it if it's not there yet."""
        # if image has already been generated, make sure to delete the existing one
        if img.name in self.images:
            self._deleteImg(img.name)
        # get image data
        # why use img.pixels directly, rather than img.gl_load() or gpu.texture.from_image()?
        # 3 main reasons
        # a) more control over texture settings like mipmaps, wrapping, etc (although this doesn't
        # apply to previews bc we can't use bgl there)
        # b) direct control over the pixels so we can adjust based on the selected wii format
        # c) this lets us bypass image colorspace settings & use raw data for all of them
        # why bypass the colorspace settings? because blender uses linear while nw4r uses srgb
        # this makes colorspace issues a headache, one example being that since blender
        # uses linear, raw = linear, which would be misleading for this engine (since you'd expect
        # raw = srgb); we could add a special case for raw so that it is treated as srgb in
        # berrybush, but blender would still use raw = linear elsewhere (e.g., the uv editor))
        # so yeah, this is still a little misleading because we ignore the settings entirely, but i
        # think this solution will cause the least confusion so it's what i'm going with for now
        dims = np.array(img.size, dtype=np.integer)
        px = np.array(img.pixels, dtype=np.float32).reshape(*dims[::-1], img.channels)
        if dims.max() > gx.MAX_TEXTURE_SIZE: # crop to 1 pixel if too big
            dims[:] = 1
            px = px[:1, :1] * 0
        # pad all image dimensions to at least 1 (render result is 0x0 if unset)
        dimPadding = [(0, px.shape[i] == 0) for i in range(len(px.shape))]
        px = np.pad(px, dimPadding)
        # pad image to 4 channels (rgba) for wii format adjustment
        chanPadding = [(0, 0)] * len(px.shape)
        chanPadding[-1] = (0, 4 - px.shape[-1])
        px = np.pad(px, chanPadding)
        dims[:] = px.shape[:2][::-1] # update dims after padding
        imgFmt = IMG_FMTS[img.brres.fmt]
        px = imgFmt.adjustImg(px).astype(np.float32) # can't be float64 for bgl
        numChans = px.shape[-1]
        px = px.flatten()
        # load image & settings
        if self.previewMode:
            # no bgl allowed :( so use this method w/o custom mipmaps
            # instead of image bindcode & number of mipmap levels, gpu texture is stored
            fmt = ('R32F', 'RG32F', 'RGB32F', 'RGBA32F')[numChans - 1]
            pxBuf = gpu.types.Buffer('FLOAT', len(px), px)
            self.images[img.name] = gpu.types.GPUTexture(dims, format=fmt, data=pxBuf)
            return
        bindcodeBuf = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, bindcodeBuf)
        bindcode = bindcodeBuf[0] # pylint: disable=unsubscriptable-object
        self.images[img.name] = (bindcode, len(img.brres.mipmaps))
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, bindcode)
        b = bgl.Buffer(bgl.GL_FLOAT, len(px), px)
        fmt = (bgl.GL_RED, bgl.GL_RG, bgl.GL_RGB, bgl.GL_RGBA)[numChans - 1]
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, fmt, *dims, 0, fmt, bgl.GL_FLOAT, b)
        # load mipmaps if provided
        for i, mm in enumerate(img.brres.mipmaps):
            dims //= 2
            mmPx = padImgData(mm.img, (dims[1], dims[0], numChans))
            mmPx = imgFmt.adjustImg(mmPx).astype(np.float32).flatten()
            b = bgl.Buffer(bgl.GL_FLOAT, len(mmPx), mmPx)
            bgl.glTexImage2D(bgl.GL_TEXTURE_2D, i + 1, fmt, *dims, 0, fmt, bgl.GL_FLOAT, b)

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
        matIdcs = np.zeros(loopIdcs.shape)
        if len(mesh.materials) > 1: # lookup is wasteful if there's only one material
            matIdcs = getFaceMatIdcs(mesh)[getLoopFaceIdcs(mesh)][loopIdcs]
        layerNames = (None, None)
        if self.previewMode:
            layerNames = (["Col"] * gx.MAX_CLR_ATTRS, ["UVMap"] * gx.MAX_UV_ATTRS)
        elif brres:
            layerNames = (brres.meshAttrs.clrs, brres.meshAttrs.uvs)
        attrs = getLoopAttrs(mesh, *layerNames)
        objInfo.batches.clear()
        noMat = np.full(loopIdcs.shape, len(mesh.materials) == 0) # all loops w/ no material
        for i, mat in enumerate(mesh.materials):
            if mat is None:
                noMat = np.logical_or(noMat, matIdcs == i)
                continue
            if mat.name not in self.materials:
                self._updateMatCache(mat)
            matInfo = self.materials[mat.name]
            idcs = loopIdcs[matIdcs == i]
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
        if objInfo is not None:
            objInfo.matrix = obj.matrix_world.copy()

    def _deleteImg(self, imgName: str):
        """Delete an image from the rendering cache and GL context."""
        if not self.previewMode:
            bgl.glDeleteTextures(1, bgl.Buffer(bgl.GL_INT, 1, [self.images[imgName][0]]))
        del self.images[imgName]

    def isPreviewWithFloor(self):
        """Determine if a material preview with a floor is being rendered.

        (This is necessary because in this case, the floor is skipped while drawing and the
        background color is set to the world's viewport color; we can't set the background color
        like this for previews w/o floors, since those need a transparent black background
        to display properly)
        """
        # when there's a floor, there's always an object called "Floor" w/ a material called "Floor"
        # sometimes there's actually a hidden floor, but its material is called "FloorHidden"
        try:
            return self.previewMode and self.materials["Floor"] in self.objects["Floor"].batches
        except KeyError:
            return False

    def delete(self):
        for img in tuple(self.images):
            self._deleteImg(img)

    def update(self, depsgraph: bpy.types.Depsgraph, context: bpy.types.Context = None,
               isFinal = False):
        """Update this renderer's settings from a Blender depsgraph & optional context.

        If no context is provided, then the settings will be updated for new & deleted Blender
        objects, but changes to existing ones will be ignored.
        """
        # shader has to be compiled in drawing context or something so do that here instead of init
        # (otherwise there are issues w/ material previews, which are my archnemesis at this point)
        if self.shader is None:
            self.shader = self._compileShader()
        # update scene render settings
        if depsgraph.id_type_updated('SCENE'):
            scene = depsgraph.scene
            isTransparent = isFinal and scene.render.film_transparent
            self.noTransparentOverwrite = isTransparent and scene.brres.renderNoTransparentOverwrite
        # remove deleted stuff
        isObjUpdate = depsgraph.id_type_updated('OBJECT')
        visObjs = set(depsgraph.objects)
        if context:
            # determine which objects are visible
            vl = context.view_layer
            vp = context.space_data
            visObjs = {ob for ob in visObjs if ob.original.visible_get(view_layer=vl, viewport=vp)}
        if isObjUpdate:
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
        isImgUpdate = depsgraph.id_type_updated('IMAGE')
        if isImgUpdate:
            usedImages = {t.imgName for m in self.materials.values() for t in m.mat.textures}
            for img in self.images.copy():
                if img not in usedImages:
                    self._deleteImg(img)
        # add new stuff
        for obj in visObjs:
            if obj.name not in self.objects:
                self._updateMeshCache(obj, depsgraph)
        # update modified stuff
        tevConfigs = context.scene.brres.tevConfigs if context else ()
        tevIDs = {t.uuid for t in tevConfigs} | {""} # for tev config deletion detection
        for update in depsgraph.updates:
            updatedID = update.id
            if isinstance(updatedID, bpy.types.Object) and updatedID.name in self.objects:
                # check above ensures hidden things that are updated stay hidden
                if update.is_updated_transform:
                    self._updateObjMtxCache(updatedID)
                if update.is_updated_geometry:
                    self._updateMeshCache(updatedID, depsgraph)
            elif isinstance(updatedID, bpy.types.Material) and update.is_updated_shading:
                if update.is_updated_geometry:
                    # this indicates some material property was changed by the user (not animation)
                    self._updateMatCache(updatedID)
                elif updatedID.name in self.materials:
                    self.materials[updatedID.name].updateAnimation(updatedID, self)
            elif isinstance(updatedID, bpy.types.Image) and updatedID.name in self.images:
                # material updates can sometimes trigger image updates for some reason,
                # so make sure this is an actual image update
                if not isMatUpdate or not update.is_updated_shading:
                    self.updateImgCache(updatedID)
            elif isinstance(updatedID, bpy.types.Scene) and update.is_updated_geometry:
                # this implies a tev update, so update all materials that use the active tev
                # it could also mean a tev was deleted, so also update materials w/ invalid tev ids
                try:
                    activeTev = context.active_object.active_material.brres.tevID
                    for matName in self.materials:
                        mat = bpy.data.materials[matName]
                        if mat.brres.tevID == activeTev:
                            self._updateMatCache(mat)
                        elif mat.brres.tevID not in tevIDs:
                            # this means the material's tev was recently deleted, so reset the uuid
                            # (proputils treats invalid id refs and empty refs the same, but this
                            # makes it easy to figure out which materials to update when tev configs
                            # are deleted, as otherwise all materials w/ no tev would update)
                            mat.brres.tevID = ""
                            self._updateMatCache(mat)
                except AttributeError:
                    # no active material, so don't worry about it
                    # (or no context provided, which means this is a final render, and in that case
                    # tev deletion makes no sense as that's not animatable)
                    pass

    def drawShadow(self, batch: gpu.types.GPUBatch):
        """Draw a "shadow" for a batch on areas of the screen that haven't yet been written to.

        This effectively makes it so that in these areas, the destination color (if referenced in
        an operation such as blending) will just be the source color instead.

        (This was used for the removed "Ignore Background" feature, taken out because it's not
        actually that useful and creates artifacts when low-opacity objects are drawn then blended
        over by others)
        """
        # disable blending & logic op, as the whole point is to ignore the current dst color
        bgl.glDisable(bgl.GL_BLEND)
        bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
        bgl.glStencilFunc(bgl.GL_NOTEQUAL, True, 0xFF)
        bgl.glColorMask(True, True, True, False)
        bgl.glDepthMask(False)
        batch.draw(self.shader)
        bgl.glDepthMask(True)
        bgl.glColorMask(True, True, True, True)
        bgl.glStencilFunc(bgl.GL_ALWAYS, True, 0xFF)

    def enableBlend(self, equation: int):
        """Enable blending and specify an equation for color.

        Alpha equation may differ depending on render settings.
        """
        bgl.glEnable(bgl.GL_BLEND)
        if self.noTransparentOverwrite:
            bgl.glBlendEquationSeparate(equation, bgl.GL_MAX)
        else:
            bgl.glBlendEquation(equation)

    def disableBlend(self):
        """Effectively disable color blending.

        Note that alpha blending may technically still be enabled if preventing transparent
        overwrites is enabled.
        """
        if self.noTransparentOverwrite:
            bgl.glEnable(bgl.GL_BLEND)
            bgl.glBlendEquationSeparate(bgl.GL_FUNC_ADD, bgl.GL_MAX)
            bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ZERO)
        else:
            bgl.glDisable(bgl.GL_BLEND)


    def draw(self, projectionMtx: Matrix, viewMtx: Matrix):
        """Draw the current BRRES scene to the active framebuffer."""
        self.shader.bind()
        if self.previewMode:
            # for preview mode, bgl isn't allowed, so keep things simple w/ a forced depth test
            gpu.state.depth_test_set('LESS_EQUAL')
            # also force the drawn alpha values to be 1, since blending gets weird
            self.shader.uniform_bool("forceOpaque", [True])
        else:
            self.shader.uniform_bool("forceOpaque", [False])
            # stencil buffer is used to determine which fragments have had values written to them
            # all 0 at first, and then set to 1 on writes
            # (used for "ignore background" functionality)
            bgl.glEnable(bgl.GL_STENCIL_TEST)
            bgl.glStencilFunc(bgl.GL_ALWAYS, True, 0xFF)
            bgl.glStencilOp(bgl.GL_REPLACE, bgl.GL_REPLACE, bgl.GL_REPLACE)
        # get list of draw calls to iterate through
        # each item in this list has an object info, material, and batch for drawing
        # this is sorted based on render group, draw priority, & material name,
        # which is why we have to use this rather than just iterating through self.objInfo
        objects = self.objects
        if self.previewMode:
            objects = {n: o for n, o in objects.items() if n != "Floor"}
        drawCalls = [(o, m, b) for o in reversed(objects.values()) for m, b in o.batches.items()]
        drawCalls.sort(key=lambda v: (
            v[1] is not None and v[1].mat.isXlu,
            v[0].drawPrio,
            v[1].mat.name if v[1] is not None else ""
        ))
        for objInfo, matInfo, batch in drawCalls:
            mvMtx = viewMtx @ objInfo.matrix
            self.shader.uniform_bool("isConstAlphaWrite", [False])
            self.shader.uniform_float("modelViewProjectionMtx", projectionMtx @ mvMtx)
            self.shader.uniform_float("normalMtx", mvMtx.to_3x3().inverted_safe().transposed())
            self.shader.uniform_block("mesh", objInfo.ubo)
            # load material data
            if matInfo is not None:
                self.shader.uniform_block("material", matInfo.ubo)
                shaderMat = matInfo.mat
                if self.previewMode:
                    for i, tex in enumerate(shaderMat.textures):
                        if tex.hasImg:
                            self.shader.uniform_sampler(f"image{i}", self.images[tex.imgName])
                else:
                    # textures
                    for i, tex in enumerate(shaderMat.textures):
                        if tex.hasImg:
                            bgl.glActiveTexture(bgl.GL_TEXTURE0 + i)
                            tex.bind(*self.images[tex.imgName])
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
                            self.enableBlend(bgl.GL_FUNC_SUBTRACT)
                            bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE)
                        else:
                            self.enableBlend(bgl.GL_FUNC_ADD)
                            bgl.glBlendFunc(shaderMat.blendSrcFac, shaderMat.blendDstFac)
                    else:
                        self.disableBlend()
                        if shaderMat.enableBlendLogic:
                            bgl.glDisable(bgl.GL_BLEND)
                            bgl.glEnable(bgl.GL_COLOR_LOGIC_OP)
                            bgl.glLogicOp(shaderMat.blendLogicOp)
                        else:
                            bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
            else:
                self.shader.uniform_block("material", MaterialInfo.EMPTY_UBO)
                if not self.previewMode:
                    bgl.glDisable(bgl.GL_STENCIL_TEST)
                    bgl.glDisable(bgl.GL_BLEND)
                    bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
                    bgl.glDisable(bgl.GL_DITHER)
                    bgl.glDisable(bgl.GL_CULL_FACE)
                    bgl.glEnable(bgl.GL_DEPTH_TEST)
            # draw
            batch.draw(self.shader)
            # write constant alpha if enabled (must be done after blending, hence 2 draw calls)
            if not self.previewMode and matInfo and matInfo.mat.enableConstAlpha:
                bgl.glDisable(bgl.GL_BLEND)
                bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
                bgl.glDisable(bgl.GL_DITHER)
                self.shader.uniform_bool("isConstAlphaWrite", [True])
                bgl.glColorMask(False, False, False, True)
                batch.draw(self.shader)
                bgl.glColorMask(True, True, True, True)
        if self.previewMode:
            gpu.state.depth_test_set('NONE')
        else:
            bgl.glDisable(bgl.GL_BLEND)
            bgl.glDisable(bgl.GL_COLOR_LOGIC_OP)
            bgl.glDisable(bgl.GL_DITHER)
            bgl.glDisable(bgl.GL_CULL_FACE)
            bgl.glDisable(bgl.GL_DEPTH_TEST)


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
        self.mainRenderer = MainBRRESRenderer(self.is_preview)
        self.backgroundColor = self._getWorldColor()
        # shader & batch
        verts = {"position": [[-1, -1], [1, -1], [-1, 1], [1, 1]]}
        idcs = [[0, 1, 2], [3, 2, 1]]
        self.shader = self._compileShader()
        self.batch: gpu.types.GPUBatch = batch_for_shader(self.shader, 'TRIS', verts, indices=idcs)
        # offscreen
        self.offscreen: gpu.types.GPUOffScreen = None
        # dimensions
        self._updateDims((1, 1))

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
    def _getWorldColor(cls):
        """Get the color of the active world in Blender, converted to the SRGB color space."""
        return np.array(bpy.context.scene.world.color[:3]) ** .4545

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
            self.mainRenderer.delete()
            self.offscreen.free()
        except AttributeError:
            pass

    def drawScene(self, projectionMtx: Matrix, viewMtx: Matrix, doAlpha = False):
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
            fb.clear(color=(*self.backgroundColor, 0), depth=1, stencil=0)
            self.mainRenderer.draw(projectionMtx, viewMtx)
        # pass 2: post-processing
        bgl.glBindFramebuffer(bgl.GL_FRAMEBUFFER, activeFb[0]) # pylint: disable=unsubscriptable-object
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.offscreen.color_texture)
        self.shader.bind()
        self.shader.uniform_bool("doAlpha", [doAlpha])
        self.batch.draw(self.shader)

    def drawPreview(self, projectionMtx: Matrix, viewMtx: Matrix):
        """Draw the current BRRES scene in preview mode (no bgl) to the active framebuffer."""
        # pass 1: main rendering
        with self.offscreen.bind():
            fb: gpu.types.GPUFrameBuffer = gpu.state.active_framebuffer_get()
            fb.clear(color=(*self.backgroundColor, 0), depth=1)
            self.mainRenderer.draw(projectionMtx, viewMtx)
        # pass 2: post-processing
        self.shader.bind()
        self.shader.uniform_sampler("tex", self.offscreen.texture_color)
        self.shader.uniform_bool("doAlpha", [True])
        self.batch.draw(self.shader)

    def render(self, depsgraph: bpy.types.Depsgraph):
        self.mainRenderer.update(depsgraph, isFinal=True)
        scene = depsgraph.scene
        render = scene.render
        scale = render.resolution_percentage / 100
        dims = (int(render.resolution_x * scale), int(render.resolution_y * scale))
        self._updateDims(dims)
        result = self.begin_result(0, 0, *dims, layer=depsgraph.view_layer.name)
        isTransparentRender = render.film_transparent and not self.is_preview
        if isTransparentRender or (self.is_preview and not self.mainRenderer.isPreviewWithFloor()):
            self.backgroundColor = (0, 0, 0)
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
                self.drawPreview(projectionMtx, viewMtx)
            else:
                self.drawScene(projectionMtx, viewMtx, isTransparentRender)
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

        self.mainRenderer.update(depsgraph, context)
        dims = (context.region.width, context.region.height)
        if dims != self.dims:
            self._updateDims(dims)
        if depsgraph.id_type_updated('WORLD'):
            self.backgroundColor = self._getWorldColor()
        # this makes response immediate for some updates that otherwise result in a delayed draw
        # (for instance, using the proputils collection move operators)
        self.tag_redraw()

        # pr.disable()
        # with open(LOG_PATH, "w", encoding="utf-8") as logFile:
        #     ps = pstats.Stats(pr, stream=logFile).sort_stats(SortKey.CUMULATIVE)
        #     ps.print_stats()

    def view_draw(self, context, depsgraph):
        self.drawScene(context.region_data.window_matrix, context.region_data.view_matrix)


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
