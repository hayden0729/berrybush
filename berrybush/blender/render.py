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
from .shaderstruct import (
    ShaderBool, ShaderInt, ShaderFloat, ShaderVec, ShaderArr, ShaderMat, ShaderStruct
)
from .material import ColorRegSettings, IndTransform, LightChannelSettings
from .texture import TexSettings, TextureTransform
from .tev import TevStageSettings
from ..wii import gx, tex0, transform as tf


# CURRENT SHADER APPROACH (ubershader vs dynamic):
# one vertex shader & fragment shader compiled for everything, taking material info through a ubo
# this has good info & links about this issue:
# https://community.khronos.org/t/ubershader-and-branching-cost/108571


TextureManagerT = TypeVar("TextureManagerT", bound="TextureManager")
MaterialManagerT = TypeVar("MaterialManagerT", bound="MaterialManager")


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
    constColors = ShaderMat(ShaderFloat, gx.MAX_TEV_CONST_COLORS)
    outputColors = ShaderMat(ShaderFloat, gx.MAX_TEV_STAND_COLORS + 1)
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


class TextureManager(ABC):

    @abstractmethod
    def updateTexture(self, tex: ShaderTexture):
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
        self._textures: dict[ShaderTexture, tuple[int, int]] = {}
        """OpenGL bindcode & mipmap count for each texture."""
        self._images: dict[str, list[bgl.Buffer]] = {}
        """Data buffer for each mipmap of each image (original included)."""
        self._usedTextures: set[ShaderTexture] = set()
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

    def _getTexture(self, tex: ShaderTexture):
        """Get the bindcode and mipmap count for a texture, updating if nonexistent."""
        try:
            return self._textures[tex]
        except KeyError:
            self.updateTexture(tex)
            return self._textures[tex]

    def updateTexture(self, tex: ShaderTexture):
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

    def bindTexture(self, texture: ShaderTexture):
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
        self._textures: dict[ShaderTexture, gpu.types.GPUTexture] = {}
        """GPUTexture for each texture."""
        self._images: dict[str, gpu.types.Buffer] = {}
        """GPU data buffer for each image."""
        self._usedTextures: set[ShaderTexture] = set()
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

    def getTexture(self, tex: ShaderTexture):
        """Get the GPUTexture corresponding to a texture, updating if nonexistent."""
        self._usedTextures.add(tex)
        try:
            return self._textures[tex]
        except KeyError:
            self.updateTexture(tex)
            return self._textures[tex]

    def updateTexture(self, tex: ShaderTexture):
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
            renderTex: gpuTex
            for renderTex, gpuTex in self._textures.items()
            if renderTex in self._usedTextures
        }
        self._usedTextures.clear()

    def delete(self):
        pass


class RenderMaterial:
    """Maintains a ShaderMaterial and UBO holding its data."""

    _EMPTY_UBO_BYTES = b"\x00" * ShaderMaterial.getSize()
    EMPTY_UBO = gpu.types.GPUUniformBuf(_EMPTY_UBO_BYTES)

    def __init__(self):
        self._ubo = gpu.types.GPUUniformBuf(self._EMPTY_UBO_BYTES)
        self._shaderMat = ShaderMaterial()

    @property
    def ubo(self):
        return self._ubo

    @property
    def shaderMat(self):
        return self._shaderMat

    def updateAnimation(self, mat: bpy.types.Material):
        """Update this material's animatable settings based on a Blender material."""
        brres = mat.brres
        shaderMat = self.shaderMat
        # color registers
        shaderMat.setColorRegs(brres.colorRegs)
        for lc, rlc in zip(brres.lightChans, shaderMat.lightChans):
            rlc.setRegs(lc)
        # texture matrices & active texture images
        tfGen = brres.miscSettings.getTexTransformGen()
        for tex, rTex in zip(brres.textures, shaderMat.textures):
            rTex.setMtx(tex.transform, tfGen)
            if rTex.imgSlot != tex.activeImgSlot:
                rTex.imgSlot = tex.activeImgSlot
                img = tex.activeImg
                if img is not None:
                    rTex.hasImg = True
                    rTex.imgName = img.name
                else:
                    rTex.hasImg = False
        shaderMat.setIndMtcs(brres.indSettings.transforms, tfGen)
        self.ubo.update(self.shaderMat.pack())

    def update(self, mat: bpy.types.Material):
        """Update this RenderMaterial based on a Blender material."""
        brres = mat.brres
        shaderMat = self.shaderMat
        shaderMat.name = mat.name
        # tev settings
        try:
            tev = bpy.context.scene.brres.tevConfigs[brres.tevID]
            shaderMat.colorSwaps = tuple(
                tuple(enumVal(colorSwap, component) for component in "rgba")
                for colorSwap in tev.colorSwaps
            )
            enabledStages = tuple(stage for stage in tev.stages if not stage.hide)
            shaderMat.numStages = len(enabledStages)
            shaderMat.stages = tuple(
                ShaderTevStage.fromStageSettings(stage) for stage in enabledStages
            )
            indTexSlots = tev.indTexSlots
        except KeyError:
            shaderMat.numStages = 0
            indTexSlots = (1, ) * gx.MAX_INDIRECTS
        # textures
        tfGen = brres.miscSettings.getTexTransformGen()
        shaderMat.numTextures = len(brres.textures)
        shaderMat.textures = tuple(
            ShaderTexture.fromTexSettings(texture, tfGen) for texture in brres.textures
        )
        # indirect textures
        rInds: list[ShaderIndTex] = []
        for texSlot, ind in zip(indTexSlots, brres.indSettings.texConfigs):
            coordScale = tuple(int(s[4:]) for s in (ind.scaleU, ind.scaleV))
            rInd = ShaderIndTex()
            rInd.texIdx = texSlot - 1
            rInd.mode = enumVal(ind, "mode")
            rInd.lightIdx = ind.lightSlot - 1
            rInd.coordScale = coordScale
            rInds.append(rInd)
        shaderMat.inds = tuple(rInds)
        shaderMat.setIndMtcs(brres.indSettings.transforms, tfGen)
        # color regs
        shaderMat.setColorRegs(brres.colorRegs)
        # light channels
        shaderMat.lightChans = tuple(
            ShaderLightChan.fromLightChanSettings(lightChan) for lightChan in brres.lightChans
        )
        # alpha settings
        alphaSettings = brres.alphaSettings
        shaderMat.enableBlend = alphaSettings.enableBlendOp
        shaderMat.blendSubtract = alphaSettings.blendOp == 'SUBTRACT'
        shaderMat.blendSrcFac = BLEND_FACS[alphaSettings.blendSrcFactor]
        shaderMat.blendDstFac = BLEND_FACS[alphaSettings.blendDstFactor]
        shaderMat.enableBlendLogic = alphaSettings.enableLogicOp
        shaderMat.blendLogicOp = BLEND_LOGIC_OPS[alphaSettings.logicOp]
        shaderMat.enableDither = alphaSettings.enableDither
        shaderMat.blendUpdateColorBuffer = alphaSettings.enableColorUpdate
        shaderMat.blendUpdateAlphaBuffer = alphaSettings.enableAlphaUpdate
        shaderMat.enableCulling = alphaSettings.cullMode != 'NONE'
        if shaderMat.enableCulling:
            shaderMat.cullMode = CULL_MODES[alphaSettings.cullMode]
        shaderMat.isXlu = alphaSettings.isXlu
        shaderMat.alphaTestVals = tuple(alphaSettings.testVals)
        shaderMat.alphaTestComps = tuple(
            enumVal(alphaSettings, f"testComp{i + 1}") for i in range(2)
        )
        shaderMat.alphaTestLogic = enumVal(alphaSettings, "testLogic")
        shaderMat.alphaTestEnable = True
        shaderMat.enableConstAlpha = alphaSettings.enableConstVal
        shaderMat.constAlpha = alphaSettings.constVal
        # depth settings
        depthSettings = brres.depthSettings
        shaderMat.enableDepthTest = depthSettings.enableDepthTest
        shaderMat.depthFunc = DEPTH_FUNCS[depthSettings.depthFunc]
        shaderMat.enableDepthUpdate = depthSettings.enableDepthUpdate
        self.ubo.update(self.shaderMat.pack())


class MaterialManager(ABC, Generic[TextureManagerT]):

    @abstractmethod
    def getMaterial(self, mat: bpy.types.Material) -> RenderMaterial:
        """Get the RenderMaterial for a Blender material, updating if nonexistent."""

    @abstractmethod
    def popInvalidMaterials(self) -> list[RenderMaterial]:
        """Remove & return all RenderMaterials that lack associated Blender materials."""

    @abstractmethod
    def updateMaterialsUsingTevConfig(self, tevId: str):
        """Update all RenderMaterials that use some TEV config (referenced by UUID)."""

    @abstractmethod
    def updateMaterialsUsingInvalidTevIds(self, validIds: set[str]):
        """Update all RenderMaterials that use a TEV config not found in the given set."""

    @abstractmethod
    def processDepsgraphUpdate(self, update: bpy.types.DepsgraphUpdate):
        """Update a RenderMaterial from a DepsgraphUpdate.
        
        (Update is only performed if the RenderMaterial already exists; new ones are not created)
        """


class StandardMaterialManager(MaterialManager[TextureManagerT]):

    def __init__(self, textureManager: TextureManagerT, assumeOpaqueMats: bool = False):
        self._textureManager = textureManager
        self._materials: dict[str, RenderMaterial] = {}
        """RenderMaterial for each Blender material."""
        self._assumeOpaqueMats = assumeOpaqueMats
        """If enabled, all materials w/o blending have constant alpha enabled & set to 1."""

    def getMaterial(self, mat: bpy.types.Material) -> RenderMaterial:
        try:
            return self._materials[mat.name]
        except KeyError:
            self._updateMaterial(mat)
            return self._materials[mat.name]

    def popInvalidMaterials(self):
        invalid: list[RenderMaterial] = []
        for blendMatName in tuple(self._materials): # tuple() so removal doesn't mess w/ iteration
            if blendMatName not in bpy.data.materials:
                invalid.append(self._materials.pop(blendMatName))
        return invalid

    def updateMaterialsUsingTevConfig(self, tevId: str):
        for blendMatName, shaderMat in self._materials.items():
            blendMat = bpy.data.materials[blendMatName]
            if blendMat.brres.tevID == tevId:
                shaderMat.update(blendMat)

    def updateMaterialsUsingInvalidTevIds(self, validIds: set[str]):
        for blendMatName, shaderMat in self._materials.items():
            blendMat = bpy.data.materials[blendMatName]
            if blendMat.brres.tevID not in validIds:
                shaderMat.update(blendMat)

    def processDepsgraphUpdate(self, update: bpy.types.DepsgraphUpdate):
        mat = update.id
        if isinstance(mat, bpy.types.Material) and mat.name in self._materials:
            if update.is_updated_shading and update.is_updated_geometry:
                # this indicates some material property was changed by the user (not animation)
                self._updateMaterial(mat)
            else:
                self._updateMaterialAnimation(mat)

    def _updateMaterial(self, mat: bpy.types.Material):
        """Update the RenderMaterial for a Blender material, creating if nonexistent."""
        if mat.name not in self._materials:
            self._materials[mat.name] = RenderMaterial()
        renderMat = self._materials[mat.name]
        renderMat.update(mat)
        shaderMat = renderMat.shaderMat
        for tex in shaderMat.textures:
            self._textureManager.updateTexture(tex)
        if self._assumeOpaqueMats and not shaderMat.enableBlend:
            shaderMat.enableConstAlpha = True
            shaderMat.constAlpha = 1

    def _updateMaterialAnimation(self, mat: bpy.types.Material):
        """Update animation data for the RenderMaterial of a Blender material if it exists."""
        try:
            renderMat = self._materials[mat.name]
            renderMat.updateAnimation(mat)
            for tex in renderMat.shaderMat.textures:
                self._textureManager.updateTexture(tex)
        except KeyError:
            pass


class RenderObject:

    _EMPTY_UBO_BYTES = b"\x00" * ShaderMesh.getSize()

    def __init__(self):
        self.batches: dict[RenderMaterial, BatchInfo] = {}
        self.drawPrio = 0
        self.matrix: Matrix = Matrix.Identity(4)
        self.usedAttrs: set[str] = set()
        self.shaderMesh = ShaderMesh()
        self.ubo = gpu.types.GPUUniformBuf(self._EMPTY_UBO_BYTES)


class ObjectManager(Generic[MaterialManagerT]):

    def __init__(self, materialManager: MaterialManagerT):
        self._materialManager = materialManager
        self._objects: dict[str, RenderObject] = {}
        """Object for each object."""

    def getDrawCalls(self):
        """List of things drawn by this renderer.
        
        Each item has a RenderObject, RenderMaterial, and BatchInfo for drawing,
        sorted as sorted on the Wii.
        """
        objects = reversed(self._objects.values())
        drawCalls = [(o, m, b) for o in objects for m, b in o.batches.items()]
        drawCalls.sort(key=lambda v: (
            v[1] is not None and v[1].shaderMat.isXlu,
            v[0].drawPriority,
            v[1].shaderMat.name if v[1] else ""
        ))
        return drawCalls

    def addNewAndRemoveUnusedObjects(self, depsgraph: bpy.types.Depsgraph,
                                     context: bpy.types.Context = None):
        """Add new objects to this manager from depsgraph & context data, and remove RenderObjects
        without corresponding Blender objects.
        
        Aside from potential removal, existing RenderObjects are not updated."""
        visibleObjects = set(depsgraph.objects)
        if context:
            # determine which objects are visible
            vl = context.view_layer
            vp = context.space_data
            visibleObjects = {
                obj
                for obj in visibleObjects
                if obj.original.visible_get(view_layer=vl, viewport=vp)
            }
        names = {obj.name for obj in visibleObjects}
        for obj in visibleObjects:
            if obj.name not in self._objects:
                self._updateObject(obj)
        self._objects = {n: info for n, info in self._objects.items() if n in names}

    def updateObjectsUsingMaterial(self, renderMat: RenderMaterial, depsgraph: bpy.types.Depsgraph):
        """Update all objects in this manager that use some RenderMaterial."""
        for blendObjName, renderObject in self._objects.items():
            if renderMat in renderObject.batches:
                self._updateObject(depsgraph.objects[blendObjName])

    def processDepsgraphUpdate(self, update: bpy.types.DepsgraphUpdate):
        """Update a RenderObject from a DepsgraphUpdate.
        
        (Update is only performed if the RenderObject already exists; new ones are not created)
        """
        obj = update.id
        if isinstance(obj, bpy.types.Object) and obj.name in self._objects:
            if update.is_updated_transform:
                self._updateObjectMatrix(obj)
            if update.is_updated_geometry:
                self._updateObject(obj)

    def _getBrresLayerNames(self, mesh: bpy.types.Mesh) -> tuple[list[str], list[str]]:
        """Get the BRRES color & UV attribute names for a mesh."""
        return (mesh.brres.meshAttrs.clrs, mesh.brres.meshAttrs.uvs)

    def _updateObjectMatrix(self, obj: bpy.types.Object):
        """Update a RenderObject's matrix from a Blender object.
        
        If the Blender object has no corresponding RenderObject, do nothing.
        """
        renderObject = self._objects.get(obj.name)
        if renderObject:
            renderObject.matrix = obj.matrix_world.copy()

    def _updateObject(self, obj: bpy.types.Object):
        """Update a RenderObject from a Blender object, or create if nonexistent.

        If the Blender object has no mesh, add nothing and delete its RenderObject if it exists.
        """
        # get mesh (and delete if object has none and there's one in the cache)
        try:
            mesh: bpy.types.Mesh = obj.to_mesh()
        except RuntimeError: # object doesn't have geometry (it's a camera, light, etc)
            if obj.name in self._objects:
                del self._objects[obj.name]
            return
        brres = obj.original.data.brres if obj.original.type == 'MESH' else None
        # get object info/create if none exists
        try:
            renderObject = self._objects[obj.name]
        except KeyError: # add object to cache if not in it yet
            self._objects[obj.name] = renderObject = RenderObject()
            self._updateObjectMatrix(obj)
        # set draw priority
        renderObject.drawPrio = brres.drawPrio if brres and brres.enableDrawPrio else 0
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
        renderObject.batches.clear()
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
            renderMat = self._materialManager.getMaterial(mat)
            idcs = loopIdcs[np.isin(matLoopIdcs, idcs)]
            renderObject.batches[renderMat] = BatchInfo('TRIS', attrs, idcs)
        if np.any(noMat):
            idcs = loopIdcs[noMat]
            renderObject.batches[None] = BatchInfo('TRIS', attrs, idcs)
        obj.to_mesh_clear()
        # set constant vals for unprovided attributes (or -1 if provided)
        # constant val is usually 0, except for previews, where colors get 1
        usedAttrs = set(attrs)
        if renderObject.usedAttrs != usedAttrs:
            renderObject.usedAttrs = usedAttrs
            m = renderObject.shaderMesh
            m.colors = tuple(-1 if f"color{i}" in attrs else 1 for i in range(gx.MAX_CLR_ATTRS))
            m.uvs = tuple(-1 if f"uv{i}" in attrs else 0 for i in range(gx.MAX_UV_ATTRS))
            renderObject.ubo.update(m.pack())


class BatchInfo:
    """Data necessary to create a GPU batch, independent from a shader."""

    def __init__(self, batchType: str, content: dict[str, np.ndarray], indices: np.ndarray):
        self._batchType = batchType
        self._content = content
        self._indices = indices

    def forShader(self, shader: gpu.types.GPUShader):
        """Create a GPUBatch for a shader from this BatchInfo.

        (Note: Faster than gpu_extras.batch.batch_for_shader(), as unused data isn't filled in)
        """
        # vboFormat = gpu.types.GPUVertFormat()
        # attrInfo = {name: int(attrType[-1]) for name, attrType in shader.attrs_info_get()}
        # for name, attrLen in attrInfo.items():
        #     vboFormat.attr_add(id=name, comp_type='F32', len=attrLen, fetch_mode='FLOAT')
        vboFormat = shader.format_calc()
        vboLen = len(next(iter(self._content.values())))
        vbo = gpu.types.GPUVertBuf(vboFormat, vboLen)
        # for name, attrLen in attrInfo.items():
        #     try:
        #         data = content[name]
        #     except KeyError:
        #         data = np.empty((vboLen, attrLen))
        #     vbo.attr_fill(name, data)
        for name, data in self._content.items():
            vbo.attr_fill(name, data)
        ibo = gpu.types.GPUIndexBuf(type=self._batchType, seq=self._indices)
        return gpu.types.GPUBatch(type=self._batchType, buf=vbo, elem=ibo)


class BatchManager:

    def __init__(self, shader: gpu.types.GPUShader):
        self._shader = shader
        self._batches: dict[BatchInfo, gpu.types.GPUBatch] = {}

    def getBatch(self, batchInfo: BatchInfo):
        """Get a batch for a BatchInfo, generating one if it doesn't yet exist in this manager."""
        try:
            return self._batches[batchInfo]
        except KeyError:
            batch = batchInfo.forShader(self._shader)
            self._batches[batchInfo] = batch
            return batch

    def removeBatch(self, batchInfo: BatchInfo):
        """Remove the batch for a BatchInfo from this manager. Do nothing if it doesn't exist."""
        try:
            del self._batches[batchInfo]
        except KeyError:
            pass


class BrresRenderer(ABC, Generic[TextureManagerT]):
    """Renders a Blender BRRES scene."""

    def __init__(self):
        self.shader: gpu.types.GPUShader = None
        self._textureManager: TextureManagerT = None
        self._materialManager: MaterialManager[TextureManagerT] = None
        self._objectManager: ObjectManager[StandardMaterialManager[TextureManagerT]] = None
        self._batchManager: BatchManager = None

    @classmethod
    def _compileShader(cls) -> gpu.types.GPUShader:
        """Compile & return the main BRRES shader.
        
        (Must be called outside of init, or else there are issues w/ material previews, which are
        my archnemesis at this point!!)
        """
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

    def _getBrresLayerNames(self, mesh: bpy.types.Mesh) -> tuple[list[str], list[str]]:
        """Get the BRRES color & UV attribute names for a mesh."""
        return (mesh.brres.meshAttrs.clrs, mesh.brres.meshAttrs.uvs)

    def delete(self):
        """Clean up resources managed by this renderer."""
        self._textureManager.delete()

    def update(self, depsgraph: bpy.types.Depsgraph, context: bpy.types.Context = None):
        """Update this renderer's settings from a Blender depsgraph & optional context.

        If no context is provided, then the settings will be updated for new & deleted Blender
        objects, but changes to existing ones will be ignored.
        """
        if not self.shader:
            self.shader = self._compileShader()
            self._batchManager = BatchManager(self.shader)
        # remove deleted stuff & add new stuff
        self._objectManager.addNewAndRemoveUnusedObjects(depsgraph, context)
        isMatUpdate = depsgraph.id_type_updated('MATERIAL')
        if isMatUpdate:
            # for all deleted (or renamed) materials, remove them & regenerate relevant objects
            for renderMat in self._materialManager.popInvalidMaterials():
                self._objectManager.updateObjectsUsingMaterial(renderMat, depsgraph)
        # update modified stuff
        tevConfigs = context.scene.brres.tevConfigs if context else ()
        validTevIds = {t.uuid for t in tevConfigs} | {""} # for tev config deletion detection
        for update in depsgraph.updates:
            updateId = update.id
            if isinstance(updateId, bpy.types.Object):
                self._objectManager.processDepsgraphUpdate(update)
            elif isinstance(updateId, bpy.types.Material):
                self._materialManager.processDepsgraphUpdate(update)
            elif isinstance(updateId, bpy.types.Image):
                # material updates can sometimes trigger image updates for some reason,
                # so make sure this is an actual image update
                if not isMatUpdate or not update.is_updated_shading:
                    self._textureManager.updateTexturesUsingImage(updateId)
            elif isinstance(updateId, bpy.types.Scene) and update.is_updated_geometry:
                if update.is_updated_geometry:
                    # this implies a tev update, so update all materials that use the active tev
                    # it could also mean a tev was deleted, so also update mats w/ invalid tev ids
                    try:
                        activeTev = context.active_object.active_material.brres.tevID
                        self._materialManager.updateMaterialsUsingTevConfig(activeTev)
                        # if a tev id isn't in the set, it means the material's tev was recently
                        # deleted, so reset uuid (proputils treats invalid id refs and empty refs
                        # the same, but this allows figuring out which materials to update when
                        # configs are deleted)
                        self._materialManager.updateMaterialsUsingInvalidTevIds(validTevIds)
                    except AttributeError:
                        # no active material, so don't worry about it
                        # (or no context provided, which means this is a final render, and in that
                        # case tev deletion makes no sense as that's not animatable)
                        pass

    def draw(self, projectionMtx: Matrix, viewMtx: Matrix):
        """Draw the current BRRES scene to the active framebuffer."""
        self.preDraw()
        for (renderObject, renderMat, batchInfo) in self._objectManager.getDrawCalls():
            batch = self._batchManager.getBatch(batchInfo)
            self.processDrawCall(viewMtx, projectionMtx, renderObject, renderMat, batch)
        self.postDraw()
        self._textureManager.removeUnused()

    @abstractmethod
    def processDrawCall(self, viewMtx: Matrix, projectionMtx: Matrix, renderObject: RenderObject,
                        renderMat: RenderMaterial, batch: gpu.types.GPUBatch):
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
        self._noTransparentOverwrite = noTransparentOverwrite
        self._textureManager = BglTextureManager()
        self._materialManager = StandardMaterialManager(self._textureManager, assumeOpaqueMats)
        self._objectManager = ObjectManager(self._materialManager)

    def preDraw(self):
        self.shader.bind()
        self.shader.uniform_bool("forceOpaque", [False])
        # stencil buffer is used to determine which fragments have had values written to them
        # all 0 at first, and then set to 1 on writes
        # (used for "ignore background" functionality)
        bgl.glEnable(bgl.GL_STENCIL_TEST)
        bgl.glStencilFunc(bgl.GL_ALWAYS, True, 0xFF)
        bgl.glStencilOp(bgl.GL_REPLACE, bgl.GL_REPLACE, bgl.GL_REPLACE)

    def processDrawCall(self, viewMtx: Matrix, projectionMtx: Matrix, renderObject: RenderObject,
                        renderMat: RenderMaterial, batch: gpu.types.GPUBatch):
        mvMtx = viewMtx @ renderObject.matrix
        self.shader.uniform_bool("isConstAlphaWrite", [False])
        self.shader.uniform_float("modelViewProjectionMtx", projectionMtx @ mvMtx)
        self.shader.uniform_float("normalMtx", mvMtx.to_3x3().inverted_safe().transposed())
        self.shader.uniform_block("mesh", renderObject.ubo)
        # load material data
        if renderMat:
            self.shader.uniform_block("material", renderMat.ubo)
            shaderMat = renderMat.shaderMat
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
            self.shader.uniform_block("material", RenderMaterial.EMPTY_UBO)
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
        if renderMat and renderMat.shaderMat.enableConstAlpha:
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
        self._materialManager = StandardMaterialManager(self._textureManager)
        self._objectManager = ObjectManager(self._materialManager)

    def _getBrresLayerNames(self, mesh: bpy.types.Mesh):
        return (["Col"] * gx.MAX_CLR_ATTRS, ["UVMap"] * gx.MAX_UV_ATTRS)

    def preDraw(self):
        self.shader.bind()
        # since bgl isn't allowed, keep things simple w/ a forced depth test
        gpu.state.depth_test_set('LESS_EQUAL')
        # also force drawn alpha values to be 1, since blending gets weird
        self.shader.uniform_bool("forceOpaque", [True])

    def processDrawCall(self, viewMtx: Matrix, projectionMtx: Matrix, renderObject: RenderObject,
                        renderMat: RenderMaterial, batch: gpu.types.GPUBatch):
        mvMtx = viewMtx @ renderObject.matrix
        self.shader.uniform_bool("isConstAlphaWrite", [False])
        self.shader.uniform_float("modelViewProjectionMtx", projectionMtx @ mvMtx)
        self.shader.uniform_float("normalMtx", mvMtx.to_3x3().inverted_safe().transposed())
        self.shader.uniform_block("mesh", renderObject.ubo)
        # load material data
        if renderMat:
            self.shader.uniform_block("material", renderMat.ubo)
            for i, tex in enumerate(renderMat.shaderMat.textures):
                if tex.hasImg:
                    self.shader.uniform_sampler(f"image{i}", self._textureManager.getTexture(tex))
        else:
            self.shader.uniform_block("material", RenderMaterial.EMPTY_UBO)
        # draw
        batch.draw(self.shader)

    def postDraw(self):
        gpu.state.depth_test_set('NONE')


class BerryBushRenderEngine(bpy.types.RenderEngine):
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
