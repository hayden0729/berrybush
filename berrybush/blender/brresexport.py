# standard imports
from cProfile import Profile
from datetime import datetime
from functools import cache
from pathlib import Path
from pstats import SortKey, Stats
import os
from typing import Generic, TypeVar
# 3rd party imports
import bpy
from bpy_extras.io_utils import ExportHelper, axis_conversion
from mathutils import Euler, Matrix, Vector
import numpy as np
# internal imports
from .common import (
    MTX_TO_BONE, MTX_FROM_BONE, MTX_TO_BRRES, LOG_PATH,
    solidView, restoreView, limitIncludes, usedMatSlots, makeUniqueName, foreachGet, getLayerData,
    getLoopVertIdcs, getLoopFaceIdcs, getFaceMatIdcs, getPropName, drawCheckedProp,
    simplifyLayerData
)
from .material import AlphaSettings, DepthSettings, LightChannelSettings, MiscMatSettings
from .tev import TevSettings, TevStageSettings
from .texture import TexSettings
from .updater import addonVer, verStr
from .verify import verifyBRRES
from ..wii import (
    animation, binaryutils, brres, chr0, clr0, gx, mdl0, pat0, srt0, tex0, transform as tf, vis0
)


ANIM_SUBFILE_T = TypeVar("ANIM_SUBFILE_T", bound=animation.AnimSubfile)


IMG_FMTS: dict[str, type[tex0.ImageFormat]] = {
    'I4': tex0.I4,
    'I8': tex0.I8,
    'IA4': tex0.IA4,
    'IA8': tex0.IA8,
    'RGB565': tex0.RGB565,
    'RGB5A3': tex0.RGB5A3,
    'RGBA8': tex0.RGBA8,
    'CMPR': tex0.CMPR
}


IDENTITY_EULER = Euler()


def padImgData(img: bpy.types.Image, shape: tuple[int, int, int]):
    """Get an image's 3D pixel data, cropped or padded w/ 0s to some shape.

    If the image is None, an array full of 0s is returned.

    Mainly useful for dealing with mipmaps that have improper dimensions.
    """
    output = np.zeros(shape, dtype=np.float32)
    if img is not None:
        # get data & crop or pad w/ 0s if mipmap dims are improper
        curShape = (img.size[1], img.size[0], img.channels)
        imgPx = np.array(img.pixels, dtype=np.float32).reshape(curShape)
        output[:curShape[0], :curShape[1], :curShape[2]] = imgPx[:shape[0], :shape[1], :shape[2]]
    return output


class BRRESMdlExporter():

    def __init__(self, parentExporter: "BRRESExporter", rigObj: bpy.types.Object):
        settings = parentExporter.settings
        self.parentExporter = parentExporter
        self.model = mdl0.MDL0(rigObj.name)
        self.rigObj = rigObj
        self.parentExporter.res.folder(mdl0.MDL0).append(self.model)
        # generate joints
        rig: bpy.types.Armature = rigObj.data
        restorePosePosition = rig.pose_position
        if not settings.useCurrentPose:
            rig.pose_position = 'REST' # temporarily put rig in rest position (pose restored at end)
            parentExporter.depsgraph.update()
        self._hasExtraRoot = settings.forceRootBones and len(rig.bones) > 0
        self.joints: dict[str, mdl0.Joint] = {}
        self._exportJoints(rigObj)
        # generate meshes & everything they use
        self.mats: dict[str, mdl0.Material] = {}
        self.tevConfigs: dict[str, mdl0.TEVConfig] = {}
        for obj in bpy.data.objects:
            if not limitIncludes(settings.limitTo, obj):
                continue
            if settings.applyModifiers:
                obj: bpy.types.Object = obj.evaluated_get(parentExporter.depsgraph)
            parent = obj.parent
            if parent is None or parent.type != 'ARMATURE' or parent.data.name != rig.name:
                continue
            self._exportMeshObj(obj)
        # remove unused joints if enabled
        if settings.removeUnusedBones:
            self._removeUnusedJoints()
        # sort materials so arbitrary draw order (when group & prio are equal) is based on name
        self.model.mats.sort(key=lambda mat: mat.name)
        # restore current rig pose in case we changed it to ignore it
        rig.pose_position = restorePosePosition

    def _genTEVStage(self, stageSettings: TevStageSettings):
        """Generate a MDL0 TEV stage from Blender settings."""
        stage = mdl0.TEVStage()
        # selections
        stage.texIdx = stage.texCoordIdx = stageSettings.sels.texSlot - 1
        stage.alphaParams.textureSwapIdx = stageSettings.sels.texSwapSlot - 1
        stage.constColorSel = gx.TEVConstSel[stageSettings.sels.constColor]
        stage.constAlphaSel = gx.TEVConstSel[stageSettings.sels.constAlpha]
        stage.rasterSel = gx.TEVRasterSel[stageSettings.sels.rasterSel]
        stage.alphaParams.rasterSwapIdx = stageSettings.sels.rasSwapSlot - 1
        # indirect texturing
        uiInd = stageSettings.indSettings
        ind = stage.indSettings
        ind.indirectID = uiInd.slot - 1
        ind.format = gx.ColorBitSel[uiInd.fmt]
        ind.biasS, ind.biasT, ind.biasU = uiInd.enableBias
        ind.bumpAlphaComp = gx.TexCoordSel[uiInd.bumpAlphaComp]
        ind.mtxType = gx.IndMtxType[uiInd.mtxType]
        ind.mtxIdx = gx.IndMtxIdx(uiInd.mtxSlot) if uiInd.enable else gx.IndMtxIdx.NONE
        ind.wrapS = gx.IndWrapSet[uiInd.wrapU]
        ind.wrapT = gx.IndWrapSet[uiInd.wrapV]
        ind.utcLOD = uiInd.utcLOD
        ind.addPrev = uiInd.addPrev
        # color/alpha params
        modelCalcParams = (stage.colorParams, stage.alphaParams)
        uiCalcParams = (stageSettings.colorParams, stageSettings.alphaParams)
        argEnums = (gx.TEVColorArg, gx.TEVAlphaArg)
        for modelParams, uiParams, argEnum in zip(modelCalcParams, uiCalcParams, argEnums):
            modelParams.args = (argEnum[arg] for arg in uiParams.args)
            modelParams.clamp = uiParams.clamp
            modelParams.output = int(uiParams.output)
            if uiParams.compMode:
                modelParams.bias = gx.TEVBias.COMPARISON_MODE
                modelParams.op = gx.TEVOp[uiParams.compOp]
                modelParams.compareMode = gx.TEVScaleChan[uiParams.compChan]
            else:
                modelParams.bias = gx.TEVBias[uiParams.bias]
                modelParams.op = gx.TEVOp[uiParams.op]
                modelParams.scale = gx.TEVScaleChan[uiParams.scale]
        return stage

    def _exportTEVConfig(self, tevSettings: TevSettings):
        """Export a MDL0 TEV configuration from Blender settings."""
        tev = mdl0.TEVConfig()
        self.tevConfigs[tevSettings.name] = tev
        self.model.tevConfigs.append(tev)
        # color swap table
        for mdlSwap, uiSwap in zip(tev.colorSwaps, tevSettings.colorSwaps):
            mdlSwap.r = gx.ColorChannel[uiSwap.r]
            mdlSwap.g = gx.ColorChannel[uiSwap.g]
            mdlSwap.b = gx.ColorChannel[uiSwap.b]
            mdlSwap.a = gx.ColorChannel[uiSwap.a]
        # indirect selections
        indSrcs = tev.indSources
        indSrcs.texIdcs = indSrcs.texCoordIdcs = [slot - 1 for slot in tevSettings.indTexSlots]
        # stages
        tev.stages = [self._genTEVStage(s) for s in tevSettings.stages if not s.hide]
        return tev

    def _exportTex(self, parentMat: mdl0.Material, texSettings: TexSettings):
        """Export a texture into a MDL0 material from Blender settings."""
        # image
        uiImg = texSettings.activeImg
        if uiImg is not None:
            if uiImg not in self.parentExporter.images and self.parentExporter.onlyUsedImg:
                self.parentExporter.exportImg(uiImg)
        # texture
        tex = mdl0.Texture()
        parentMat.textures.append(tex)
        tex.imgName = self.parentExporter.imgName(uiImg)
        t = texSettings.transform
        tex.setSRT(t.scale, [np.rad2deg(t.rotation)], t.translation)
        try:
            tex.mapMode = mdl0.TexMapMode[texSettings.mapMode]
        except KeyError:
            mapMode, coordSlot = texSettings.mapMode.split("_")
            tex.mapMode = mdl0.TexMapMode[mapMode]
            tex.coordIdx = int(coordSlot) - 1
        tex.wrapModes = [mdl0.WrapMode[texSettings.wrapModeU], mdl0.WrapMode[texSettings.wrapModeV]]
        if uiImg is None or len(uiImg.brres.mipmaps) == 0:
            tex.minFilter = mdl0.MinFilter[texSettings.minFilter]
        else:
            minFilter = mdl0.MinFilter[texSettings.minFilter]
            mipFilter = mdl0.MinFilter[texSettings.mipFilter]
            tex.minFilter = mdl0.MinFilter(mipFilter.value * 2 + minFilter.value + 2)
        tex.magFilter = mdl0.MagFilter[texSettings.magFilter]
        tex.lodBias = texSettings.lodBias
        tex.maxAnisotropy = mdl0.MaxAnisotropy[texSettings.maxAnisotropy]
        tex.clampBias = texSettings.clampBias
        tex.texelInterpolate = texSettings.texelInterpolate
        tex.usedCam = texSettings.camSlot - 1 if texSettings.useCam else -1
        tex.usedLight = texSettings.lightSlot - 1 if texSettings.useLight else -1

    def _genLightChan(self, uiLc: LightChannelSettings):
        """Generate a MDL0 light channel based on settings from Blender."""
        lc = mdl0.LightChannel()
        ceFlags = mdl0.LightChannel.ColorEnableFlags
        # pylint: disable=unsupported-binary-operation
        lc.difColor = list(uiLc.difColor)
        lc.ambColor = list(uiLc.ambColor)
        ccFlags = mdl0.LightChannel.ColorControlFlags
        controlFlags: list[ccFlags] = []
        for controlSettings in (uiLc.colorSettings, uiLc.alphaSettings):
            control = ccFlags(0)
            if not controlSettings.difFromReg:
                control |= ccFlags.DIFFUSE_FROM_VERTEX
            if not controlSettings.ambFromReg:
                control |= ccFlags.AMBIENT_FROM_VERTEX
            if controlSettings.enableDiffuse:
                control |= ccFlags.DIFFUSE_ENABLE
            if controlSettings.diffuseMode == 'SIGNED':
                control |= ccFlags.DIFFUSE_SIGNED
            if controlSettings.diffuseMode == 'CLAMPED':
                control |= ccFlags.DIFFUSE_CLAMPED
            if controlSettings.enableAttenuation:
                control |= ccFlags.ATTENUATION_ENABLE
            if controlSettings.attenuationMode == 'SPOTLIGHT':
                control |= ccFlags.ATTENUATION_SPOTLIGHT
            lightFlags = (
                ccFlags.LIGHT_0_ENABLE,
                ccFlags.LIGHT_1_ENABLE,
                ccFlags.LIGHT_2_ENABLE,
                ccFlags.LIGHT_3_ENABLE,
                ccFlags.LIGHT_4_ENABLE,
                ccFlags.LIGHT_5_ENABLE,
                ccFlags.LIGHT_6_ENABLE,
                ccFlags.LIGHT_7_ENABLE,
            )
            for lightEnabled, flag in zip(controlSettings.enabledLights, lightFlags):
                if lightEnabled:
                    control |= flag
            controlFlags.append(control)
        lc.colorControl, lc.alphaControl = controlFlags # pylint: disable=unbalanced-tuple-unpacking
        return lc

    def _exportAlphaSettings(self, mat: mdl0.Material, alphaSettings: AlphaSettings):
        """Set a MDL0 material's alpha settings based on settings from Blender."""
        blendSettings = mat.blendSettings
        blendSettings.enableBlend = alphaSettings.enableBlendOp
        blendSettings.subtract = (alphaSettings.enableBlendOp and alphaSettings.blendOp == "-")
        blendSettings.srcFactor = gx.BlendSrcFactor[alphaSettings.blendSrcFactor]
        blendSettings.dstFactor = gx.BlendDstFactor[alphaSettings.blendDstFactor]
        blendSettings.enableLogic = alphaSettings.enableLogicOp
        blendSettings.logic = gx.BlendLogicOp[alphaSettings.logicOp]
        blendSettings.enableDither = alphaSettings.enableDither
        blendSettings.updateColor = alphaSettings.enableColorUpdate
        blendSettings.updateAlpha = alphaSettings.enableAlphaUpdate
        mat.constAlphaSettings.enable = alphaSettings.enableConstVal
        mat.constAlphaSettings.value = alphaSettings.constVal
        mat.cullMode = gx.CullMode[alphaSettings.cullMode]
        mat.renderGroup = mdl0.RenderGroup.XLU if alphaSettings.isXlu else mdl0.RenderGroup.OPA
        testSettings = mat.alphaTestSettings
        testSettings.comps = [gx.CompareOp[comp] for comp in alphaSettings.testComps]
        testSettings.values = list(alphaSettings.testVals)
        testSettings.logic = gx.AlphaLogicOp[alphaSettings.testLogic]

    def _exportDepthSettings(self, mat: mdl0.Material, depthSettings: DepthSettings):
        """Set a MDL0 material's depth settings based on settings from Blender."""
        mat.depthSettings.enable = depthSettings.enableDepthTest
        mat.depthSettings.updateDepth = depthSettings.enableDepthUpdate
        mat.depthSettings.depthOp = gx.CompareOp[depthSettings.depthFunc]

    def _exportMiscSettings(self, mat: mdl0.Material, miscSettings: MiscMatSettings):
        """Set a MDL0 material's misc settings based on settings from Blender."""
        mat.lightSet = miscSettings.lightSet - 1 if miscSettings.useLightSet else -1
        mat.fogSet = miscSettings.fogSet - 1 if miscSettings.useFogSet else -1

    def _exportMaterial(self, mat: bpy.types.Material):
        """Export a Blender material, TEV & textures included, into a MDL0 model."""
        brresMat = mdl0.Material(mat.name)
        self.mats[mat.name] = brresMat
        self.model.mats.append(brresMat)
        brresMatSettings = mat.brres
        # tev
        try:
            tevSettings = self.parentExporter.context.scene.brres.tevConfigs[brresMatSettings.tevID]
            if tevSettings.name not in self.tevConfigs:
                self._exportTEVConfig(tevSettings)
            brresMat.tevConfig = self.tevConfigs[tevSettings.name]
        except KeyError:
            pass # material has no tev
        # textures
        brresMat.mtxGen = brresMatSettings.miscSettings.getTexTransformGen()
        for uiTex in brresMatSettings.textures:
            self._exportTex(brresMat, uiTex)
        # indirect configurations
        nrmModes = {'NORMAL_MAP', 'NORMAL_MAP_SPEC'}
        for ind, uiInd in zip(brresMat.indTextures, brresMatSettings.indSettings.texConfigs):
            ind.mode = mdl0.IndTexMode[uiInd.mode]
            ind.lightIdx = uiInd.lightSlot - 1 if ind.mode in nrmModes else -1
            ind.coordScales[:] = (gx.IndCoordScalar[uiInd.scaleU], gx.IndCoordScalar[uiInd.scaleV])
        for uiSRT in brresMatSettings.indSettings.transforms:
            t = uiSRT.transform
            mdlSRT = mdl0.IndTransform()
            brresMat.indSRTs.append(mdlSRT)
            mdlSRT.setSRT(t.scale, [np.rad2deg(t.rotation)], t.translation)
        # lighting channels
        brresMat.lightChans = [self._genLightChan(lc) for lc in brresMatSettings.lightChans]
        # color registers
        brresMat.constColors = [list(c) for c in brresMatSettings.colorRegs.constant]
        brresMat.standColors = [list(c) for c in brresMatSettings.colorRegs.standard[1:]]
        # other settings
        self._exportAlphaSettings(brresMat, brresMatSettings.alphaSettings)
        self._exportDepthSettings(brresMat, brresMatSettings.depthSettings)
        self._exportMiscSettings(brresMat, brresMatSettings.miscSettings)

    def _exportAttrData(self, groupType: type[mdl0.VertexAttrGroup], layer: bpy.types.Attribute,
                        data: np.ndarray):
        """Export a MDL0 vertex attribute group based on a mesh layer's data.

        Return None & do nothing if the layer/data are None.
        """
        if layer is None:
            return None
        group = groupType(f"{layer.id_data.name}__{layer.name}")
        self.model.vertGroups[groupType].append(group)
        group.setArr(data)
        return group

    def _getParentAndApplyTransform(self, mesh: bpy.types.Mesh, obj: bpy.types.Object):
        """Get a mesh object's MDL0 joint parent (None if skinning used).

        Additionally, transform the mesh based on its matrices and any relevant coordinate system
        conversions if applicable.
        """
        singleBindJoint: mdl0.Joint = None
        modelMtx = obj.matrix_local.copy()
        if obj.parent_type == 'BONE' and obj.parent_bone in self.joints:
            # object is parented to single bone
            singleBindJoint = self.joints[obj.parent_bone]
            # object is positioned relative to bone tail, but we want relative to head, so convert
            parentLen = obj.parent.data.bones[obj.parent_bone].length
            headRel = Matrix.Translation((0, parentLen, 0))
            coordConversion = MTX_FROM_BONE.to_4x4() @ self.parentExporter.mtxBoneToBRRES.to_4x4()
            modelMtx = coordConversion @ headRel @ modelMtx
        elif not self.hasSkinning(obj):
            # object is parented straight to armature, so we need extra root for proper parenting
            singleBindJoint = self._extraRoot()
        else:
            # skinning is used, so no single-bind joint or extra transformation needed
            pass
        mesh.transform(MTX_TO_BRRES @ modelMtx)
        return singleBindJoint

    def _normalizeDeformer(self, df: mdl0.Deformer | dict[mdl0.Joint, float]):
        """Return a dict representing a deformer with its weights normalized.

        Weights of 0 are removed. If the deformer is empty or all its weights are 0, the returned
        dict will only contain a weight for the model's extra root (set to 1.0).
        """
        weightNorm = sum(df.values())
        if weightNorm == 0: # bind vertices w/o weights to extra root
            return {self._extraRoot(): 1.0}
        else:
            return {j: w / weightNorm for j, w in df.items() if w > 0}

    def _getVertDfs(self, obj: bpy.types.Object, settings: "ExportBRRES") -> list[mdl0.Deformer]:
        """Get a list with the MDL0 deformer for each vertex of an object."""
        vertDfs = [{} for _ in range(len(obj.data.vertices))]
        for vg in obj.vertex_groups:
            try:
                joint = self.joints[vg.name]
            except KeyError: # vertex group doesn't correspond to a bone
                continue
            for i, vertDf in enumerate(vertDfs):
                try:
                    vertDf[joint] = vg.weight(i)
                except RuntimeError: # vertex not in group
                    pass
        for i, df in enumerate(vertDfs):
            # normalize weights & convert dict to actual deformer
            newDf = self._normalizeDeformer(df)
            if settings.doQuantize:
                # for quantization, round weights based on number of steps and then re-normalize
                steps = settings.quantizeSteps
                newDf = self._normalizeDeformer({j: round(w * steps) for j, w in newDf.items()})
            vertDfs[i] = mdl0.Deformer(newDf)
        return vertDfs

    def _exportMeshObj(self, obj: bpy.types.Object):
        """Export a Blender mesh object, material & all included.

        Do nothing if the object doesn't support a mesh (i.e., it's a light, camera, etc).
        """
        # generate mesh
        try:
            depsgraph = self.parentExporter.depsgraph
            mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        except RuntimeError: # object type doesn't support meshes
            return
        # get parent joint if applicable & apply transform
        singleBindJoint = self._getParentAndApplyTransform(mesh, obj)
        # get mesh info, including positions & normals
        settings = self.parentExporter.settings
        mesh.calc_normals_split()
        mesh.calc_loop_triangles()
        triLoopIdcs = foreachGet(mesh.loop_triangles, "loops", 3, np.integer)[:, ::-1].flatten()
        loopVertIdcs = getLoopVertIdcs(mesh)
        loopFaceIdcs = getLoopFaceIdcs(mesh)
        triMatIdcs = getFaceMatIdcs(mesh)[loopFaceIdcs][triLoopIdcs]
        psns = foreachGet(mesh.vertices, "co", 3) * settings.scale
        nrms = foreachGet(mesh.loops, "normal", 3)
        # get info for skinning, and adjust psns & normals if skinning is used
        hasSkinning = singleBindJoint is None
        vertDfs: list[mdl0.Deformer] = []
        vertDfIdcs = np.ndarray(0)
        dfIdcs = np.ndarray(0)
        if hasSkinning:
            # first, get several useful arrays for dealing with the mesh's deformers
            # vertDfs: deformer for each vertex
            # dfs: all unique deformers in vertDfs
            # vertDfIdcs: for each vert in vertDfs, index of that vert's deformer in dfs
            # dfIdcs: for each deformer in dfs, index of first vertex w/ that deformer in vertDfs
            vertDfs = self._getVertDfs(obj, settings)
            vertDfHashes = np.array([hash(df) for df in vertDfs], dtype=np.int64)
            _, dfIdcs, vertDfIdcs = np.unique(vertDfHashes, return_index=True, return_inverse=True)
            dfs: list[mdl0.Deformer] = [vertDfs[i] for i in dfIdcs]
            # then, adjust positions & normals
            # basically, vertices w/ single-joint deformers are stored relative to those joints,
            # so we have to convert them from rest pose to that relative space
            # note: this does not apply to multi-weight deformers,
            # presumably because their matrices aren't guaranteed to be invertible?
            # (for instance, imagine a deformer for 2 equally weighted joints w/ opposite rotations)
            mdl = self.model
            dfMtcs = np.array([df.mtx(mdl) if len(df) == 1 else np.identity(4) for df in dfs])
            invDfMtcs = np.array([np.linalg.inv(m) for m in dfMtcs])
            vertDfMtcs = dfMtcs[vertDfIdcs]
            invVertDfMtcs = invDfMtcs[vertDfIdcs]
            paddedPsns = np.pad(psns, ((0, 0), (0, 1)), constant_values=1) # needed for 4x4 matmul
            # https://stackoverflow.com/questions/35894631/multiply-array-of-vectors-with-array-of-matrices-return-array-of-vectors
            psns = np.einsum("ij, ijk->ik", paddedPsns, invVertDfMtcs.swapaxes(1, 2))[:, :3]
            nrms = np.einsum("ij, ijk->ik", nrms, vertDfMtcs[loopVertIdcs, :3, :3])
        # generate position group
        psns, unqPsnInv = np.unique(psns, return_inverse=True, axis=0)
        psnGroup = mdl0.PsnGroup(obj.name)
        self.model.vertGroups[mdl0.PsnGroup].append(psnGroup)
        psnGroup.setArr(psns)
        # generate normal group
        nrms = simplifyLayerData(nrms)
        nrms, unqNrmInv = np.unique(nrms, return_inverse=True, axis=0)
        nrmGroup = mdl0.NrmGroup(mesh.name)
        self.model.vertGroups[mdl0.NrmGroup].append(nrmGroup)
        nrmGroup.setArr(nrms)
        # generate color & uv groups
        try:
            # brres mesh attributes may be deleted by modifiers, so try to get from original mesh
            # & fall back on evaluated mesh's attributes in case this fails
            meshAttrs = obj.original.data.brres.meshAttrs
        except AttributeError:
            meshAttrs = mesh.brres.meshAttrs
        clrData = getLayerData(mesh, meshAttrs.clrs, isUV=False)
        uvData = getLayerData(mesh, meshAttrs.uvs, isUV=True)
        clrGroups = [self._exportAttrData(mdl0.ClrGroup, l, d) for l, d, i in clrData]
        uvGroups = [self._exportAttrData(mdl0.UVGroup, l, d) for l, d, i in uvData]
        # generate brres mesh for each material used
        for matSlot in usedMatSlots(obj, mesh):
            mat = matSlot.material
            if not mat:
                continue
            usedLoops = triLoopIdcs[triMatIdcs == matSlot.slot_index]
            # generate brres mesh
            brresMesh = mdl0.Mesh(f"{obj.name}__{mat.name}")
            self.model.meshes.append(brresMesh)
            if singleBindJoint is not None:
                brresMesh.singleBind = singleBindJoint.deformer
            brresMesh.vertGroups = {
                mdl0.PsnGroup: {0: psnGroup},
                mdl0.NrmGroup: {0: nrmGroup},
                mdl0.ClrGroup: {i: g for i, g in enumerate(clrGroups) if g is not None},
                mdl0.UVGroup: {i: g for i, g in enumerate(uvGroups) if g is not None}
            }
            # visibility joint
            boneVisSuffix = ".hide"
            boneVisSuffixLen = len(boneVisSuffix)
            if obj.animation_data:
                for fc in obj.animation_data.drivers:
                    if fc.data_path != "hide_viewport":
                        continue
                    for var in fc.driver.variables:
                        if var.type != 'SINGLE_PROP':
                            continue
                        dataPath = var.targets[0].data_path
                        if dataPath.endswith(boneVisSuffix):
                            try:
                                # cut off ".hide" to get bone
                                bone = self.rigObj.path_resolve(dataPath[:-boneVisSuffixLen])
                                if not isinstance(bone, bpy.types.Bone):
                                    continue
                            except ValueError:
                                continue # not a bone visibility path
                            brresMesh.visJoint = self.joints[bone.name]
                            break
                    if brresMesh.visJoint:
                        break
            if not brresMesh.visJoint:
                # object doesn't have visibility driver, so use extra root for visibility joint
                brresMesh.visJoint = self._extraRoot()
            # generate material
            if mat.name not in self.mats:
                self._exportMaterial(mat)
            brresMesh.mat = self.mats[mat.name]
            brresMesh.drawPrio = mesh.brres.drawPrio if mesh.brres.enableDrawPrio else 0
            # separate primitives into draw groups
            drawGroupData: list[tuple[set[int], list[np.ndarray]]] = []
            if hasSkinning:
                # for each face, go through the existing draw groups.
                # if there's a draw group found that supports this face (i.e., all the face's
                # deformers are in the draw group, or it has room to add those deformers), add the
                # face (and its deformers if necessary) to the draw group.
                # otherwise, make a new draw group and add this face & its deformers to it.
                # this approach seems a bit naive, but there are more important things to optimize
                # (e.g., triangle stripping), so seeing if this can be improved isn't a priority rn
                # (btw based on a quick code glimpse, i think this is also what brawlcrate does)
                loopDfIdcs = vertDfIdcs[loopVertIdcs]
                for face in usedLoops.reshape(-1, 3):
                    groupFound = False
                    uniqueFaceDfs = set(loopDfIdcs[face])
                    for dgDfs, dgFaces in drawGroupData:
                        newDfs = uniqueFaceDfs.difference(dgDfs)
                        if len(dgDfs) <= gx.MAX_ATTR_MTCS - len(newDfs):
                            dgDfs |= newDfs
                            dgFaces.append(face)
                            groupFound = True
                            break
                    if not groupFound:
                        newGroup = (uniqueFaceDfs, [face])
                        drawGroupData.append(newGroup)
            else:
                drawGroupData.append((set(), [usedLoops]))
            # generate primitive commands
            maxStripLen = gx.DrawTriangleStrip.maxLen()
            maxTriLen = gx.DrawTriangles.maxLen()
            for dgDfs, dgFaces in drawGroupData:
                dg = mdl0.DrawGroup()
                brresMesh.drawGroups.append(dg)
                dg.deformers = [vertDfs[dfIdcs[dfIdx]] for dfIdx in dgDfs]
                dgLoopIdcs = np.concatenate(dgFaces)
                numLoops = len(dgLoopIdcs)
                dgVertIdcs = loopVertIdcs[dgLoopIdcs]
                dgFaceIdcs = loopFaceIdcs[dgLoopIdcs]
                # set up command w/ basic vertex attrs
                # this is not used in the model, it's just used to store the attrs
                # and then it's converted to triangle strips, which are stored, for compression
                cmd = gx.DrawTriangles(numLoops)
                cmd.psns = unqPsnInv[dgVertIdcs].reshape(1, -1)
                cmd.nrms = unqNrmInv[dgLoopIdcs].reshape(1, -1)
                domains = {'POINT': dgVertIdcs, 'FACE': dgFaceIdcs}
                for attrData, cmdAttr in zip((clrData, uvData), (cmd.clrs, cmd.uvs)):
                    for (layer, data, idcs), cmdData in zip(attrData, cmdAttr):
                        if layer is not None:
                            domain = layer.domain if hasattr(layer, "domain") else 'CORNER'
                            cmdData[:] = idcs[domains.get(domain, dgLoopIdcs)]
                # then set up matrix attrs (for skinning)
                if dgDfs:
                    # we have absolute df indices, but we need relative to this draw group's df list
                    # we get this using np.searchsorted(), based on this
                    # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
                    # (accepted answer, not most upvoted; most upvoted answers the wrong question)
                    dgLoopAbsDfs = loopDfIdcs[dgLoopIdcs] # absolute df index for each loop in dg
                    dgDfs = np.array(tuple(dgDfs))
                    sortedDgDfIdcs = np.argsort(dgDfs)
                    dgLoopDfs = sortedDgDfIdcs[np.searchsorted(dgDfs[sortedDgDfIdcs], dgLoopAbsDfs)]
                    dgLoopDfs = dgLoopDfs.reshape(1, -1) # reshape for use in commands
                    # now, just put the indices (converted to addresses) in the commands
                    dgLoopPsnMtxAddrs = gx.LoadPsnMtx.idxToAddr(dgLoopDfs) // 4
                    dgLoopTexMtxAddrs = gx.LoadTexMtx.idxToAddr(dgLoopDfs) // 4
                    cmd.psnMtcs = dgLoopPsnMtxAddrs
                    for i, tex in enumerate(self.mats[mat.name].textures):
                        if tex.mapMode is not mdl0.TexMapMode.UV:
                            cmd.texMtcs[i] = dgLoopTexMtxAddrs
                # apply triangle stripping for compression
                verts, vertIdcs = np.unique(cmd.vertData, return_inverse=True, axis=0)
                strips = self._tristrip(vertIdcs.reshape(-1, 3).tolist(), maxStripLen)
                soloTris = []
                for strip in strips:
                    if len(strip) == 3:
                        soloTris += strip
                    else:
                        stripCmd = gx.DrawTriangleStrip(vertData=verts[strip])
                        dg.cmds.append(stripCmd)
                # compile isolated triangles into their own command
                # (multiple commands if too many to fit into one)
                numSoloVerts = len(soloTris)
                soloVertData = verts[soloTris]
                for vertStart in range(0, numSoloVerts, maxTriLen):
                    vertEnd = min(vertStart + maxTriLen, numSoloVerts)
                    soloCmd = gx.DrawTriangles(vertData=soloVertData[vertStart:vertEnd])
                    dg.cmds.append(soloCmd)

    def _exportJoints(self, rigObj: bpy.types.Object):
        mtcs = {bone: bone.matrix for bone in rigObj.pose.bones}
        mtcs = {bone: (mtx, mtx.inverted()) for bone, mtx in mtcs.items()}
        localScales = {} # for segment scale compensate calculations
        prevRots = {} # for euler compatibility
        for poseBone in rigObj.pose.bones:
            bone = poseBone.bone
            joint = mdl0.Joint(self.joints[bone.parent.name] if bone.parent else None, bone.name)
            self.joints[joint.name] = joint
            if joint.parent is None:
                if self.model.rootJoint is None:
                    self.model.rootJoint = joint
                else:
                    joint.parent = self._extraRoot()
            joint.isVisible = not bone.hide
            # treat no scale inheritance as segment scale compensate
            # note: on its own, disabling scale inheritance also disables inheritance beyond the
            # parent (disables grandparent, etc), which is different from segment scale compensate
            # however, getLocalSRT() (used for rest pose & animation) automatically adjusts for
            # this, as it just gets the parent-relative transform taking ssc into account
            # enabling ssc just makes it less likely that things will be inaccurate for cases where
            # data is unavoidably lost (situations w/ non-uniform scaling), and does not add to the
            # data loss whatsoever (in fact, when ssc controllers are set up properly like w/
            # import, segment scale compensate can be exported perfectly)
            # this is hard to put into words but hopefully that made sense
            joint.segScaleComp = bone.inherit_scale == 'NONE'
            # store parent-relative transform
            srt = np.array(self.parentExporter.getLocalSRT(poseBone, localScales, mtcs, prevRots))
            srt[np.isclose(srt, 0, atol=0.001)] = 0
            joint.setSRT(*srt)
        for boneName, joint in self.joints.items():
            boneSettings = rigObj.data.bones[boneName].brres
            joint.bbMode = mdl0.BillboardMode[boneSettings.bbMode]
            try:
                joint.bbParent = self.joints[boneSettings.bbParent]
            except KeyError:
                pass

    def _extraRoot(self):
        """Extra root joint to be used in case of multiple roots or objects without bone parenting.

        This creates one joint the first time it's called, and returns that same joint for
        subsequent calls. If "force root bones" is enabled and this model's corresponding armature
        has at least one bone, this simply returns the root joint (doesn't create a new one)."""
        if self._hasExtraRoot:
            return self.model.rootJoint
        self._hasExtraRoot = True
        usedBoneNames = {b.name for b in self.rigObj.data.bones}
        extraRoot = mdl0.Joint(name=makeUniqueName(self.rigObj.data.name, usedBoneNames))
        if self.model.rootJoint is not None:
            self.model.rootJoint.parent = extraRoot
        self.model.rootJoint = extraRoot
        return extraRoot

    def _removeUnusedJoints(self):
        """Remove joints that aren't used for deform/vis & aren't ancestors of those that are"""
        dfs: set[mdl0.Deformer] = set().union(*(m.getDeformers() for m in self.model.meshes))
        dfJoints = set().union(*(set(df.joints) for df in dfs))
        visJoints = {m.visJoint for m in self.model.meshes if m.visJoint}
        usedJoints = dfJoints | visJoints
        for joint in reversed(tuple(self.model.rootJoint.deepChildren(includeSelf=False))):
            if joint not in usedJoints and not joint.children:
                joint.parent = None # disconnect joint from model

    def hasSkinning(self, obj: bpy.types.Object):
        """True if an object uses this exporter"s armature for deformation. (Not just parenting)"""
        if obj.parent.original is not self.rigObj or obj.parent_type == 'BONE':
            return False
        if obj.parent_type == 'ARMATURE':
            return True
        for m in obj.modifiers:
            if m.type == 'ARMATURE' and m.object is not None and m.object.original is self.rigObj:
                return True
        return False

    @classmethod
    def _tristrip(cls, tris: list[tuple[int, int, int]], maxLen: int = None):
        """Convert triangles (tuples w/ 3 vertex indices) to strips (lists of vertex indices)."""
        # this is a basic implementation that pretty much makes random lines until it can't anymore
        # the result's not too shabby though!
        strips: list[list[int]] = []
        edgeAdjacentVerts: dict[tuple[int, int], list[int]] = {}
        # create map from edges to the all their adjacent vertices
        for tri in tris:
            edgeAdjacentVerts.setdefault((tri[0], tri[1]), []).append(tri[2])
            edgeAdjacentVerts.setdefault((tri[1], tri[2]), []).append(tri[0])
            edgeAdjacentVerts.setdefault((tri[2], tri[0]), []).append(tri[1])
        # create strips by picking arbitrary starting points and then going down arbitrary paths
        while edgeAdjacentVerts:
            firstEdge, firstEdgeAdjacentVerts = edgeAdjacentVerts.popitem() # pop to get edge fast
            edgeAdjacentVerts[firstEdge] = firstEdgeAdjacentVerts # add back so item isn't removed
            strip = list(firstEdge)
            strips.append(strip)
            cls._expandTristrip(strip, edgeAdjacentVerts, maxLen)
            # after initial strip expansion, expand it in the opposite direction as well
            # to do this, reverse the strip, then expand it
            # then, if reversing flipped the faces (which happens if the strip has an odd length),
            # we have to flip them back by adding an extra vert to the beginning
            # isFlipped = len(strip) % 2
            # strip.reverse()
            # cls._expandTristrip(strip, edgeAdjacentVerts, maxLen, isReversed=True)
            # if isFlipped:
            #     if len(strip) % 2:
            #         strip.reverse()
            #     else:
            #         strip.insert(0, strip[0]) # reverse doesn't flip faces; only way is extra vert
            # COMMENTED OUT FOR NOW BECAUSE THE INSERTION IN THE LINE ABOVE MAKES REVERSING NOT
            # WORTH IT (compression gains are balanced out by the extra vertices)
        return strips

    @classmethod
    def _expandTristrip(cls, strip: list[int], edgeAdjacentVerts: dict[tuple[int, int], list[int]],
                        maxLen: int = None, isReversed = False):
        """Expand a triangle strip forwards until it can't be expanded anymore."""
        # order alternates with every entry (clockwise vs counter)
        # isReversed determines which order to start with
        doReverse = isReversed
        latestEdge = tuple(strip[-2:])
        noMaxLen = maxLen is None
        while latestEdge in edgeAdjacentVerts and (noMaxLen or len(strip) < maxLen):
            adjacentVerts = edgeAdjacentVerts[latestEdge]
            newVert = adjacentVerts.pop() # pop one vert adjacent to this edge
            strip.append(newVert)
            tri = (*latestEdge, newVert) * 2
            for edgeIdx in range(1, 3):
                # in addition to deleting the data for this edge, delete the data for
                # equivalent edges/adjacent verts
                # for instance, the tri (1, 2, 3) will have an entry for (1, 2) to 3,
                # (2, 3) to 1, and (3, 1) to 2; if we're looking at the (1, 2) edge, the
                # other entries will still need to be popped as well since they represent
                # the same tri
                offsetEdge = tri[edgeIdx : edgeIdx + 2]
                offsetAdjacentVerts = edgeAdjacentVerts[offsetEdge]
                offsetAdjacentVerts.remove(tri[edgeIdx + 2])
                if not offsetAdjacentVerts:
                    del edgeAdjacentVerts[offsetEdge]
            if not adjacentVerts:
                del edgeAdjacentVerts[latestEdge]
            doReverse = not doReverse
            latestEdge = tuple(strip[-2:][::-1]) if doReverse else tuple(strip[-2:])


class BRRESAnimExporter(Generic[ANIM_SUBFILE_T]):

    ANIM_TYPE: type[ANIM_SUBFILE_T]

    def __init__(self, parentExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        self.parentExporter = parentExporter
        self.track = track

    def getAnim(self, name: str, action: bpy.types.Action) -> ANIM_SUBFILE_T:
        """Get an animation of this exporter's type from its parent BRRES exporter by name.

        If the requested animation doesn't exist yet, a new one is created.

        In either case, its length and looping flag are updated based on its current settings & 
        the provided action.
        """
        try:
            anim = self.parentExporter.anims[type(self)][name]
        except KeyError:
            anim = self.ANIM_TYPE(name)
            self.parentExporter.anims[type(self)][name] = anim
            self.parentExporter.res.folder(self.ANIM_TYPE).append(anim)
        anim.enableLoop = action.use_cyclic
        # add 1 as brres "length" is number of frames (including both endpoints),
        # as opposed to length of frame span (which doesn't include one endpoint, and is what
        # we have here)
        newLen = int(np.ceil(action.frame_range[1] + 1 - self.parentExporter.settings.frameStart))
        anim.length = max(anim.length, newLen)
        return anim


class BRRESChrExporter(BRRESAnimExporter[chr0.CHR0]):

    ANIM_TYPE = chr0.CHR0

    def __init__(self, parentExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentExporter, track)
        # get action & rig object
        # also, get some settings we'll have to alter for baking so we can restore them later
        settings = parentExporter.settings
        scene = parentExporter.context.scene
        viewLayer = parentExporter.context.view_layer
        strip = track.strips[0]
        action = strip.action
        frameStart, frameEnd = action.frame_range
        rigObj: bpy.types.Object = track.id_data
        frameRestore = scene.frame_current
        animDataRestore: dict[bpy.types.Object, tuple[bpy.types.Action, bool, bool]] = {}
        hideRestore = {obj: obj.hide_get() for obj in bpy.data.objects}
        # make sure this action involves bone transforms in some way; otherwise, don't export
        if not any(fc.data_path.startswith("pose") for fc in action.fcurves):
            return
        # create a collection where we'll put our stuff to animate
        # (all others are temporarily disabled to improve performance)
        animColl = bpy.data.collections.new("")
        scene.collection.children.link(animColl)
        excludeRestore: dict[bpy.types.LayerCollection, bool] = {}
        for child in viewLayer.layer_collection.children:
            excludeRestore[child] = child.exclude
            child.exclude = child.collection is not animColl
        # get objects to animate & back up their current settings
        # (to deal w/ constraints and drivers in the armature referencing other objects,
        # all objects w/ an nla track corresponding to this animation are evaluated)
        for obj in bpy.data.objects:
            animData = obj.animation_data
            try:
                objAction = animData.nla_tracks[track.name].strips[0].action
                _ = action.name # this lets us know if action is none via attribute error
            except (AttributeError, KeyError, IndexError):
                continue # no corresponding track/strip/action for this object
            animDataRestore[obj] = (animData.action, animData.use_nla, animData.use_tweak_mode)
            animData.use_tweak_mode = False
            animData.use_nla = False
            animData.action = objAction
            animColl.objects.link(obj)
        # bake animation data
        numFrames = int(np.ceil(frameEnd - frameStart)) + 1
        frameRange = np.linspace(frameStart, frameEnd, numFrames)
        emptyKfs = np.zeros((numFrames, 3))
        emptyKfs[:, 0] = frameRange - settings.frameStart
        bones = rigObj.pose.bones
        jointFrames = {bone: np.empty((numFrames, 3, 3)) for bone in bones}
        hasChild = {bone: bool(bone.children) for bone in bones}
        subframes = frameRange % 1
        roundedFrames = frameRange.astype(int)
        # side note: when blender alters bone matrices, the references don't change, only values
        # this means we can store matrix_basis now and access it directly later to get new values
        # rather than using bone.matrix_basis then (which is slower)
        localScales = {} # for segment scale compensate calculations
        prevRots: dict[bpy.types.PoseBone, Euler] = {} # for euler compatibility
        lastNewFrame: dict[bpy.types.PoseBone, int] = {}
        mtcs: dict[bpy.types.PoseBone, tuple[Matrix, Matrix]] = {b: (None, None) for b in bones}
        mtcsChanged = {bone: False for bone in bones}
        for kfIdx, (frame, subframe) in enumerate(zip(roundedFrames.tolist(), subframes.tolist())):
            # note: frame_set() is incredibly expensive because it updates everything!
            # this one line actually takes up the majority of export time for most models
            # unfortunately it's unavoidable, as it is the fastest way to properly update all
            # drivers and constraints, and it's what all the standard blender exporters use
            scene.frame_set(frame, subframe=subframe)
            # update bone matrices and which ones have changed
            for bone, (oldMtx, oldInv) in mtcs.items():
                mtx = bone.matrix
                mtcsChanged[bone] = mtxChanged = mtx != oldMtx
                if mtxChanged:
                    mtcs[bone] = (mtx.copy(), mtx.inverted() if hasChild[bone] else None)
            # save animation data for bones w/ changed matrices or changed parent matrices
            # (this ensures all possible modes of inheritance & ssc cases are covered)
            for bone, frameVals in jointFrames.items():
                parent = bone.parent
                if mtcsChanged[bone] or (parent and mtcsChanged[parent]):
                    try:
                        lastNewFrameIdx = lastNewFrame[bone]
                        frameVals[lastNewFrameIdx:kfIdx] = frameVals[lastNewFrameIdx]
                    except KeyError:
                        pass
                    frameVals[kfIdx] = parentExporter.getLocalSRT(bone, localScales, mtcs, prevRots)
                    lastNewFrame[bone] = kfIdx
        for bone, lastNewFrameIdx in lastNewFrame.items():
            frameVals = jointFrames[bone]
            frameVals[lastNewFrameIdx:] = frameVals[lastNewFrameIdx]
        # create chr0
        chrAnim = self.getAnim(track.name, action)
        usedJointNames = {jointAnim.jointName for jointAnim in chrAnim.jointAnims}
        jointAnims: dict[bpy.types.PoseBone, chr0.JointAnim] = {}
        animFmts = [animation.I12, animation.D4, animation.I12] # formats for s, r, t respectively
        for bone, frameVals in jointFrames.items():
            if bone.name in usedJointNames:
                continue # ignore bones already in chr0, in case it already existed from another rig
            jointAnims[bone] = jointAnim = chr0.JointAnim(bone.name)
            jointAnim.animFmts[:] = animFmts
            if bone.bone.inherit_scale == 'NONE':
                jointAnim.segScaleComp = True
                try:
                    jointAnims[bone.parent].segScaleCompParent = True
                except KeyError:
                    pass
            chrAnim.jointAnims.append(jointAnim)
            allAnims = (jointAnim.scale, jointAnim.rot, jointAnim.trans)
            frameVals[np.isclose(frameVals, 0, atol=0.001)] = 0
            for anims, frames in zip(allAnims, frameVals.transpose((1, 2, 0))):
                for anim, compVals in zip(anims, frames):
                    # filter out frames w/ the same values as prev & next frames
                    anim.length = chrAnim.length
                    eqNext = np.isclose(compVals[:-1], compVals[1:])
                    dupFrames = np.logical_and(eqNext[:-1], eqNext[1:])
                    # note: last frame is filtered out if it equals prev, but first is always needed
                    frameFltr = np.logical_not(np.concatenate(([False], dupFrames, eqNext[-1:])))
                    keyframes = emptyKfs[frameFltr].copy()
                    keyframes[:, 1] = compVals[frameFltr]
                    anim.keyframes = keyframes
        # restore stuff we had to change for baking
        for obj, restore in animDataRestore.items():
            animData = obj.animation_data
            animData.action, animData.use_nla, animData.use_tweak_mode = restore
        scene.frame_set(frameRestore)
        for coll, exclude in excludeRestore.items():
            coll.exclude = exclude
        bpy.data.collections.remove(animColl)
        for obj, hide in hideRestore.items():
            # this restoration is needed because the collection exclude thing can change obj hiding
            obj.hide_set(hide)


class BRRESClrExporter(BRRESAnimExporter[clr0.CLR0]):

    ANIM_TYPE = clr0.CLR0

    @classmethod
    @cache
    def regPaths(cls):
        """Tuple of all possible CLR0 register paths."""
        return (
            "brres.lightChans.coll_[0].difColor",
            "brres.lightChans.coll_[1].difColor",
            "brres.lightChans.coll_[0].ambColor",
            "brres.lightChans.coll_[1].ambColor",
            "brres.colorRegs.standard2",
            "brres.colorRegs.standard3",
            "brres.colorRegs.standard4",
            "brres.colorRegs.constant1",
            "brres.colorRegs.constant2",
            "brres.colorRegs.constant3",
            "brres.colorRegs.constant4"
        )

    def __init__(self, parentExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentExporter, track)
        frameStart = parentExporter.settings.frameStart
        strip = track.strips[0]
        action = strip.action
        matAnim = clr0.MatAnim(strip.id_data.name)
        regPaths = self.regPaths()
        # look through fcurves, updating mat animation if any are relevant to clr0
        for fcurve in action.fcurves:
            # parse data path to get register id in allRegs
            # if path is invalid for clr0, skip this fcurve
            try:
                regIdx = regPaths.index(fcurve.data_path)
            except ValueError:
                continue
            # get animation data by evaluating curve
            maxFrame = int(np.ceil(fcurve.range()[1]))
            frameRange = range(frameStart, frameStart + maxFrame + 1)
            frameVals = [fcurve.evaluate(frameIdx) for frameIdx in frameRange]
            # grab existing reg animation or create new one if necessary, then add new data
            # note that clr0 animations aren't keyframed, so we must start at the initial frame
            # this means if the animation already exists, and the animation for this component is
            # longer than what already exists, we just extend it using np.pad()
            regAnim = matAnim.allRegs[regIdx]
            if regAnim is None:
                regAnim = clr0.RegAnim(np.zeros((0, 4)), ~np.zeros(4, dtype=np.uint8))
                matAnim.setRegAnim(regIdx, regAnim)
            regAnim.mask[fcurve.array_index] = 0
            colors = regAnim.normalized
            curLen = len(colors)
            newLen = len(frameVals)
            if newLen > curLen:
                extendMode = "edge" if curLen > 0 else "constant"
                colors = np.pad(colors, ((0, newLen - curLen), (0, 0)), extendMode)
            colors[:newLen, fcurve.array_index] = np.reshape(frameVals, (1, -1))
            regAnim.normalized = colors
        # if mat anim is non-empty (relevant fcurves were found), update clr anim
        if any(reg is not None for reg in matAnim.allRegs):
            clrAnim = self.getAnim(track.name, action)
            clrAnim.matAnims.append(matAnim)


class BRRESPatExporter(BRRESAnimExporter[pat0.PAT0]):

    ANIM_TYPE = pat0.PAT0

    @classmethod
    @cache
    def pathInfo(cls):
        """Mapping from every Blender path for PAT0 animations to the path's texture index."""
        return {f"brres.textures.coll_[{i}].activeImgSlot": i for i in range(gx.MAX_TEXTURES)}

    def __init__(self, parentExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentExporter, track)
        frameStart = parentExporter.settings.frameStart
        strip = track.strips[0]
        action = strip.action
        mat: bpy.types.Material = strip.id_data
        matAnim = pat0.MatAnim(mat.name)
        pathInfo = self.pathInfo()
        # look through fcurves, updating mat animation if any are relevant to pat0
        for fcurve in action.fcurves:
            # parse data path to get texture index & specific property
            # if path is invalid for pat0, skip this fcurve
            try:
                texIdx = pathInfo[fcurve.data_path]
            except KeyError:
                continue
            texAnim = pat0.TexAnim()
            matAnim.texAnims[texIdx] = texAnim
            texImgs = mat.brres.textures[texIdx].imgs
            for texImg in texImgs:
                if texImg.img not in parentExporter.images and parentExporter.onlyUsedImg:
                    self.parentExporter.exportImg(texImg.img)
                texAnim.texNames.append(parentExporter.imgName(texImg.img))
            # fill out animation data by evaluating curve
            frameIdcs = []
            frameVals = []
            minFrame, maxFrame = fcurve.range()
            texAnim.length = int(np.ceil(maxFrame - minFrame))
            prevVal: int = None
            for frameIdx in np.linspace(minFrame, maxFrame, int(np.ceil(texAnim.length)) + 1):
                frameVal = fcurve.evaluate(frameIdx)
                if frameVal != prevVal:
                    frameIdcs.append(frameIdx - frameStart)
                    frameVals.append(frameVal - 1)
                    prevVal = frameVal
            texAnim.keyframes = np.array([frameIdcs, frameVals, [0] * len(frameIdcs)]).T
        # if mat anim is non-empty (relevant fcurves were found), update pat anim
        if matAnim.texAnims:
            patAnim = self.getAnim(track.name, action)
            patAnim.matAnims.append(matAnim)


class BRRESSrtExporter(BRRESAnimExporter[srt0.SRT0]):

    ANIM_TYPE = srt0.SRT0

    @classmethod
    @cache
    def pathInfo(cls):
        """Mapping from every possible Blender path for SRT0 animations to info about the path.

        This info is a tuple containing the name of the MatAnim collection in which the animation
        corresponding to the path belongs (i.e., regular or indirect texture), its index within that
        collection, and the final property accessed by the animation (i.e., scale/rot/trans).
        """
        texAnimPaths = { # maps collection to blender collection path & max length
            "texAnims": ("brres.textures", gx.MAX_TEXTURES),
            "indAnims": ("brres.indSettings.transforms", gx.MAX_INDIRECT_MTCS)
        }
        propPaths = { # maps prop to blender prop path
            "scale": "transform.scale",
            "rot": "transform.rotation",
            "trans": "transform.translation"
        }
        pathInfo: dict[str, tuple[str, int, str]] = {}
        for texAnimColl, (texAnimPath, texAnimCollLen) in texAnimPaths.items():
            for i in range(texAnimCollLen):
                for prop, propPath in propPaths.items():
                    fullPath = f"{texAnimPath}.coll_[{i}].{propPath}"
                    pathInfo[fullPath] = (texAnimColl, i, prop)
        return pathInfo

    def __init__(self, parentExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentExporter, track)
        frameStart = parentExporter.settings.frameStart
        strip = track.strips[0]
        action = strip.action
        matAnim = srt0.MatAnim(strip.id_data.name)
        pathInfo = self.pathInfo()
        # look through fcurves, updating mat animation if any are relevant to srt0
        for fcurve in action.fcurves:
            # parse data path to get texture index & specific property
            # if path is invalid for srt0, skip this fcurve
            try:
                texAnimColl, texIdx, texProp = pathInfo[fcurve.data_path]
            except KeyError:
                continue
            # grab existing texture animation or create new one if this is the first fcurve read
            # that uses this texture
            texAnimColl: dict[int, srt0.TexAnim] = getattr(matAnim, texAnimColl)
            try:
                texAnim = texAnimColl[texIdx]
            except KeyError:
                texAnim = srt0.TexAnim()
                texAnimColl[texIdx] = texAnim
            compAnim: animation.Animation = getattr(texAnim, texProp)[fcurve.array_index]
            # fill out animation data by evaluating curve
            # this is a bit crude - in the future maybe we can make things more robust to compress
            # the output data more, but this works fine for now
            frameIdcs = []
            frameVals = []
            minFrame, maxFrame = fcurve.range()
            compAnim.length = int(np.ceil(maxFrame - minFrame))
            for frameIdx in np.linspace(minFrame, maxFrame, compAnim.length + 1):
                frameIdcs.append(frameIdx - frameStart)
                frameVals.append(fcurve.evaluate(frameIdx))
            if texProp == "rot":
                frameVals = np.rad2deg(frameVals)
            compAnim.keyframes = np.array([frameIdcs, frameVals, [0] * len(frameIdcs)]).T
            if len(set(frameVals)) == 1:
                compAnim.keyframes = compAnim.keyframes[:1] # all keyframes are same, so reduce to 1
        # if mat anim is non-empty (relevant fcurves were found), update srt anim
        if matAnim.texAnims or matAnim.indAnims:
            srtAnim = self.getAnim(track.name, action)
            srtAnim.mtxGen = tf.MayaMtxGen2D
            srtAnim.matAnims.append(matAnim)


class BRRESVisExporter(BRRESAnimExporter[vis0.VIS0]):

    ANIM_TYPE = vis0.VIS0

    def __init__(self, parentExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentExporter, track)
        frameStart = parentExporter.settings.frameStart
        strip = track.strips[0]
        action = strip.action
        rig: bpy.types.Object | bpy.types.Armature = track.id_data
        boneVisSuffix = ".hide"
        boneVisSuffixLen = len(boneVisSuffix)
        jointAnims: list[vis0.JointAnim] = []
        for fcurve in action.fcurves:
            dataPath = fcurve.data_path
            if not dataPath.endswith(boneVisSuffix):
                continue
            try:
                # cut off ".hide" to get bone
                bone: bpy.types.Bone = rig.path_resolve(dataPath[:-boneVisSuffixLen])
                if not isinstance(bone, bpy.types.Bone):
                    continue
            except ValueError:
                continue # not a bone visibility path
            jointAnim = vis0.JointAnim(bone.name)
            jointAnims.append(jointAnim)
            maxFrame = int(np.ceil(fcurve.range()[1]))
            frameRange = range(frameStart, frameStart + maxFrame + 1)
            jointAnim.frames = np.logical_not([fcurve.evaluate(i) for i in frameRange], dtype=bool)
        # if any joint anims were created, add to vis0 (creating if it doesn't already exist)
        if jointAnims:
            visAnim = self.getAnim(track.name, action)
            # only add joint anims w/ names not already in vis0
            usedJointNames: set[str] = set()
            combinedAnims = visAnim.jointAnims + jointAnims
            visAnim.jointAnims = []
            for jointAnim in combinedAnims:
                if jointAnim.jointName not in usedJointNames:
                    usedJointNames.add(jointAnim.jointName)
                    visAnim.jointAnims.append(jointAnim)


class BRRESExporter():

    def __init__(self, context: bpy.types.Context, file, settings: "ExportBRRES",
                 baseData: bytes = b""):
        self.res = brres.BRRES()
        self.context = context
        self.depsgraph = context.evaluated_depsgraph_get()
        self.settings = settings
        self.models: dict[bpy.types.Object, BRRESMdlExporter] = {}
        self.images: dict[bpy.types.Image, tex0.TEX0] = {}
        self.onlyUsedImg = settings.doImg and not settings.includeUnusedImg
        # set up bone axis conversion matrices
        self.mtxBoneToBRRES: Matrix = axis_conversion(
            from_forward='X',
            from_up='Y',
            to_forward=settings.secondaryBoneAxis,
            to_up=settings.primaryBoneAxis
        )
        self.mtxBoneFromBRRES: Matrix = self.mtxBoneToBRRES.inverted()
        self.mtxBoneToBRRES4x4 = self.mtxBoneToBRRES.to_4x4()
        self.mtxBoneFromBRRES4x4 = self.mtxBoneFromBRRES.to_4x4()
        # generate models & animations
        # note: originally i used the depsgraph instead of bpy.data, but that sometimes comes with
        # warnings & crashes and can generally be hard to predict, so i just do things this way and
        # use the depsgraph only when needed (e.g., evaluating meshes)
        self.anims: dict[type[BRRESAnimExporter], dict[str, animation.AnimSubfile]]
        self.anims = {t: {} for t in (
            BRRESChrExporter, BRRESClrExporter, BRRESPatExporter, BRRESSrtExporter, BRRESVisExporter
        )}
        if settings.includeUnusedImg:
            # export all images
            # (if this setting's disabled, image export is handled by model/anim exporters)
            for img in bpy.data.images:
                self.exportImg(img)
        for obj in bpy.data.objects:
            # export armatures included in limit, as well as armature animations (chr/vis)
            if obj.type == 'ARMATURE' and limitIncludes(settings.limitTo, obj):
                self._exportModel(obj)
                if settings.doAnim and settings.includeArmAnims:
                    if obj.animation_data:
                        usedNames = set()
                        for track in obj.animation_data.nla_tracks:
                            if self.testAnimTrack(track, usedNames):
                                self._exportAnim(BRRESChrExporter, track)
                                self._exportAnim(BRRESVisExporter, track)
                    # vis animations can either be on armature objects or the armatures themselves
                    if obj.data.animation_data:
                        usedNames = set()
                        for track in obj.data.animation_data.nla_tracks:
                            if self.testAnimTrack(track, usedNames):
                                self._exportAnim(BRRESVisExporter, track)
        if settings.doAnim and settings.includeMatAnims:
            # finally, export material animations
            mats = {bpy.data.materials[mat] for mdl in self.models.values() for mat in mdl.mats}
            for mat in mats:
                if mat.animation_data:
                    usedNames = set()
                    for track in mat.animation_data.nla_tracks:
                        if self.testAnimTrack(track, usedNames):
                            self._exportAnim(BRRESClrExporter, track)
                            self._exportAnim(BRRESPatExporter, track)
                            self._exportAnim(BRRESSrtExporter, track)
        if not settings.doArmGeo:
            # if armature/geometry export is disabled, for now, just delete models before packing
            # (reason being that models are still needed for animation export; this could be
            # optimized, since geometry data doesn't have to be processed in this case, but this is
            # rarely even a useful setting so for now, idrc)
            self.res.files.pop(mdl0.MDL0, None)
        # optionally merge with existing file
        if settings.doMerge and baseData:
            baseRes = brres.BRRES.unpack(baseData)
            for fType, folder in self.res.files.items():
                # get files from base folder & new folder and overwrite those w/ same names
                baseFolder = baseRes.folder(fType)
                baseFiles = {f.name: f for f in baseFolder}
                newFiles = {f.name: f for f in folder}
                baseFolder[:] = (baseFiles | newFiles).values() # in conflicts, new files win
            self.res = baseRes
        # write file
        self.res.sort()
        packed = self.res.pack()
        packed += binaryutils.pad(f"BerryBush {verStr(addonVer())}".encode("ascii"), 16)
        if settings.padEnable:
            packed = binaryutils.pad(packed, settings.padSize * int(settings.padUnit))
        # copy = brres.BRRES.unpack(packed)
        file.write(packed)

    def _exportModel(self, rigObj: bpy.types.Object):
        self.models[rigObj] = BRRESMdlExporter(self, rigObj)

    def exportImg(self, bImg: bpy.types.Image):
        img = tex0.TEX0(bImg.name)
        self.images[bImg] = img
        self.res.folder(tex0.TEX0).append(img)
        dims = np.array(bImg.size, dtype=np.integer)
        px = np.array(bImg.pixels).reshape(dims[1], dims[0], bImg.channels)
        # pad all image dimensions to at least 1 (render result is 0x0 if unset) & channels to 4
        px = np.pad(px, ((0, dims[1] == 0), (0, dims[0] == 0), (0, 4 - bImg.channels)))
        dims[dims == 0] = 1
        img.images.append(px[::-1])
        img.fmt = IMG_FMTS[bImg.brres.fmt]
        for mm in bImg.brres.mipmaps:
            dims //= 2
            mmPx = padImgData(mm.img, (dims[1], dims[0], 4))[::-1]
            img.images.append(mmPx)

    def imgName(self, img: bpy.types.Image):
        """Get the BRRES name corresponding to a Blender image."""
        try:
            return self.images[img].name
        except KeyError:
            try:
                return img.name
            except AttributeError:
                return None

    def _exportAnim(self, t: type[BRRESAnimExporter], track: bpy.types.NlaTrack):
        t(self, track)

    def testAnimTrack(self, track: bpy.types.NlaTrack, usedNames: set[str]):
        """Determine if an NLA track should be used for BRRES export."""
        if self.settings.includeMutedAnims or not track.mute:
            if track.strips and track.strips[0].action:
                if track.name not in usedNames:
                    usedNames.add(track.name)
                    return True
        return False

    def getLocalSRT(self, bone: bpy.types.PoseBone, localScales: dict[bpy.types.Bone, Vector],
                    mtcs: dict[bpy.types.PoseBone, tuple[Matrix, Matrix]],
                    prevRots: dict[bpy.types.PoseBone, Euler]):
        """Get a bone's local SRT in BRRES space based on its current pose.

        A dict mapping each bone processed so far to its bone-space local scale must be provided for
        the sake of segment scale compensate. (An entry will be added to this dict for the current
        bone, and its parent's entry may be accessed)

        Additionally, a dict mapping each bone to its matrix and inverse matrix should be provided
        for optimization's sake.

        Finally, a dict mapping each bone to its previous rotation should be provided to ensure that
        Euler compatibility is maintained. If not applicable, an empty dict will suffice.
        """
        # standard calculation: we define the local values via
        # bone.matrix = parent @ local t @ local r @ local s
        # so, to get them we do inv parent @ bone.matrix (w/ space conversion)
        parent = bone.parent
        mtx: Matrix = mtcs[parent][1].copy() if parent else MTX_TO_BONE.to_4x4()
        mtx @= mtcs[bone][0]
        # additional calculation for segment scale compensate: we define the locals via
        # bone.matrix = parent @ local t @ inv parent local s @ local r @ local s
        # so the calculation is slightly more complex:
        # local t @ parent local s @ inv local t @ inv parent @ bone.matrix
        # if you write everything out, you can see that stuff cancels out and
        # you end up w/ local t @ local r @ local s
        if bone.bone.inherit_scale == 'NONE' and parent:
            parentScale = Matrix.LocRotScale(None, None, localScales[parent])
            trans = Matrix.Translation(mtx.to_translation())
            mtx = trans @ parentScale @ trans.inverted() @ mtx
        localScales[bone] = mtx.to_scale()
        # and then we can't forget about space conversions!
        if parent:
            mtx = self.mtxBoneToBRRES4x4 @ mtx
        mtx @= self.mtxBoneFromBRRES4x4
        # finally, decompose
        s = mtx.to_scale()
        try:
            r = mtx.to_euler("XYZ", prevRots[bone])
        except KeyError:
            r = mtx.to_euler("XYZ")
        prevRots[bone] = r
        t = mtx.to_translation() * self.settings.scale
        # getting euler values to convert to radians is slow af for some reason,
        # so only do that if rotation exists (all angles aren't 0)
        if r != IDENTITY_EULER:
            r = np.rad2deg(r)
        return (s, r, t)


def drawOp(self, context: bpy.types.Context):
    self.layout.operator(ExportBRRES.bl_idname, text="Binary Revolution Resource (.brres)")


class ExportBRRES(bpy.types.Operator, ExportHelper):
    """Write a BRRES file"""

    bl_idname = "export_scene.brres"
    bl_label = "Export BRRES"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".brres"
    filter_glob: bpy.props.StringProperty(
        default="*.brres",
        options={'HIDDEN'},
    )

    includeSuppressed: bpy.props.BoolProperty(
        name="Bypass Warning Suppression",
        description="Report all detected problems, including those flagged to be ignored",
        default=False
    )

    padEnable: bpy.props.BoolProperty(
        name="Enable Padding",
        description="Pad file with null bytes if smaller than the goal size",
        default=False
    )

    padSize: bpy.props.IntProperty(
        name="Pad To",
        description="Goal size to which the exported file should be padded with null bytes (useful for testing mods via Riivolution in Dolphin, where you can alter files without restarting your game as long as their sizes stay constant)", # pylint: disable=line-too-long
        min=0,
        default=0
    )

    padUnit: bpy.props.EnumProperty(
        name="Padding Unit",
        description="Unit to use for padded file size",
        items=(
            (str(2 ** 0), "B", ""),
            (str(2 ** 10), "KB", ""),
            (str(2 ** 20), "MB", ""),
        ),
        default=str(2 ** 10)
    )

    limitTo: bpy.props.EnumProperty(
        name="Limit To",
        description="Objects to export",
        items=(
            ('ALL', "All", ""),
            ('SELECTED', "Selected", ""),
            ('VISIBLE', "Visible", ""),
        ),
        default='ALL'
    )

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Export is scaled up by this factor. Recommended values are 16 for most models and 30 for worldmaps", # pylint: disable=line-too-long
        default=16
    )

    doBackup: bpy.props.BoolProperty(
        name="Create Backup",
        description="If a file with this name already exists, create a backup",
        default=True
    )

    doMerge: bpy.props.BoolProperty(
        name="Merge Files",
        description="If a file with this name already exists, also include its sub-files not overwritten by the current model in the export", # pylint: disable=line-too-long
        default=False
    )

    doArmGeo: bpy.props.BoolProperty(
        name="Export Armatures & Geometry",
        default=True
    )

    applyModifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Apply modifiers to mesh objects",
        default=True,
    )

    useCurrentPose: bpy.props.BoolProperty(
        name="Use Current Pose",
        description="Apply current armature pose as rest pose",
        default=False
    )

    forceRootBones: bpy.props.BoolProperty(
        name="Force Root Bones",
        description="Use current armature roots for unparented assets rather than creating new ones", # pylint: disable=line-too-long
        default=False
    )

    removeUnusedBones: bpy.props.BoolProperty(
        name="Remove Unused Bones",
        description="Only export bones used for deforming and visibility (and their ancestors). Other bones are still taken into account for constraint & driver evaluation, but not exported.",  # pylint: disable=line-too-long
        default=False
    )

    doQuantize: bpy.props.BoolProperty(
        name="Simplify Weights",
        description="Quantize vertex weights to a fixed interval before normalization (good for file size, performance, & preventing crashes)", # pylint: disable=line-too-long
        default=False,
    )

    quantizeSteps: bpy.props.FloatProperty(
        name="Simplify Weights",
        description="Number of steps between 0 and 1 (lower -> more simplified but potentially less accurate)", # pylint: disable=line-too-long
        min=1,
        max=100,
        default=4
    )

    primaryBoneAxis: bpy.props.EnumProperty(
        name="Primary Bone Axis",
        description="Source primary bone axis (Blender uses Y)",
        items=(
            ('X', "X", ""),
            ('Y', "Y", ""),
            ('Z', "Z", ""),
            ('-X', "-X", ""),
            ('-Y', "-Y", ""),
            ('-Z', "-Z", ""),
        ),
        default='X',
    )

    secondaryBoneAxis: bpy.props.EnumProperty(
        name="Secondary Bone Axis",
        description="Source secondary bone axis (Blender uses X)",
        items=(
            ('X', "X", ""),
            ('Y', "Y", ""),
            ('Z', "Z", ""),
            ('-X', "-X", ""),
            ('-Y', "-Y", ""),
            ('-Z', "-Z", ""),
        ),
        default='Y',
    )

    doImg: bpy.props.BoolProperty(
        name="Export Images",
        default=True
    )

    includeUnusedImg: bpy.props.BoolProperty(
        name="Include Unused",
        description="Export all images from the current Blend file, including those not otherwise used by the exported model", # pylint: disable=line-too-long
        default=False
    )

    doAnim: bpy.props.BoolProperty(
        name="Export Animations",
        default=True
    )

    includeArmAnims: bpy.props.BoolProperty(
        name="Armatures",
        description="Include armature animations (CHR, VIS)",
        default=True
    )

    includeMatAnims: bpy.props.BoolProperty(
        name="Materials",
        description="Include material animations (CLR, PAT, SRT)",
        default=True
    )

    includeMutedAnims: bpy.props.BoolProperty(
        name="Include Muted Tracks",
        description="Export actions from muted NLA tracks",
        default=True
    )

    frameStart: bpy.props.IntProperty(
        name="Frame Start",
        description="First frame of animation",
        default=1
    )

    def execute(self, context):
        profiler = Profile()
        profiler.enable()
        restoreShading = solidView(context) # temporarily set viewports to solid view for speed
        context.window.cursor_set('WAIT')
        self.report({'INFO'}, "Exporting BRRES...")
        baseData = b""
        if self.doMerge or self.doBackup: # get existing data for merge/backup
            try:
                with open(self.filepath, "rb") as f:
                    baseData = f.read()
                    if self.doBackup: # back up data into another file if enabled
                        path = Path(self.filepath)
                        backupLabel = datetime.now().strftime("_backup_%Y-%m-%d_%H-%M-%S")
                        backupPath = Path(path.parent, path.stem + backupLabel + path.suffix)
                        with open(str(backupPath), "wb") as f:
                            f.write(baseData)
            except FileNotFoundError:
                pass
        with open(self.filepath, "wb") as f: # export main file
            BRRESExporter(context, f, self, baseData)
        name = f"\"{os.path.basename(self.filepath)}\""
        warns, suppressed = verifyBRRES(self, context)
        if warns:
            plural = "s" if warns > 1 else ""
            sup = f" and {suppressed} suppressed" if suppressed else ""
            e = f"Exported {name} with {warns} warning{plural}{sup}. Check the Info Log for details"
            self.report({'WARNING'}, e)
        elif suppressed:
            plural = "s" if suppressed > 1 else ""
            self.report({'INFO'}, f"Exported {name} with {suppressed} suppressed warning{plural}")
        else:
            self.report({'INFO'}, f"Exported {name} without any warnings")
        context.window.cursor_set('DEFAULT')
        restoreView(restoreShading)
        profiler.disable()
        with open(LOG_PATH, "w", encoding="utf-8") as logFile:
            Stats(profiler, stream=logFile).sort_stats(SortKey.CUMULATIVE).print_stats()
        return {'FINISHED'}

    def draw(self, context: bpy.types.Context):
        pass # leave menu drawing up to the export panels


class ExportPanel(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return context.space_data.active_operator.bl_idname == "EXPORT_SCENE_OT_brres"


class GeneralPanel(ExportPanel):
    bl_idname = "BRRES_PT_export_general"
    bl_label = "General"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ExportBRRES = context.space_data.active_operator
        # padding options
        row = layout.row().split(factor=.4)
        row.use_property_split = False
        padLabelCol = row.column()
        padLabelCol.alignment = 'RIGHT'
        padLabelCol.label(text=getPropName(settings, "padSize"))
        padRow = row.column().row(align=True)
        padRow.prop(settings, "padEnable", text="")
        padPropRow = padRow.row()
        padPropRow.use_property_decorate = padPropRow.use_property_split = False
        padPropRow.enabled = settings.padEnable
        padPropRow = padPropRow.split(factor=.67, align=True)
        padPropRow.prop(settings, "padSize", text="")
        padPropRow.prop(settings, "padUnit", text="")
        # everything else
        layout.prop(settings, "limitTo")
        layout.prop(settings, "scale")
        layout.prop(settings, "doBackup")
        layout.prop(settings, "doMerge")


class ArmGeoPanel(ExportPanel):
    bl_idname = "BRRES_PT_export_arm_geo"
    bl_label = "Armatures & Geometry"

    def draw_header(self, context: bpy.types.Context):
        settings: ExportBRRES = context.space_data.active_operator
        self.layout.prop(settings, "doArmGeo", text="")

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ExportBRRES = context.space_data.active_operator
        layout.enabled = settings.doArmGeo
        layout.prop(settings, "applyModifiers")
        layout.prop(settings, "useCurrentPose")
        layout.prop(settings, "forceRootBones")
        layout.prop(settings, "removeUnusedBones")
        drawCheckedProp(layout, settings, "doQuantize", settings, "quantizeSteps")
        layout.prop(settings, "primaryBoneAxis", expand=True)
        layout.prop(settings, "secondaryBoneAxis", expand=True)


class ImagePanel(ExportPanel):
    bl_idname = "BRRES_PT_export_image"
    bl_label = "Images"

    def draw_header(self, context: bpy.types.Context):
        settings: ExportBRRES = context.space_data.active_operator
        self.layout.prop(settings, "doImg", text="")

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ExportBRRES = context.space_data.active_operator
        layout.enabled = settings.doImg
        layout.prop(settings, "includeUnusedImg")


class AnimPanel(ExportPanel):
    bl_idname = "BRRES_PT_export_anim"
    bl_label = "Animations"

    def draw_header(self, context: bpy.types.Context):
        settings: ExportBRRES = context.space_data.active_operator
        self.layout.prop(settings, "doAnim", text="")

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ExportBRRES = context.space_data.active_operator
        layout.enabled = settings.doAnim
        layout.prop(settings, "includeArmAnims")
        layout.prop(settings, "includeMatAnims")
        layout.prop(settings, "includeMutedAnims")
        layout.prop(settings, "frameStart")
