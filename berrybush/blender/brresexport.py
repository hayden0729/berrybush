# standard imports
from cProfile import Profile
from functools import cache
from pstats import SortKey, Stats
import os
from typing import Generic, TypeVar
# 3rd party imports
import bpy
from bpy_extras.io_utils import ExportHelper, axis_conversion
from mathutils import Euler, Matrix, Vector
import numpy as np
# internal imports
from .backup import tryBackup
from .common import (
    MTX_TO_BONE, MTX_FROM_BONE, MTX_TO_BRRES, LOG_PATH,
    solidView, restoreView, usedMatSlots, makeUniqueName, foreachGet, getLayerData,
    getLoopVertIdcs, getLoopFaceIdcs, getFaceMatIdcs, getPropName, drawCheckedProp,
    simplifyLayerData, layerDataLoopDomain
)
from .limiter import ObjectLimiter, ObjectLimiterFactory
from .material import AlphaSettings, DepthSettings, LightChannelSettings, MiscMatSettings
from .mesh import MeshSettings
from .render import BlendImageExtractor
from .tev import TevSettings, TevStageSettings
from .texture import TexSettings
from .updater import addonVer, verStr
from .verify import verifyBRRES
from ..wii import (
    animation, binaryutils, brres, chr0, clr0, gx, mdl0, pat0, srt0, tex0, transform as tf, vis0
)


ANIM_SUBFILE_T = TypeVar("ANIM_SUBFILE_T", bound=animation.AnimSubfile)


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


class GeometryInfo():
    """Calculates and stores a bunch of geometry-related arrays for Blender mesh object export."""

    def __init__(self):
        self.loopIdcs: np.ndarray = None
        self.loopVertIdcs: np.ndarray = None
        self.loopFaceIdcs: np.ndarray = None
        self.triLoopIdcs: np.ndarray = None
        self.triMatIdcs: np.ndarray = None
        self.vertDfs: list[mdl0.Deformer] = []
        self.dfs: list[mdl0.Deformer] = []
        self.dfVertIdcs: list[int] = None
        self.vertDfIdcs: np.ndarray = None
        self.loopDfIdcs: np.ndarray = None
        self.loopPsns: np.ndarray = None
        self.loopNrms: np.ndarray = None
        self.loopClrs: dict[int, tuple[str, np.ndarray]] = {}
        self.loopUVs: dict[int, tuple[str, np.ndarray]] = {}

    def update(self, mesh: bpy.types.Mesh, obj: bpy.types.Object, scale = 1.0):
        """Update geometry data from a Blender mesh object."""
        self.triLoopIdcs = foreachGet(mesh.loop_triangles, "loops", 3, int)[:, ::-1].flatten()
        self.loopVertIdcs = getLoopVertIdcs(mesh)
        self.loopFaceIdcs = getLoopFaceIdcs(mesh)
        self.triMatIdcs = getFaceMatIdcs(mesh)[self.loopFaceIdcs][self.triLoopIdcs]
        try:
            # brres mesh attributes may be deleted by modifiers, so try to get from original mesh
            # and fall back on evaluated mesh's attributes in case this fails
            meshAttrs = obj.original.data.brres.meshAttrs
        except AttributeError:
            meshAttrs = mesh.brres.meshAttrs
        vertPsns = foreachGet(mesh.vertices, "co", 3) * scale
        self.loopPsns = vertPsns[self.loopVertIdcs]
        self.loopNrms = foreachGet(mesh.loops, "normal", 3)
        self.loopClrs = self._processLayerData(getLayerData(mesh, meshAttrs.clrs, False))
        self.loopUVs = self._processLayerData(getLayerData(mesh, meshAttrs.uvs, True))

    def updateSkinning(self, mesh: bpy.types.Mesh, obj: bpy.types.Object,
                       mdlExporter: "BRRESMdlExporter"):
        """Update skinning-related geometry data from a Blender mesh object.

        Requires a BRRES MDL exporter for deformer information."""
        # first, get several useful arrays for dealing with the mesh's deformers
        # vertDfs: deformer for each vertex
        # dfs: all unique deformers in vertDfs
        # dfVertIdcs: for each deformer in dfs, index of first vertex in vertDfs
        # vertDfIdcs: for each vert in vertDfs, index of its deformer in dfs
        self.vertDfs = mdlExporter.getVertDfs(mesh, obj)
        dfVertIdxMap = {df: i for i, df in enumerate(self.vertDfs)}
        dfIdxMap = {df: i for i, df in enumerate(dfVertIdxMap)}
        self.dfs = list(dfVertIdxMap.keys())
        self.dfVertIdcs = [dfVertIdxMap[df] for df in self.dfs]
        self.vertDfIdcs = np.array([dfIdxMap[df] for df in self.vertDfs], dtype=int)
        self.loopDfIdcs = self.vertDfIdcs[self.loopVertIdcs]
        # then, adjust positions & normals
        # basically, vertices w/ single-joint deformers are stored relative to those joints,
        # so we have to convert them from rest pose to that relative space
        # note: this does not apply to multi-weight deformers,
        # presumably because their matrices aren't guaranteed to be invertible?
        # (for instance, imagine a deformer for 2 equally weighted joints w/ opposite rotations)
        mdl = mdlExporter.model
        dfMtcs = np.array([df.mtx(mdl) if len(df) == 1 else np.identity(4) for df in self.dfs])
        invDfMtcs = np.array([np.linalg.inv(m) for m in dfMtcs])
        loopDfMtcs = dfMtcs[self.loopDfIdcs]
        invLoopDfMtcs = invDfMtcs[self.loopDfIdcs]
        padPsns = np.pad(self.loopPsns, ((0, 0), (0, 1)), constant_values=1) # pad for 4x4 matmul
        # https://stackoverflow.com/questions/35894631/multiply-array-of-vectors-with-array-of-matrices-return-array-of-vectors
        self.loopPsns = np.einsum("ij, ijk->ik", padPsns, invLoopDfMtcs.swapaxes(1, 2))[:, :3]
        self.loopNrms = np.einsum("ij, ijk->ik", self.loopNrms, loopDfMtcs[:, :3, :3])

    def simplifyData(self):
        """Simplify vertex attribute data for the sake of file size optimization."""
        self.loopNrms[:] = simplifyLayerData(self.loopNrms)
        for attrLayers in (self.loopClrs, self.loopUVs):
            for layerName, layerData in attrLayers.values():
                layerData[:] = simplifyLayerData(layerData)

    @classmethod
    def _processLayerData(cls, layerData: list) -> dict[int, tuple[str, np.ndarray]]:
        """Process layer data obtained via getLayerData() to prepare it for export.

        The returned dict maps slot indices to layer names & loop-domain data.
        """
        processed = {}
        for i, (layer, data, idcs) in enumerate(layerData):
            if layer:
                processed[i] = (layer.name, layerDataLoopDomain(layer, data, idcs))
        return processed


class VertexAttrGroupExporter():
    """Exports a single MDL0 vertex attribute group."""

    def __init__(self, loopData: np.ndarray, attrName = ""):
        self.attrName = attrName
        self.loopData = loopData
        self.groupData, self.loopDataIdcs = np.unique(loopData, return_inverse=True, axis=0)

    def getName(self, baseName = "", index = 0):
        """Generate a name for this group based on an ID for the base name & an optional index."""
        name = baseName
        if self.attrName:
            name = "__".join((name, self.attrName))
        if index:
            name = "_".join((name, str(index)))
        return name

    def exportGroup(self, groupType: type[mdl0.VertexAttrGroup], baseName = "", index = 0):
        """Export a group for this exporter, w/ a name based on baseName and index."""
        return groupType(self.getName(baseName, index), self.groupData)


class GeometrySlice():
    """Represents a slice of a GeometryInfo containing an arbitrary subset of its triangles."""

    def __init__(self, geoInfo: GeometryInfo, usedTriLoopIdcs: np.ndarray):
        self.geoInfo = geoInfo
        self.usedTriLoopIdcs = usedTriLoopIdcs
        self.usedLoopIdcs = np.unique(self.usedTriLoopIdcs)
        self.attrExporters: dict[type[mdl0.VertexAttrGroup], dict[int, VertexAttrGroupExporter]] = {
            mdl0.PsnGroup: {0: VertexAttrGroupExporter(geoInfo.loopPsns[self.usedLoopIdcs])},
            mdl0.NrmGroup: {0: VertexAttrGroupExporter(geoInfo.loopNrms[self.usedLoopIdcs])},
            mdl0.ClrGroup: self.createAttrGroupExporters(geoInfo.loopClrs),
            mdl0.UVGroup: self.createAttrGroupExporters(geoInfo.loopUVs)
        }

    def createAttrGroupExporters(self, groupInfo: dict[int, tuple[str, np.ndarray]]):
        """Generate a dict of vertex attr group exporters from group names & data."""
        loopFilter = self.usedLoopIdcs
        return {i: VertexAttrGroupExporter(d[loopFilter], n) for i, (n, d) in groupInfo.items()}

    def exportAttrGroups(self, meshName = "", objName = "", matName = "", sliceIdx = 0):
        """Create all of the attribute groups for this geometry slice.

        The mesh name, object name, material name, and slice index are all used for the group names.
        """
        exported = {}
        for groupType, exportSlots in self.attrExporters.items():
            exportedSlots = exported.setdefault(groupType, {})
            baseName = "__".join((objName if groupType is mdl0.PsnGroup else meshName, matName))
            for slot, exporter in exportSlots.items():
                exportedSlots[slot] = exporter.exportGroup(groupType, baseName, sliceIdx)
        return exported


class MeshExporter():

    def __init__(self, parentMdlExporter: "BRRESMdlExporter"):
        self.parentMdlExporter = parentMdlExporter
        self.meshes: list[mdl0.Mesh] = []
        self.mesh: bpy.types.Mesh = None
        self.meshSettings: MeshSettings = None
        self.obj: bpy.types.Object = None
        self._geoInfo = GeometryInfo()
        self._singleBind: mdl0.Joint = None
        self._visJoint: mdl0.Joint = None

    def update(self, mesh: bpy.types.Mesh, obj: bpy.types.Object):
        """Update this exporter and its parent with BRRES data based on a mesh object."""
        parentMdlExporter = self.parentMdlExporter
        self._singleBind = parentMdlExporter.getParentJoint(obj)
        self._visJoint = parentMdlExporter.getVisJoint(obj)
        self.mesh = mesh
        try:
            self.meshSettings = obj.original.data.brres
        except AttributeError:
            self.meshSettings = mesh.brres
        self.obj = obj
        # get geometry info
        geoInfo = self._geoInfo
        geoScale = parentMdlExporter.parentResExporter.settings.scale
        geoInfo.update(mesh, obj, geoScale)
        if self._singleBind is None:
            geoInfo.updateSkinning(mesh, obj, parentMdlExporter)
        geoInfo.simplifyData()
        # generate brres mesh for each material used
        maxAttrGroupLen = mdl0.VertexAttrGroup.MAX_LEN
        usedSlots = usedMatSlots(obj, mesh)
        for matSlot in obj.material_slots:
            blendMat = matSlot.material
            if not blendMat:
                continue
            # note that material is always exported, regardless of whether it's used for geometry
            # (handy for working around game hardcodes that need materials to exist in certain ways)
            mat = parentMdlExporter.exportMaterial(blendMat)
            if matSlot not in usedSlots:
                continue
            curMatTriLoopIdcs = geoInfo.triLoopIdcs[geoInfo.triMatIdcs == matSlot.slot_index]
            for sliceIdx, triStart in enumerate(range(0, len(curMatTriLoopIdcs), maxAttrGroupLen)):
                usedTriLoopIdcs = curMatTriLoopIdcs[triStart : triStart + maxAttrGroupLen]
                geoSlice = GeometrySlice(geoInfo, usedTriLoopIdcs)
                brresMesh = self.generateMesh(mat, geoSlice, sliceIdx)
                self.meshes.append(brresMesh)
                for groups in brresMesh.vertGroups.values():
                    for group in groups.values():
                        parentMdlExporter.exportAttrGroup(group)

    def generateMesh(self, mat: mdl0.Material, geoSlice: GeometrySlice, sliceIdx: int):
        """Generate a BRRES mesh & vertex groups for some geometry."""
        sliceSuffix = f"_{sliceIdx}" if sliceIdx else ""
        brresMesh = mdl0.Mesh(f"{self.obj.name}__{mat.name}{sliceSuffix}")
        if self._singleBind:
            brresMesh.singleBind = self._singleBind.deformer
        brresMesh.visJoint = self._visJoint
        brresMesh.vertGroups = geoSlice.exportAttrGroups(self.mesh.name, self.obj.name,
                                                         mat.name, sliceIdx)
        brresMesh.mat = mat
        brresMesh.drawPrio = self.meshSettings.drawPrio if self.meshSettings.enableDrawPrio else 0
        # separate primitives into draw groups
        drawGroupData = self._getDrawGroupData(geoSlice)
        # generate primitive commands
        for dgDfs, dgFaces in drawGroupData:
            dg = self._exportDrawGroup(mat, geoSlice, dgDfs, dgFaces)
            brresMesh.drawGroups.append(dg)
        return brresMesh

    def _getDrawGroupData(self, geoSlice: GeometrySlice):
        """Separate a geometry slice into tuples representing BRRES mesh draw groups.

        Each tuple contains a set of its used deformer indices, as well as a list of arrays with its
        loop indices.
        """
        # note: if skinning isn't used, we just shove everything into one draw group
        # however, if it is used, we have to use separate draw groups because each one is limited
        # to just 10 deformers (weighted joint combinations)
        geoInfo = self._geoInfo
        drawGroupData: list[tuple[set[int], list[np.ndarray]]] = []
        if not self._singleBind:
            # for each face, go through the existing draw groups.
            # if there's a draw group found that supports this face (i.e., all the face's
            # deformers are in the draw group, or it has room to add those deformers), add
            # the face (and its deformers if necessary) to the draw group.
            # otherwise, make a new draw group and add this face & its deformers to it.
            # this approach feels naive, but it seems to work well enough for now
            # (btw based on a quick code glimpse, i think this is also what brawlcrate does)
            # for each used face
            for face in geoSlice.usedTriLoopIdcs.reshape(-1, 3):
                groupFound = False
                uniqueFaceDfs = set(geoInfo.loopDfIdcs[face])
                # go through existing draw groups
                for dgDfs, dgFaces in drawGroupData:
                    newDfs = uniqueFaceDfs.difference(dgDfs)
                    # if there's a group found that supports this face (i.e., all the
                    # face's deformers are in the draw group, or it has room to add those
                    # deformers), add the face (and its deformers if necessary) to the
                    # draw group
                    if len(dgDfs) <= gx.MAX_ATTR_MTCS - len(newDfs):
                        dgDfs |= newDfs
                        dgFaces.append(face)
                        groupFound = True
                        break
                if not groupFound:
                    newGroup = (uniqueFaceDfs, [face])
                    drawGroupData.append(newGroup)
        else:
            drawGroupData.append((set(), [geoSlice.usedTriLoopIdcs]))
        return drawGroupData

    def _exportDrawGroup(self, mat: mdl0.Material, geoSlice: GeometrySlice,
                        dgDfs: set[int], dgFaces: list[np.ndarray]):
        """Turn a tuple generated by getDrawGroupData() into an actual BRRES mesh draw group."""
        dg = mdl0.DrawGroup()
        geoInfo = self._geoInfo
        dg.deformers = [geoInfo.vertDfs[geoInfo.dfVertIdcs[dfIdx]] for dfIdx in dgDfs]
        # get loops used by this draw group
        # dgAbsLoopIdcs has absolute indices, dgUsedLoopIdcs is relative to used loops
        dgAbsLoopIdcs = np.concatenate(dgFaces)
        dgUsedLoopIdcs = np.searchsorted(geoSlice.usedLoopIdcs, dgAbsLoopIdcs)
        numLoops = len(dgUsedLoopIdcs)
        # set up command w/ basic vertex attrs
        # this is not used in the model, it's just used to store the attrs
        # and then it's converted to triangle strips, which are stored, for compression
        cmd = gx.DrawTriangles(numLoops)
        cmdAttrs = (cmd.psns, cmd.nrms, cmd.clrs, cmd.uvs)
        for groups, cmdAttr in zip(geoSlice.attrExporters.values(), cmdAttrs):
            for groupExp, cmdData in zip(groups.values(), cmdAttr):
                cmdData[:] = groupExp.loopDataIdcs[dgUsedLoopIdcs]
        # then set up matrix attrs (for skinning)
        if dgDfs:
            # we have absolute df indices, but we need relative to this dg's df list
            # we get this using np.searchsorted(), based on this
            # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
            # (accepted answer, not most upvoted; latter answers the wrong question)
            dgAbsLoopDfs = geoInfo.loopDfIdcs[dgAbsLoopIdcs] # abs df index for each loop in dg
            dgDfs = np.array(tuple(dgDfs))
            sortedDgDfIdcs = np.argsort(dgDfs)
            sortedDgDfs = dgDfs[sortedDgDfIdcs]
            dgLoopDfs = sortedDgDfIdcs[np.searchsorted(sortedDgDfs, dgAbsLoopDfs)]
            dgLoopDfs = dgLoopDfs.reshape(1, -1) # reshape for use in commands
            # now, just put the indices (converted to addresses) in the commands
            dgLoopPsnMtxAddrs = gx.LoadPsnMtx.idxToAddr(dgLoopDfs) // 4
            dgLoopTexMtxAddrs = gx.LoadTexMtx.idxToAddr(dgLoopDfs) // 4
            cmd.psnMtcs = dgLoopPsnMtxAddrs
            for i, tex in enumerate(mat.textures):
                if tex.mapMode is not mdl0.TexMapMode.UV:
                    cmd.texMtcs[i] = dgLoopTexMtxAddrs
        # apply triangle stripping & return
        dg.cmds[:] = cmd.strip()
        return dg


class BRRESMdlExporter():

    def __init__(self, parentResExporter: "BRRESExporter", armObj: bpy.types.Object):
        settings = parentResExporter.settings
        self.parentResExporter = parentResExporter
        self.model = mdl0.MDL0(armObj.name)
        self.armObj = armObj
        self.parentResExporter.res.folder(mdl0.MDL0).append(self.model)
        # generate joints
        arm: bpy.types.Armature = armObj.data
        restorePosePosition = arm.pose_position
        if not settings.useCurrentPose:
            arm.pose_position = 'REST' # temporarily put rig in rest position (pose restored at end)
            parentResExporter.depsgraph.update()
        # hasExtraRoot is usually false here (extra root hasn't been created),
        # but if the option is "never" and the armature alreaday has a root, new root creation is
        # bypassed by setting it to true
        # btw even if create new root is "never", we do still need to make a root if there are
        # no bones, which is why that check exists
        self._hasExtraRoot = settings.addNewRoots == 'NEVER' and len(arm.bones) > 0
        self.joints: dict[str, mdl0.Joint] = {}
        self._exportJoints(armObj)
        if settings.addNewRoots == 'ALWAYS':
            # this ensures extra root always exists, even if not used
            self._extraRoot()
        # generate meshes & everything they use
        self._mats: dict[str, mdl0.Material] = {}
        self.tevConfigs: dict[str, mdl0.TEVConfig] = {}
        self._meshes: dict[bpy.types.Object, list[mdl0.Mesh]] = {}
        for obj in bpy.data.objects:
            if not parentResExporter.limiter.includes(obj):
                continue
            if settings.applyModifiers:
                obj: bpy.types.Object = obj.evaluated_get(parentResExporter.depsgraph)
            parent = obj.parent
            if parent is None or parent.type != 'ARMATURE' or parent.data.name != arm.name:
                continue
            self.exportMeshObj(obj)
        # remove unused joints if enabled
        if settings.removeUnusedBones:
            self._removeUnusedJoints()
        # sort materials so arbitrary draw order (when group & prio are equal) is based on name
        self.model.mats.sort(key=lambda mat: mat.name)
        # restore current rig pose in case we changed it to ignore it
        arm.pose_position = restorePosePosition

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
            if uiImg not in self.parentResExporter.images and self.parentResExporter.onlyUsedImg:
                self.parentResExporter.exportImg(uiImg)
        # texture
        tex = mdl0.Texture()
        parentMat.textures.append(tex)
        tex.imgName = self.parentResExporter.imgName(uiImg)
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

    def exportMaterial(self, mat: bpy.types.Material):
        """Export a Blender material to a BRRES material, added to this model.

        Used images are exported as well. If a BRRES material already exists
        for this Blender material, it is returned.
        """
        if mat.name in self._mats:
            return self._mats[mat.name]
        brresMat = mdl0.Material(mat.name)
        self._mats[mat.name] = brresMat
        self.model.mats.append(brresMat)
        brresMatSettings = mat.brres
        # tev
        try:
            tevConfigs = self.parentResExporter.context.scene.brres.tevConfigs
            tevSettings = tevConfigs[brresMatSettings.tevID]
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
        return brresMat

    def getParentJoint(self, obj: bpy.types.Object):
        """Get a Blender object's MDL joint parent (None if skinning used)."""
        if self.hasBoneParent(obj):
            return self.joints[obj.parent_bone]
        elif self.hasSkinning(obj):
            return None
        # if we get here, object is a direct child of the armature (no bone) so make extra root
        return self._extraRoot()

    def getVisJoint(self, obj: bpy.types.Object):
        """Get a Blender object's MDL visibility joint."""
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
                            bone = self.armObj.path_resolve(dataPath[:-boneVisSuffixLen])
                            if not isinstance(bone, bpy.types.Bone):
                                continue
                        except ValueError:
                            continue # not a bone visibility path
                        return self.joints[bone.name]
        # object doesn't have visibility driver, so use extra root for visibility joint
        return self._extraRoot()

    def _applyTransform(self, mesh: bpy.types.Mesh, obj: bpy.types.Object):
        """
        Transform a mesh object to prep it for BRRES export.

        This should only be used with temporary meshes created via to_mesh(); do not modify actual
        blendfile data!
        """
        modelMtx = obj.matrix_local.copy()
        if self.hasBoneParent(obj):
            parentBone: bpy.types.Bone = obj.parent.data.bones[obj.parent_bone]
            if parentBone.use_relative_parent:
                # "relative parenting" means parent matrix isn't accounted for automatically
                modelMtx = parentBone.matrix_local.inverted() @ modelMtx
            else:
                # object is positioned relative to bone tail, but we want relative to head
                headCorrection = Matrix.Translation((0, parentBone.length, 0))
                modelMtx = headCorrection @ modelMtx
            # correct for bone space
            mtxBoneToBRRES = self.parentResExporter.mtxBoneToBRRES.to_4x4()
            coordConversion = MTX_FROM_BONE.to_4x4() @ mtxBoneToBRRES
            modelMtx = coordConversion @ modelMtx
        mesh.transform(MTX_TO_BRRES @ modelMtx)

    def getVertDfs(self, mesh: bpy.types.Mesh, obj: bpy.types.Object) -> list[mdl0.Deformer]:
        """Get a list with the MDL0 deformer for each vertex of a mesh object."""
        settings = self.parentResExporter.settings
        vertDfs = [{} for _ in range(len(mesh.vertices))]
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

    def exportMeshObj(self, obj: bpy.types.Object) -> list[mdl0.Mesh]:
        """Export a Blender mesh object to a list of BRRES meshes, added to this model.

        Dependencies such as materials are exported as well. If BRRES meshes already exist
        for this object, these are returned. If the object doesn't support a mesh
        (i.e., it's a light, camera, etc.), the returned list is empty.
        """
        if obj in self._meshes:
            return self._meshes[obj]
        # generate mesh
        try:
            depsgraph = self.parentResExporter.depsgraph
            mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        except RuntimeError: # object type doesn't support meshes
            return []
        # process mesh & export to brres meshes
        self._applyTransform(mesh, obj)
        mesh.calc_normals_split()
        mesh.calc_loop_triangles()
        meshExporter = MeshExporter(self)
        meshExporter.update(mesh, obj)
        # add to this model & return
        self.model.meshes += meshExporter.meshes
        self._meshes[obj] = meshExporter.meshes
        return meshExporter.meshes

    def exportAttrGroup(self, group: mdl0.VertexAttrGroup):
        self.model.vertGroups[type(group)].append(group)

    def _exportJoints(self, armObj: bpy.types.Object):
        parentResExporter = self.parentResExporter
        mtcs = {bone: bone.matrix for bone in armObj.pose.bones}
        mtcs = {bone: (mtx, mtx.inverted()) for bone, mtx in mtcs.items()}
        localScales = {} # for segment scale compensate calculations
        prevRots = {} # for euler compatibility
        for poseBone in armObj.pose.bones:
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
            srt = np.array(parentResExporter.getLocalSRT(poseBone, localScales, mtcs, prevRots))
            srt[np.isclose(srt, 0, atol=0.001)] = 0
            joint.setSRT(*srt)
        for boneName, joint in self.joints.items():
            boneSettings = armObj.data.bones[boneName].brres
            joint.bbMode = mdl0.BillboardMode[boneSettings.bbMode]
            try:
                joint.bbParent = self.joints[boneSettings.bbParent]
            except KeyError:
                pass

    def _extraRoot(self):
        """Extra root joint to be used in case of multiple roots or objects without bone parenting.

        (This may be a dynamically created bonus joint, or just the regular root, depending on
        "Create New Roots")"""
        if self._hasExtraRoot:
            return self.model.rootJoint
        self._hasExtraRoot = True
        usedBoneNames = {b.name for b in self.armObj.data.bones}
        extraRoot = mdl0.Joint(name=makeUniqueName(self.armObj.data.name, usedBoneNames))
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

    def hasBoneParent(self, obj: bpy.types.Object):
        """True if an object is a child of a bone in this exporter's armature."""
        return (
            obj.parent.original is self.armObj
            and obj.parent_type == 'BONE'
            and obj.parent_bone in self.joints
        )

    def hasSkinning(self, obj: bpy.types.Object):
        """True if an object uses this exporter's armature for deformation. (Not just parenting)"""
        if obj.parent.original is not self.armObj or obj.parent_type == 'BONE':
            return False
        if obj.parent_type == 'ARMATURE':
            return True
        for m in obj.modifiers:
            if m.type == 'ARMATURE' and m.object is not None and m.object.original is self.armObj:
                return True
        return False


class BRRESAnimExporter(Generic[ANIM_SUBFILE_T]):

    ANIM_TYPE: type[ANIM_SUBFILE_T]

    def __init__(self, parentResExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        self.parentResExporter = parentResExporter
        self.track = track

    def getAnim(self, name: str, action: bpy.types.Action) -> ANIM_SUBFILE_T:
        """Get an animation of this exporter's type from its parent BRRES exporter by name.

        If the requested animation doesn't exist yet, a new one is created.

        In either case, its length and looping flag are updated based on its current settings & 
        the provided action.
        """
        try:
            anim = self.parentResExporter.anims[type(self)][name]
        except KeyError:
            anim = self.ANIM_TYPE(name)
            self.parentResExporter.anims[type(self)][name] = anim
            self.parentResExporter.res.folder(self.ANIM_TYPE).append(anim)
        anim.enableLoop = action.use_cyclic
        # add 1 as brres "length" is number of frames (including both endpoints),
        # as opposed to length of frame span (which doesn't include one endpoint, and is what
        # we have here)
        frameStart = self.parentResExporter.settings.frameStart
        newLen = int(np.ceil(action.frame_range[1] + 1 - frameStart))
        anim.length = max(anim.length, newLen)
        return anim


class BRRESChrExporter(BRRESAnimExporter[chr0.CHR0]):

    ANIM_TYPE = chr0.CHR0

    def __init__(self, parentResExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentResExporter, track)
        # get action & rig object
        # also, get some settings we'll have to alter for baking so we can restore them later
        settings = parentResExporter.settings
        scene = parentResExporter.context.scene
        viewLayer = parentResExporter.context.view_layer
        strip = track.strips[0]
        action = strip.action
        frameStart, frameEnd = action.frame_range
        armObj: bpy.types.Object = track.id_data
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
        bones = armObj.pose.bones
        jointFrames = {bone: np.empty((numFrames, 3, 3)) for bone in bones}
        hasChild = {bone: bool(bone.children) for bone in bones}
        subframes = frameRange % 1
        roundedFrames = frameRange.astype(int)
        scales = {} # for segment scale compensate calculations
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
                    frameVals[kfIdx] = parentResExporter.getLocalSRT(bone, scales, mtcs, prevRots)
                    lastNewFrame[bone] = kfIdx
        for bone, lastNewFrameIdx in lastNewFrame.items():
            frameVals = jointFrames[bone]
            frameVals[lastNewFrameIdx:] = frameVals[lastNewFrameIdx]
        # create chr0
        chrAnim = self.getAnim(track.name, action)
        usedJointNames = {jointAnim.jointName for jointAnim in chrAnim.jointAnims}
        jointAnims: dict[bpy.types.PoseBone, chr0.JointAnim] = {}
        for bone, frameVals in jointFrames.items():
            if bone.name in usedJointNames:
                continue # ignore bones already in chr0, in case it already existed from another rig
            jointAnims[bone] = jointAnim = chr0.JointAnim(bone.name)
            jointAnim.animFmts[:] = (animation.I12, animation.I12, animation.I12)
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
                    anim.length = chrAnim.length
                    # filter out frames w/ the same values as prev & next frames
                    eqNext = np.isclose(compVals[:-1], compVals[1:])
                    dupFrames = np.logical_and(eqNext[:-1], eqNext[1:])
                    # note: last frame is filtered out if it equals prev, but first is always needed
                    frameFltr = np.logical_not(np.concatenate(([False], dupFrames, eqNext[-1:])))
                    keyframes = emptyKfs[frameFltr].copy()
                    keyframes[:, 1] = compVals[frameFltr]
                    anim.keyframes = keyframes
                    # then, further simplify lossily if enabled
                    if settings.doAnimSimplify:
                        anim.setSmooth()
                        anim.simplify(settings.animMaxError)
            # determine whether discrete format is worth it (would save space) for rotation
            # (only test for rotation as it's not supported for scale or translation)
            rotLength = sum(max(a.keyframes[-1, 0], 0) + 1 for a in jointAnim.rot)
            rotFrames = sum(len(a.keyframes) for a in jointAnim.rot)
            if rotFrames * 12 > rotLength * 4: # rotFrames is # frames for i12, rotLength is for d4
                jointAnim.animFmts[1] = animation.D4
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

    def __init__(self, parentResExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentResExporter, track)
        frameStart = parentResExporter.settings.frameStart
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

    def __init__(self, parentResExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentResExporter, track)
        frameStart = parentResExporter.settings.frameStart
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
                img = texImg.img
                if img and img not in parentResExporter.images and parentResExporter.onlyUsedImg:
                    self.parentResExporter.exportImg(img)
                texAnim.texNames.append(parentResExporter.imgName(img))
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

    def __init__(self, parentResExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentResExporter, track)
        settings = parentResExporter.settings
        frameStart = settings.frameStart
        strip = track.strips[0]
        action = strip.action
        matAnim = srt0.MatAnim(strip.id_data.name)
        pathInfo = self.pathInfo()
        # look through fcurves, updating mat animation if any are relevant to srt0
        for fcurve in action.fcurves:
            # parse data path to get texture index & specific property
            # if path is invalid for srt0, skip this fcurve
            try:
                texAnimCollName, texIdx, texProp = pathInfo[fcurve.data_path]
            except KeyError:
                continue
            # grab existing texture animation or create new one if this is the first fcurve read
            # that uses this texture
            texAnimColl: dict[int, srt0.TexAnim] = getattr(matAnim, texAnimCollName)
            try:
                texAnim = texAnimColl[texIdx]
            except KeyError:
                texAnim = srt0.TexAnim()
                try:
                    # try to set default transform values to those actually used in the model
                    # (by default, they're just identity, which may not be desired)
                    texTransform = strip.id_data.brres.textures[texIdx].transform
                    texAnim.scale[0].keyframes[0, 1] = texTransform.scale[0]
                    texAnim.scale[1].keyframes[0, 1] = texTransform.scale[1]
                    texAnim.rot[0].keyframes[0, 1] = texTransform.rotation
                    texAnim.trans[0].keyframes[0, 1] = texTransform.translation[0]
                    texAnim.trans[1].keyframes[0, 1] = texTransform.translation[1]
                except IndexError:
                    # texture doesn't actually exist - whatever, just use default values (identity)
                    pass
                texAnimColl[texIdx] = texAnim
            compAnim: animation.Animation = getattr(texAnim, texProp)[fcurve.array_index]
            # fill out animation data by evaluating curve
            # maybe this should be proper exact conversion to hermite someday?
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
            # if all keyframes are same, reduce to 1
            if len(set(frameVals)) == 1:
                compAnim.keyframes = compAnim.keyframes[:1]
            # then, further simplify lossily if enabled
            if settings.doAnimSimplify:
                compAnim.setSmooth()
                compAnim.simplify(settings.animMaxError)
        # if mat anim is non-empty (relevant fcurves were found), update srt anim
        if matAnim.texAnims or matAnim.indAnims:
            srtAnim = self.getAnim(track.name, action)
            srtAnim.mtxGen = tf.MayaMtxGen2D
            srtAnim.matAnims.append(matAnim)


class BRRESVisExporter(BRRESAnimExporter[vis0.VIS0]):

    ANIM_TYPE = vis0.VIS0

    def __init__(self, parentResExporter: "BRRESExporter", track: bpy.types.NlaTrack):
        super().__init__(parentResExporter, track)
        frameStart = parentResExporter.settings.frameStart
        strip = track.strips[0]
        action = strip.action
        arm: bpy.types.Object | bpy.types.Armature = track.id_data
        boneVisSuffix = ".hide"
        boneVisSuffixLen = len(boneVisSuffix)
        jointAnims: list[vis0.JointAnim] = []
        for fcurve in action.fcurves:
            dataPath = fcurve.data_path
            if not dataPath.endswith(boneVisSuffix):
                continue
            try:
                # cut off ".hide" to get bone
                bone: bpy.types.Bone = arm.path_resolve(dataPath[:-boneVisSuffixLen])
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

    def __init__(self, settings: "ExportBRRES", limiter: ObjectLimiter):
        self.res = brres.BRRES()
        self.settings = settings
        self.limiter = limiter
        self.context: bpy.types.Context = None
        self.depsgraph: bpy.types.Depsgraph = None
        self.models: dict[bpy.types.Object, BRRESMdlExporter] = {}
        self.images: dict[bpy.types.Image, tex0.TEX0] = {}
        self.anims: dict[type[BRRESAnimExporter], dict[str, animation.AnimSubfile]]
        self.anims = {t: {} for t in (
            BRRESChrExporter, BRRESClrExporter, BRRESPatExporter, BRRESSrtExporter, BRRESVisExporter
        )}
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

    def update(self, context: bpy.types.Context):
        """Update this exporter's BRRES data based on the current Blender context & data."""
        self.context = context
        self.depsgraph = context.evaluated_depsgraph_get()
        settings = self.settings
        if settings.includeUnusedImg:
            # export all images
            # (if this setting's disabled, image export is handled by model/anim exporters)
            for img in bpy.data.images:
                self.exportImg(img)
        for obj in bpy.data.objects:
            # export armatures included in limit, as well as armature animations (chr/vis)
            if obj.type == 'ARMATURE' and self.limiter.includes(obj):
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
            mats = {bpy.data.materials[mat] for mdl in self.models.values() for mat in mdl._mats}
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

    def merge(self, baseRes: brres.BRRES):
        """Replace this exporter's BRRES with a new "base", and merge the original subfiles into it.

        In case of naming conflicts, subfiles in the base BRRES will be overwritten.
        """
        for fType, folder in self.res.files.items():
            # get files from base folder & new folder and overwrite those w/ same names
            baseFolder = baseRes.folder(fType)
            baseFiles = {f.name: f for f in baseFolder}
            newFiles = {f.name: f for f in folder}
            baseFolder[:] = (baseFiles | newFiles).values() # in conflicts, new files win
        self.res = baseRes

    @classmethod
    def export(cls, context: bpy.types.Context, settings: "ExportBRRES",
               limiter: ObjectLimiter, baseData = b"") -> bytes:
        """Export a BRRES based on a Blender context & settings.
        
        Data for a "base BRRES" can be provided for merging (though this is only used if enabled in
        the settings).
        """
        exporter = cls(settings, limiter)
        exporter.update(context)
        # optionally merge with existing file
        if settings.doMerge and baseData:
            exporter.merge(brres.BRRES.unpack(baseData))
        # write file
        exporter.res.sort()
        packed = exporter.res.pack()
        packed += binaryutils.pad(f"BerryBush {verStr(addonVer())}".encode("ascii"), 16)
        if settings.padEnable:
            packed = binaryutils.pad(packed, settings.padSize * int(settings.padUnit))
        return packed

    def _exportModel(self, armObj: bpy.types.Object):
        """Export a MDL0 based on a Blender armature object and add it to this BRRES."""
        self.models[armObj] = BRRESMdlExporter(self, armObj)

    def exportImg(self, bImg: bpy.types.Image):
        """Export a TEX0 based on a Blender image and add it to this BRRES."""
        img = tex0.TEX0(bImg.name)
        self.images[bImg] = img
        self.res.folder(tex0.TEX0).append(img)
        img.fmt = BlendImageExtractor.getFormat(bImg)
        px = BlendImageExtractor.getRgba(bImg)
        img.images.append(px[::-1])
        dims = np.array(px.shape[:2][::-1], dtype=np.integer)
        for mm in bImg.brres.mipmaps:
            dims //= 2
            mmPx = BlendImageExtractor.getRgbaWithDims(mm.img, dims)[::-1]
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
        """Attempt to export an animation based on a Blender NLA track and add it to this BRRES.

        (Note that if animation data turns out to be empty, the animation is discarded)
        """
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


class ExportSettingsMixin():
    """Contains bpy property definitions for the BRRES export settings.
    
    This class is designed as a mixin rather than a direct child of `bpy.types.PropertyGroup` so
    that these settings can be included in the export operator directly, enabling easy customization
    of export hotkeys within Blender's UI.
    """

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
        default=True
    )

    addNewRoots: bpy.props.EnumProperty(
        name="Add New Roots",
        description="When to add new root bones to BRRES MDLs",
        items=(
            ('ALWAYS', "Always", "Always create a new root bone for each armature, named after the armature"), # pylint: disable=line-too-long
            ('NECESSARY', "When Necessary", "Only create new root bones when needed for assets without all BRRES bone relations (parenting/skinning, visibility) set"), # pylint: disable=line-too-long
            ('NEVER', "Never", "Never create new root bones, and use existing roots when needed for assets without all BRRES bone relations (parenting/skinning, visibility) set. Note that if multiple root bones exist in the initial armature, one will be chosen arbitrarily as the exported root, and the others will be exported as its children."), # pylint: disable=line-too-long
        ),
        default='NECESSARY',
    )

    removeUnusedBones: bpy.props.BoolProperty(
        name="Remove Unused Bones",
        description="Only export bones used for deforming and visibility (and their ancestors). Other bones are still taken into account for constraint & driver evaluation, but not exported",  # pylint: disable=line-too-long
        default=False
    )

    doQuantize: bpy.props.BoolProperty(
        name="Simplify Weights",
        description="Quantize vertex weights to a fixed interval before normalization (good for file size, performance, & preventing crashes)", # pylint: disable=line-too-long
        default=True,
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

    doAnimSimplify: bpy.props.BoolProperty(
        name="Simplify Curves",
        description="Lossily compress CHR & SRT animations by simplifying baked animation data",
        default=True
    )

    animMaxError: bpy.props.FloatProperty(
        name="Simplify Curves",
        description="How much the exported animation is allowed to deviate from the original at any given frame", # pylint: disable=line-too-long
        min=0,
        default=.01
    )

    frameStart: bpy.props.IntProperty(
        name="Frame Start",
        description="First frame of animation",
        default=1
    )


class ExportSettings(ExportSettingsMixin, bpy.types.PropertyGroup):
    pass


class ExportBRRES(bpy.types.Operator, ExportHelper, ExportSettingsMixin):
    """Write a BRRES file"""

    bl_idname = "export_scene.brres"
    bl_label = "Export BRRES"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".brres"
    filter_glob: bpy.props.StringProperty(
        default="*.brres",
        options={'HIDDEN'},
    )

    def verify(self, context: bpy.types.Context, limiter: ObjectLimiter):
        """Run the BRRES verifier based on the data exported by this exporter."""
        name = f"\"{os.path.basename(self.filepath)}\""
        warns, suppressed = verifyBRRES(self, context, limiter)
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

    def loadSettings(self, scene: bpy.types.Scene):
        """Load the saved export settings from a scene."""
        settings: ExportSettings = scene.brres.exportSettings
        for key, value in settings.items():
            try:
                # only load settings that haven't already been set by a hotkey
                if key not in self.properties: # pylint: disable=unsupported-membership-test
                    setattr(self, key, value)
            except (AttributeError, TypeError):
                pass

    def saveSettings(self, scene: bpy.types.Scene):
        """Save the current export settings to a scene."""
        settings: ExportSettings = scene.brres.exportSettings
        for key in self.properties.keys():
            try:
                settings[key] = getattr(self, key)
            except (AttributeError, TypeError):
                pass

    def invoke(self, context, event):
        self.loadSettings(context.scene)
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        profiler = Profile()
        profiler.enable()
        restoreShading = solidView(context) # temporarily set viewports to solid view for speed
        context.window.cursor_set('WAIT')
        self.report({'INFO'}, "Exporting BRRES...")
        baseData = b""
        tryBackup(self.filepath, context)
        if self.doMerge: # get existing data for merge/backup
            try:
                with open(self.filepath, "rb") as f:
                    baseData = f.read()
            except FileNotFoundError:
                pass
        limiter = ObjectLimiterFactory.create(context, self.limitTo)
        with open(self.filepath, "wb") as f: # export main file
            f.write(BRRESExporter.export(context, self, limiter, baseData))
        self.verify(context, limiter)
        self.saveSettings(context.scene)
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
        layout.prop(settings, "removeUnusedBones")
        layout.prop(settings, "addNewRoots")
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
        drawCheckedProp(layout, settings, "doAnimSimplify", settings, "animMaxError")
        layout.prop(settings, "frameStart")
