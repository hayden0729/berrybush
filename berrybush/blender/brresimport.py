# standard imports
from cProfile import Profile
import os
from pstats import SortKey, Stats
from typing import Generic, TypeVar
# 3rd party imports
import bpy
from bpy_extras.io_utils import ImportHelper, axis_conversion
import bmesh
from mathutils import Matrix
import numpy as np
# internal imports
from .common import (
    MTX_FROM_BRRES, MTX_FROM_BONE, MTX_TO_BONE, LOG_PATH,
    transformAxis, solidView, restoreView, getLoopVertIdcs, enumVal, foreachGet
)
from .material import AlphaSettings, DepthSettings, LightChannelSettings, MiscMatSettings
from .texture import TexSettings
from .tev import TevStageSettings
from ..wii import (
    animation, brres, chr0, clr0, gx, mdl0, pat0, plt0, srt0, tex0, transform as tf, vis0
)


ANIM_SUBFILE_T = TypeVar("ANIM_SUBFILE_T", bound=animation.AnimSubfile)


TEX_TRANSFORM_MODES = {
    tf.MayaMtxGen2D: 'MAYA',
    tf.XSIMtxGen2D: 'XSI',
    tf.MaxMtxGen2D: 'MAX'
}


class BRRESMdlImporter():

    def __init__(self, parentImporter: "BRRESImporter", model: mdl0.MDL0):
        self.parentImporter = parentImporter
        self.model = model
        # load joints
        rig = bpy.data.armatures.new(model.name)
        rigObj = bpy.data.objects.new(model.name, rig)
        rigObj.show_in_front = True
        parentImporter.context.collection.objects.link(rigObj)
        self._rigObj = rigObj.name
        self.bones: dict[mdl0.Joint, str] = {}
        self._sscControls: dict[mdl0.Joint, str] = {}
        self._scaledJointMtcs: dict[mdl0.Joint, np.ndarray] = {}
        self._loadJoints(model)
        # load tev configs
        self.tevConfigs: dict[mdl0.TEVConfig, str] = {}
        for tevConfig in model.tevConfigs:
            self._loadTevConfig(tevConfig)
        # load materials
        self.mats: dict[mdl0.Material, str] = {}
        for mat in model.mats:
            self._loadMat(mat)
        # load meshes
        self._loadMeshes(model)

    @property
    def rigObj(self):
        return bpy.data.objects[self._rigObj]

    def _importTevStage(self, stageSettings: TevStageSettings, stage: mdl0.TEVStage):
        # selections
        stageSettings.sels.texSlot = stage.texIdx + 1
        stageSettings.sels.texSwapSlot = stage.alphaParams.textureSwapIdx + 1
        stageSettings.sels.constColor = stage.constColorSel.name
        stageSettings.sels.constAlpha = stage.constAlphaSel.name
        stageSettings.sels.rasterSel = stage.rasterSel.name
        stageSettings.sels.rasSwapSlot = stage.alphaParams.rasterSwapIdx + 1
        # indirect texturing
        uiIndSettings = stageSettings.indSettings
        indSettings = stage.indSettings
        uiIndSettings.slot = indSettings.indirectID + 1
        uiIndSettings.fmt = indSettings.format.name
        uiIndSettings.enableBias = (indSettings.biasS, indSettings.biasT, indSettings.biasU)
        uiIndSettings.bumpAlphaComp = indSettings.bumpAlphaComp.name
        uiIndSettings.mtxType = indSettings.mtxType.name
        uiIndSettings.enable = indSettings.mtxIdx is not gx.IndMtxIdx.NONE
        if uiIndSettings.enable:
            uiIndSettings.mtxSlot = indSettings.mtxIdx.value - 1
        uiIndSettings.wrapU = indSettings.wrapS.name
        uiIndSettings.wrapV = indSettings.wrapT.name
        uiIndSettings.utcLOD = indSettings.utcLOD
        uiIndSettings.addPrev = indSettings.addPrev
        # color/alpha params
        modelCalcParams = (stage.colorParams, stage.alphaParams)
        uiCalcParams = (stageSettings.colorParams, stageSettings.alphaParams)
        for modelParams, uiParams in zip(modelCalcParams, uiCalcParams):
            uiParams.args = (arg.name for arg in modelParams.args)
            uiParams.clamp = modelParams.clamp
            uiParams.output = str(modelParams.output)
            uiParams.compMode = modelParams.bias is gx.TEVBias.COMPARISON_MODE
            if uiParams.compMode:
                uiParams.compOp = modelParams.op.name
                uiParams.compChan = modelParams.compareMode.name
            else:
                uiParams.bias = modelParams.bias.name
                uiParams.op = modelParams.op.name
                uiParams.scale = modelParams.scale.name

    def _loadTevConfig(self, tevConfig: mdl0.TEVConfig):
        tevSettings = self.parentImporter.context.scene.brres.tevConfigs.add(False)
        self.tevConfigs[tevConfig] = tevSettings.uuid
        # color swap table
        for mdlSwap, uiSwap in zip(tevConfig.colorSwaps, tevSettings.colorSwaps):
            uiSwap.r = mdlSwap.r.name
            uiSwap.g = mdlSwap.g.name
            uiSwap.b = mdlSwap.b.name
            uiSwap.a = mdlSwap.a.name
        # indirect sources
        tevSettings.indTexSlots = [idx + 1 for idx in tevConfig.indSources.texIdcs]
        # stages
        for i, stage in enumerate(tevConfig.stages):
            uiStage = tevSettings.stages.add(False)
            uiStage.name = f"Stage {i + 1}"
            self._importTevStage(uiStage, stage)
            if i == 0:
                # tev configs always need at least one stage
                # once we've added our first stage, we can remove the one that was already here
                tevSettings.stages.remove(0)

    def _importTex(self, texSettings: TexSettings, tex: mdl0.Texture):
        # image
        try:
            blendImgName = self.parentImporter.images[(tex.imgName, tex.pltName)]
            texSettings.imgs.add(False).img = bpy.data.images[blendImgName]
            # name texture after image
            # if image ends w/ period followed by numbers, remove that
            # (it seems to be a convention used for pat0 stuff, and this removal makes things nice)
            # (note: image itself keeps the ending)
            imgName = tex.imgName
            dotIdx = imgName.rfind(".")
            if imgName[dotIdx + 1:].isdigit():
                imgName = imgName[:dotIdx]
            texSettings.name = imgName
        except KeyError:
            pass # no image (or invalid)
        # texture
        t = texSettings.transform
        t.scale, t.rotation, t.translation = tex.scale, np.deg2rad(tex.rot), tex.trans
        if tex.mapMode is mdl0.TexMapMode.UV:
            texSettings.mapMode = f"{tex.mapMode.name}_{tex.coordIdx + 1}"
        else:
            texSettings.mapMode = tex.mapMode.name
        texSettings.wrapModeU, texSettings.wrapModeV = (m.name for m in tex.wrapModes)
        if tex.minFilter in (mdl0.MinFilter.NEAREST, mdl0.MinFilter.LINEAR):
            texSettings.minFilter = tex.minFilter.name
        else:
            texSettings.minFilter = mdl0.MinFilter((tex.minFilter.value - 2) % 2).name
            texSettings.mipFilter = mdl0.MinFilter((tex.minFilter.value - 2) // 2).name
        texSettings.magFilter = tex.magFilter.name
        texSettings.lodBias = tex.lodBias
        texSettings.maxAnisotropy = tex.maxAnisotropy.name
        texSettings.clampBias = tex.clampBias
        texSettings.texelInterpolate = tex.texelInterpolate
        texSettings.useCam = tex.usedCam > -1
        if texSettings.useCam:
            texSettings.camSlot = tex.usedCam + 1
        texSettings.useLight = tex.usedLight > -1
        if texSettings.useLight:
            texSettings.lightSlot = tex.usedLight + 1

    def _importLightChan(self, uiLc: LightChannelSettings, lc: mdl0.LightChannel):
        uiLc.difColor = lc.difColor
        uiLc.ambColor = lc.ambColor
        # control flags
        ccFlags = mdl0.LightChannel.ColorControlFlags
        lcControls = (lc.colorControl, lc.alphaControl)
        uiLcControls = (uiLc.colorSettings, uiLc.alphaSettings)
        for c, uic in zip(lcControls, uiLcControls):
            uic.difFromReg = ccFlags.DIFFUSE_FROM_VERTEX not in c
            uic.ambFromReg = ccFlags.AMBIENT_FROM_VERTEX not in c
            uic.enableDiffuse = ccFlags.DIFFUSE_ENABLE in c
            uic.enableAttenuation = ccFlags.ATTENUATION_ENABLE in c
            uic.attenuationMode = 'SPOTLIGHT' if ccFlags.ATTENUATION_SPOTLIGHT in c else 'SPECULAR'
            if ccFlags.DIFFUSE_SIGNED in c:
                uiLc.diffuseMode = 'SIGNED'
            if ccFlags.DIFFUSE_CLAMPED in c:
                uiLc.diffuseMode = 'CLAMPED'
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
            uic.enabledLights = [flag in c for flag in lightFlags]

    def _importAlphaSettings(self, alphaSettings: AlphaSettings, mat: mdl0.Material):
        blendSettings = mat.blendSettings
        alphaSettings.enableBlendOp = blendSettings.enableBlend
        if blendSettings.subtract:
            alphaSettings.enableBlendOp = True
            alphaSettings.blendOp = "-"
        alphaSettings.blendSrcFactor = blendSettings.srcFactor.name
        alphaSettings.blendDstFactor = blendSettings.dstFactor.name
        alphaSettings.enableLogicOp = blendSettings.enableLogic and not alphaSettings.enableBlendOp
        alphaSettings.logicOp = blendSettings.logic.name
        alphaSettings.enableDither = blendSettings.enableDither
        alphaSettings.enableColorUpdate = blendSettings.updateColor
        alphaSettings.enableAlphaUpdate = blendSettings.updateAlpha
        alphaSettings.enableConstVal = mat.constAlphaSettings.enable
        alphaSettings.constVal = mat.constAlphaSettings.value
        alphaSettings.cullMode = mat.cullMode.name
        alphaSettings.isXlu = mat.renderGroup is mdl0.RenderGroup.XLU
        testSettings = mat.alphaTestSettings
        alphaSettings.testComps = (c.name for c in testSettings.comps)
        alphaSettings.testVals = testSettings.values
        alphaSettings.testLogic = testSettings.logic.name

    def _importDepthSettings(self, depthSettings: DepthSettings, mat: mdl0.Material):
        depthSettings.enableDepthTest = mat.depthSettings.enable
        depthSettings.enableDepthUpdate = mat.depthSettings.updateDepth
        depthSettings.depthFunc = mat.depthSettings.depthOp.name

    def _importMiscSettings(self, miscSettings: MiscMatSettings, mat: mdl0.Material):
        if mat.lightSet > -1:
            miscSettings.useLightSet = True
            miscSettings.lightSet = mat.lightSet + 1
        if mat.fogSet > -1:
            miscSettings.useFogSet = True
            miscSettings.fogSet = mat.fogSet + 1

    def _loadMat(self, mat: mdl0.Material):
        # first, if mat merging is enabled, find out if there are any materials we can merge with
        # if there are, just store that & return w/o creating anything new
        if self.parentImporter.settings.mergeMats:
            for mdl, mdlImporter in self.parentImporter.models.items():
                if mdlImporter is not self:
                    for otherMat, blenderMatName in mdlImporter.mats.items():
                        if otherMat.isDuplicate(mat):
                            self.mats[mat] = blenderMatName
                            return
        # create material
        blenderMat = bpy.data.materials.new(mat.name)
        self.mats[mat] = blenderMat.name
        matSettings = blenderMat.brres
        # tev
        if mat.tevConfig is not None:
            matSettings.tevID = self.tevConfigs[mat.tevConfig]
        # textures
        matSettings.miscSettings.texTransformMode = TEX_TRANSFORM_MODES[mat.mtxGen]
        for tex in mat.textures:
            self._importTex(matSettings.textures.add(False), tex)
        # indirect configurations
        for ind, uiInd in zip(mat.indTextures, matSettings.indSettings.texConfigs):
            uiInd.mode = ind.mode.name
            uiInd.scaleU, uiInd.scaleV = (s.name for s in ind.coordScales)
            if ind.lightIdx >= 0:
                uiInd.lightSlot = ind.lightIdx + 1
        for srt in mat.indSRTs:
            t = matSettings.indSettings.transforms.add(False).transform
            t.scale, t.rotation, t.translation = srt.scale, np.deg2rad(srt.rot), srt.trans
        # lighting channels
        for lc in mat.lightChans:
            self._importLightChan(matSettings.lightChans.add(False), lc)
        # color registers
        matSettings.colorRegs.constant = mat.constColors
        matSettings.colorRegs.standard[1:] = mat.standColors
        # other settings
        self._importAlphaSettings(matSettings.alphaSettings, mat)
        self._importDepthSettings(matSettings.depthSettings, mat)
        self._importMiscSettings(matSettings.miscSettings, mat)
        blenderMat.update_tag()

    def _altMeshName(self, mesh: mdl0.Mesh):
        """Get an alternate name for a MDL0 mesh based on the name of its first position group.

        In retail files, position groups are always named with a unique name followed by two
        underscores and then the name of the material used with it. The "alternate name" returned
        by this function is that unique name for the mesh's position group. If the position group's
        name doesn't follow this format, then its full name is just returned.
        """
        group = mesh.vertGroups[mdl0.PsnGroup][0]
        try:
            return group.name[:group.name.index(mesh.mat.name) - 2]
        except ValueError:
            return group.name

    def _joinMeshes(self, model: mdl0.MDL0):
        """Return a list mapping multi-material mesh names to lists of the MDL0 meshes they use.

        If multi-material meshes are disabled, each entry is just a mesh's name pointing to a list
        containing only that mesh.
        """
        if not self.parentImporter.settings.multiMatMeshes:
            return [(mesh.vertGroups[mdl0.PsnGroup][0].name, [mesh]) for mesh in model.meshes]
        remainingMeshes = set(model.meshes)
        meshes: list[tuple[str, list[mdl0.Mesh]]] = []
        for mesh in model.meshes:
            if mesh in remainingMeshes:
                remainingMeshes.remove(mesh)
                meshBatch = [mesh]
                name = self._altMeshName(mesh)
                meshes.append((name, meshBatch))
                # search meshes that don't yet belong to any batches, and if compatible, add to
                # this mesh's batch
                for other in model.meshes:
                    isCompatible = (other in remainingMeshes
                                  and other.visJoint is mesh.visJoint
                                  and other.singleBind is mesh.singleBind
                                  and other.drawPrio == mesh.drawPrio
                                  and self._altMeshName(other) == name)
                    if isCompatible:
                        meshBatch.append(other)
                        remainingMeshes.remove(other)
        return meshes

    def _importAttrLayers(self, mesh: bpy.types.Mesh, groupData: dict[int, np.ndarray],
                          cmdData: np.ndarray, isUV: bool, names: list[str] = None):
        """Import attribute layers for a mesh from vertex group slots & command data.

        Additionally, the "isUV" parameter specifies whether to add UV maps or vertex colors.
        "names" can be used to set the layer names, or left None to make Blender generate
        the names automatically.
        """
        useAttrNames = names is not None and self.parentImporter.settings.useAttrNames
        names = names if names else list(range(len(groupData)))
        layerNames: dict[str, str] = {} # maps brres names to blender names
        meshAttrs = (mesh.brres.meshAttrs.uvs if isUV else mesh.brres.meshAttrs.clrs)
        for groupName, (i, data) in zip(names, groupData.items()):
            # create layer if it doesn't exist yet
            if groupName not in layerNames:
                layerName = groupName if useAttrNames else ""
                layer = None
                if isUV:
                    layer = mesh.uv_layers.new(name=layerName, do_init=False)
                    data[:, 1] = 1 - data[:, 1] # flip for blender conventions
                    layer.data.foreach_set("uv", data[cmdData[i]].flatten())
                else:
                    layer = mesh.attributes.new(layerName, 'FLOAT_COLOR', 'CORNER')
                    layer.data.foreach_set("color", data[cmdData[i]].flatten())
                layerNames[groupName] = layer.name
            # assign layer for attribute slot
            meshAttrs[i] = layerNames[groupName]

    def _loadMeshes(self, model: mdl0.MDL0):
        settings = self.parentImporter.settings
        importScale = settings.scale
        for meshName, meshes in self._joinMeshes(model):
            blendMesh = bpy.data.meshes.new(meshName)
            obj = bpy.data.objects.new(meshName, blendMesh)
            self.parentImporter.context.collection.objects.link(obj)
            meshRep = meshes[0] # representative used for props shared by all meshes (eg, draw prio)
            isMultiMesh = len(meshes) > 1
            blendMesh.brres.drawPrio = meshRep.drawPrio
            blendMesh.brres.enableDrawPrio = meshRep.drawPrio > 0
            hasSkinning = any(mesh.hasPsnMtcs() for mesh in meshes)
            obj.parent = self.rigObj
            # visibility bone
            visBone = self.bones[meshRep.visJoint]
            driver = obj.driver_add("hide_viewport").driver
            driver.type = 'SUM'
            var = driver.variables.new()
            var.type = 'SINGLE_PROP'
            target = var.targets[0]
            target.id = self.rigObj
            target.data_path = f"data.bones[\"{visBone}\"].hide"
            # single-bind bone parenting
            if meshRep.singleBind is not None and not hasSkinning:
                obj.parent_type = 'BONE'
                obj.parent_bone = self.bones[meshRep.singleBind.joints[0]]
                # apply parent matrix, position relative to head instead of tail, & correct for base
                parent: bpy.types.Bone = obj.parent.data.bones[obj.parent_bone]
                parentMtx = MTX_FROM_BRRES @ Matrix(meshRep.singleBind.mtx(self.model))
                parentMtx.translation /= importScale
                parentMtx = parent.matrix_local.inverted() @ parentMtx
                headAdjustMtx = Matrix.Translation((0, -parent.length, 0))
                obj.matrix_parent_inverse = headAdjustMtx @ parentMtx @ MTX_TO_BONE.to_4x4()
            # join data from meshes in batch
            # (batches might have just one mesh, or multiple in case of multi-material meshes)
            vgData: dict[type[mdl0.VertexAttrGroup], dict[int, np.ndarray]] = {
                t: {} for t in (mdl0.PsnGroup, mdl0.NrmGroup, mdl0.ClrGroup, mdl0.UVGroup)
            }
            cmds: list[gx.DrawPrimitives] = []
            vertMats = []
            faceMats = []
            cmdAttrs = ("psns", "nrms", "clrs", "uvs")
            groupStartIdcs: dict[mdl0.VertexAttrGroup, int] = {}
            for meshIdx, mesh in enumerate(meshes):
                # add commands & vertex group data
                meshCmds = list(mesh.cmds)
                cmds += meshCmds
                # go through each attribute type (psns, nrms, clrs, uvs)
                # then, within each one, add to the corresponding vertex group and add the commands,
                # with offsets to account for the vertex groups merging (e.g., index 5 into a
                # group on the second mesh becomes 15 if the first mesh had 10 entries)
                for slotData, slots, a in zip(vgData.values(), mesh.vertGroups.values(), cmdAttrs):
                    for s, group in slots.items():
                        # get data for this group & add it to existing data, padding if necessary
                        # (if group doesn't have all dimensions stored, e.g., rgb instead of rgba)
                        groupData = group.arr.copy()
                        if isinstance(group, mdl0.ClrGroup):
                            groupData[:, :3] **= 2.2
                        try:
                            curSlotData = slotData[s]
                            # add offsets to commands to compensate for expanded vertex groups
                            try:
                                cmdOffset = groupStartIdcs[group]
                            except KeyError:
                                cmdOffset = groupStartIdcs[group] = len(curSlotData)
                            for cmd in meshCmds:
                                getattr(cmd, a)[s] += cmdOffset
                            # update vertex data
                            slotData[s] = np.concatenate((curSlotData, groupData))
                        except KeyError:
                            # this is the first group used - no offset necessary
                            slotData[s] = groupData
                            groupStartIdcs[group] = 0
                # add material
                blendMat = bpy.data.materials[self.mats[mesh.mat]] if mesh.mat is not None else None
                blendMesh.materials.append(blendMat)
                vertMats += [meshIdx] * mesh.numVerts()
                faceMats += [meshIdx] * mesh.numFaces()
            if not cmds:
                return
            # construct blender mesh
            psnArr = vgData[mdl0.PsnGroup][0]
            verts = psnArr[np.concatenate([cmd.psns[0] for cmd in cmds])]
            faces = []
            vertIdx = 0
            for cmd in cmds:
                faces.append(cmd.faces()[:, ::-1] + vertIdx)
                vertIdx += len(cmd)
            faces = [f for c in faces for f in c]
            blendMesh.from_pydata(verts / importScale, edges=(), faces=faces)
            blendMesh.polygons.foreach_set("material_index", faceMats)
            loopVertIdcs = getLoopVertIdcs(blendMesh) # vertex index for each loop
            # normals (might change when applying skinning, so actually get applied later)
            hasNormals = settings.customNormals and len(vgData[mdl0.NrmGroup]) > 0
            vertNormals: np.ndarray = None
            if hasNormals:
                nrmArr = vgData[mdl0.NrmGroup][0]
                vertNormals = nrmArr[np.concatenate([cmd.nrms[0] for cmd in cmds])]
            # colors & uvs
            meshAttrs = blendMesh.brres.meshAttrs
            meshAttrs.clrs = [""] * len(meshAttrs.clrs) # clear defaults
            meshAttrs.uvs = [""] * len(meshAttrs.uvs)
            usedGroups = [(m.vertGroups[mdl0.ClrGroup], m.vertGroups[mdl0.UVGroup]) for m in meshes]
            useAttrNames = settings.useAttrNames
            attrNames: dict[type[mdl0.VertexAttrGroup], dict[int, str]] = {}
            for groupType in (mdl0.ClrGroup, mdl0.UVGroup):
                # if enabled, import attribute names as long as meshes don't have conflicting groups
                attrNames[groupType] = groupTypeAttrNames = {}
                for m in meshes:
                    for slot, group in m.vertGroups[groupType].items():
                        try:
                            if groupTypeAttrNames[slot] != group.name:
                                useAttrNames = False
                                break
                        except KeyError:
                            groupTypeAttrNames[slot] = group.name
                    if not useAttrNames:
                        break
            self._importAttrLayers(
                mesh=blendMesh,
                groupData=vgData[mdl0.ClrGroup],
                cmdData=np.concatenate([cmd.clrs for cmd in cmds], axis=1)[:, loopVertIdcs],
                isUV=False,
                names=attrNames[mdl0.ClrGroup].values() if useAttrNames else None
            )
            self._importAttrLayers(
                mesh=blendMesh,
                groupData=vgData[mdl0.UVGroup],
                cmdData=np.concatenate([cmd.uvs for cmd in cmds], axis=1)[:, loopVertIdcs],
                isUV=True,
                names=attrNames[mdl0.UVGroup].values() if useAttrNames else None
            )
            # skinning
            vertDfs = np.ndarray(0)
            if hasSkinning:
                rigModifier = obj.modifiers.new("Armature", 'ARMATURE')
                rigModifier.object = self.rigObj
                for boneName in self.bones.values():
                    obj.vertex_groups.new(name=boneName)
                vertIdcs = np.arange(len(verts))
                vertIdx = 0
                for dg in (dg for mesh in meshes for dg in mesh.drawGroups):
                    deformerAddrs = np.concatenate([c.psnMtcs[0] for c in dg.cmds], dtype=np.uint8) # pylint: disable=unexpected-keyword-arg
                    deformerIdcs = gx.LoadPsnMtx.addrToIdx(deformerAddrs * 4)
                    deformerHashes = np.array([hash(df) for df in dg.deformers], dtype=np.int64)
                    vertDfs = np.concatenate((vertDfs, deformerHashes[deformerIdcs]))
                    dgLen = len(deformerIdcs)
                    dgVertIdcs = vertIdcs[vertIdx : vertIdx + dgLen] # verts used by this draw group
                    for i, d in enumerate(dg.deformers):
                        # get indices used by this deformer & apply
                        # note: using np array directly doesn't work so we have to convert to list
                        dVertIdcs = dgVertIdcs[deformerIdcs == i].tolist()
                        for joint, weight in d.items():
                            obj.vertex_groups[self.bones[joint]].add(dVertIdcs, weight, 'REPLACE')
                        # transform vertices to apply rest position
                        # multi-weight deformers are stored already transformed, so they need
                        # special treatment due to the scaling system described in _loadJoints()
                        # (we turn the original transform into our no-scale version through pose())
                        restFix: np.ndarray
                        if len(d) == 1:
                            restFix = d.mtx(self.model)
                        else:
                            scaledPose = d.pose(self.model, self._scaledJointMtcs)
                            restFix = np.linalg.inv(scaledPose)
                        dVerts = verts[dgVertIdcs][deformerIdcs == i]
                        dVerts = np.pad(dVerts, ((0, 0), (0, 1)), constant_values=1) @ restFix.T
                        verts[vertIdx : vertIdx + dgLen][deformerIdcs == i] = dVerts[:, :3]
                        if hasNormals:
                            dNrms = vertNormals[dgVertIdcs][deformerIdcs == i]
                            dNrms = dNrms @ np.linalg.inv(restFix)[:3, :3]
                            vertNormals[vertIdx : vertIdx + dgLen][deformerIdcs == i] = dNrms
                    vertIdx += dgLen
                # re-apply positions because of the multi-weight adjustment
                blendMesh.vertices.foreach_set("co", verts.flatten() / importScale)
            # now that normals are finalized, store them in a temporary layer to apply them later
            # (this protects them from being modified during bmesh conversion & double removal)
            nrmLayerName = ""
            if hasNormals:
                # normalize normals (must be done because of skinning transformations)
                with np.errstate(divide="ignore", invalid="ignore"): # suppress 0 division warnings
                    vertNormals /= np.linalg.norm(vertNormals, axis=1).reshape(-1, 1)
                # store in temporary data layer
                nrmLayer = blendMesh.attributes.new("normals", 'FLOAT_VECTOR', 'CORNER')
                nrmLayer.data.foreach_set("vector", vertNormals[loopVertIdcs].flatten())
                nrmLayerName = nrmLayer.name
            # remove doubles
            # note: bmesh.ops.remove_doubles() exists, but isn't suitable for a few reasons,
            # the main one being that it merges vertices w/ the same positions but different
            # bone weights, which is no good (w4 worldmap palm trees hit this case)
            bm = bmesh.new()
            bm.from_mesh(blendMesh)
            bmVerts = bm.verts
            weldMap: dict[bmesh.types.BMVert, bmesh.types.BMVert] = {}
            bmVerts.ensure_lookup_table()
            # get arrays mapping vertices to their first occurrences
            unqIdcs, unqInv = np.unique(verts, return_index=True, return_inverse=True, axis=0)[1:3]
            weldSrc = np.arange(len(verts))
            weldDst = unqIdcs[unqInv]
            # remove mappings for verts that just point to themselves, verts w/ different deformers,
            # & verts w/ different materials (different materials are usually fine, but there are
            # cases where this behavior is desirable, so we just don't merge for the sake of those
            # times; one example is mario's hair, which is stored twice w/ two identical-looking but
            # different materials and gets a little weird when the meshes are merged)
            fltr = weldSrc != weldDst
            vertMats = np.array(vertMats)
            fltr = np.logical_and(fltr, vertMats[weldSrc] == vertMats[weldDst])
            if hasSkinning:
                fltr = np.logical_and(fltr, vertDfs[weldSrc] == vertDfs[weldDst])
            # also remove mappings for verts that are part of identical faces
            # these are faces that have the same corner positions, but other differences (e.g.,
            # directions), which means we shouldn't merge the vertices (e.g., ws_w1's flowers have
            # 2 faces for each face to face both ways; you can't have 2 faces w/ the exact same
            # corner vertices in blender, so we need to keep the vertices separate)
            # the face stuff is more complex than the other stuff, so we don't bother w/ numpy here
            # first, get the faces used for each vertex and positions used by each face
            vertFaces = [set() for _ in verts]
            faceVertSets = [{tuple(verts[v]) for v in f} for f in faces]
            for i, f in enumerate(faces):
                for v in f:
                    vertFaces[v].add(i)
            # now, apply face filtering, and finally construct the dict map & weld
            for src, dst in zip(weldSrc[fltr], weldDst[fltr]):
                identicalFacesFound = False
                for srcFace in vertFaces[src]:
                    for dstFace in vertFaces[dst]:
                        if faceVertSets[srcFace] == faceVertSets[dstFace]:
                            identicalFacesFound = True
                            break
                    if identicalFacesFound:
                        break
                if not identicalFacesFound:
                    weldMap[bmVerts[src]] = bmVerts[dst]
            bmesh.ops.weld_verts(bm, targetmap=weldMap)
            bm.to_mesh(blendMesh)
            bm.free()
            # now we can actually apply the normals!
            if hasNormals:
                if hasSkinning:
                    # without this enabled, normals get scuffed when skinned meshes are moved out of
                    # rest position. however, with it enabled, they get a little weird sometimes for
                    # non-skinned meshes. so, there are two options:
                    # --------
                    # 1) disable and get perfect results, unless skinned meshes are posed
                    # 2) enable and it fixes that problem, but there are some weird artifacts (maybe
                    # just for non-skinned meshes? maybe for all meshes? not sure)
                    # --------
                    # since i'm not sure if this even creates the artifacts for skinned meshes, it's
                    # possible this is just correct. in either case though, option 2 is better than
                    # option 1 for skinned meshes, so i enable for them and not otherwise
                    blendMesh.polygons.foreach_set("use_smooth", [True] * len(blendMesh.polygons))
                blendMesh.use_auto_smooth = True
                nrmData = blendMesh.attributes[nrmLayerName].data
                blendMesh.normals_split_custom_set(foreachGet(nrmData, "vector", 3))
                # note that we access the layer via its name again bc keeping the reference after
                # the previous line seems to lead to undefined behavior
                blendMesh.attributes.remove(blendMesh.attributes[nrmLayerName])
            else:
                # custom normals disabled - assume smooth (rather than default, which is flat)
                blendMesh.polygons.foreach_set("use_smooth", [True] * len(blendMesh.polygons))
            # finish up
            # blendMesh.validate(clean_customdata=False)
            blendMesh.transform(MTX_FROM_BRRES)

    def _loadJoints(self, model: mdl0.MDL0):
        settings = self.parentImporter.settings
        importScale = settings.scale
        importBoneLen = settings.boneLen
        mtxBoneToBRRES = self.parentImporter.mtxBoneToBRRES
        rigObj = self.rigObj
        rig: bpy.types.Armature = rigObj.data
        self.parentImporter.context.view_layer.objects.active = rigObj
        bpy.ops.object.mode_set(mode='EDIT')
        # blender has no concept of rest scale. because of this, if we apply bone scaling for the
        # rest pose, information will unavoidably be lost and certain animations/poses will become
        # impossible to produce. to avoid this, we just set every bone's scaling to the identity,
        # and apply the scales through the current armature pose (which is backed up through a rest
        # action) in case the user wants that. animations work perfectly using this method, since
        # we can just set absolute scales there.
        # because we do this whole thing, we have to store the joints' initial (scaled) matrices,
        # since these matrices are needed when decoding mesh information.
        self._scaledJointMtcs = {j: j.mtx(model) for j in model.rootJoint.deepChildren()}
        initialScales: list[np.ndarray] = []
        # make a bone for each joint
        for joint in model.rootJoint.deepChildren():
            bone = rig.edit_bones.new(joint.name)
            rig.edit_bones.active = bone
            bone.length = importBoneLen
            if joint.parent is not None:
                bone.parent = rig.edit_bones[self.bones[joint.parent]]
            self.bones[joint] = bone.name
            initialScales.append(joint.scale)
            # transform bone
            joint.setSRT(s=(1, 1, 1))
            jointMtx = joint.absMtx(model)
            boneMtx = Matrix(jointMtx)
            boneMtx.translation /= importScale
            rotMtx = Matrix(tf.Rotation.extractMtx(jointMtx))
            bone.transform(MTX_FROM_BONE @ rotMtx @ mtxBoneToBRRES @ MTX_TO_BONE)
            boneVec = bone.vector
            # we've applied rotation, now do translation
            # the "scale" here is a different thing from the rest scale i was talking about earlier
            # here, it just means when transforming the bone, don't worry about its envelope
            bone.transform(MTX_FROM_BRRES @ boneMtx, scale=False, roll=False)
            bone.tail = bone.head + boneVec # keep rotation fixed
            # this makes posed locations ignore rest rotation (makes anim import a little easier)
            bone.use_local_location = False
        # once all bones have been created, perform a little processing we couldn't do in edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        mtxBoneToBRRES = np.array(mtxBoneToBRRES.to_4x4())
        for joint, poseBone, initialScale in zip(self.bones, rigObj.pose.bones, initialScales):
            bone = poseBone.bone
            bone.hide = not joint.isVisible
            bone.brres.bbMode = joint.bbMode.name
            if joint.bbParent:
                bone.brres.bbParent = self.bones[joint.bbParent]
            poseBone.rotation_mode = 'XYZ'
            poseBone.scale = tf.Scaling.fromMtx(mtxBoneToBRRES @ tf.Scaling.mtx(initialScale))
            if joint.segScaleComp and joint.parent is not None:
                self._setSegScaleComp(joint)

    def _sscControl(self, joint: mdl0.Joint):
        """Return a segment scale compensate controller for a joint to be used by its children.

        If one doesn't exist yet, it's created.
        """
        # basically we create an empty with constraints to make it function identically to a bone
        # but without its local scale (all parent scale & shear still maintained)
        # then for children with ssc enabled, we use constraints to make them act like children of
        # the empty rather than this bone (though the bone is still the technical parent)
        if joint in self._sscControls:
            return bpy.data.objects[self._sscControls[joint]]
        rigObj = self.rigObj
        bone: bpy.types.Bone = rigObj.data.bones[self.bones[joint]]
        control: bpy.types.Object = bpy.data.objects.new(f"SSC Controller ({bone.name})", None)
        self.parentImporter.context.collection.objects.link(control)
        self._sscControls[joint] = control.name
        control.empty_display_type = 'ARROWS'
        control.empty_display_size = bone.length
        control.hide_set(True)
        # parent controller to bone, or another controller if ssc enabled on this parent itself
        # note that the controller is a sibling of the bone to which it corresponds
        # this is the only way to accomplish this - if we made the controller a child, it wouldn't
        # be possible to remove the parent's scale w/o altering rotation or shear (due to the limits
        # of blender's constraint system)
        if joint.parent is not None:
            if joint.segScaleComp:
                control.parent = self._sscControl(joint.parent)
            else:
                control.parent = rigObj
                control.parent_type = 'BONE'
                control.parent_bone = bone.parent.name
        else:
            control.parent = rigObj
        # inherit rest pose relative to parent
        rest = bone.matrix_local
        if joint.parent is not None:
            rest = bone.parent.matrix_local.inverted() @ rest
        control.matrix_parent_inverse = rest
        # inherit rotation
        # (have to use drivers instead of copy rotation constraint because that can mess w/ shear)
        for compIdx, comp in enumerate('XYZ'):
            driver = control.driver_add("rotation_euler", compIdx).driver
            driver.type = 'SUM'
            var = driver.variables.new()
            var.type = 'TRANSFORMS'
            target = var.targets[0]
            target.id = rigObj
            target.bone_target = bone.name
            target.transform_type = f'ROT_{comp}'
            target.rotation_mode = 'XYZ'
            target.transform_space = 'TRANSFORM_SPACE'
        # inherit translation
        # (completely unnecessary since translation is ignored by children, but it's nice to have
        # the controllers positioned with the bones they represent)
        transConstraint = control.constraints.new('COPY_LOCATION')
        transConstraint.target = rigObj
        transConstraint.subtarget = bone.name
        transConstraint.target_space = 'WORLD'
        transConstraint.owner_space = 'WORLD'
        return control

    def _setSegScaleComp(self, joint: mdl0.Joint):
        """Create constraints & drivers for a joint's bone to replicate segment scale compensate."""
        # this function uses the controller system outlined in _createSSCControl to apply ssc
        # creating the whole thing was very easy and definitely didn't take at least 20 hours :)
        parent = joint.parent
        if parent is None:
            return
        rigObj = self.rigObj
        bone = rigObj.pose.bones[self.bones[joint]]
        bone.bone.inherit_scale = 'NONE'
        if self.parentImporter.settings.sscMode == 'SIMPLE':
            # this import option mode just disables inherit scale rather than making the whole setup
            # it's not quite correct as grandparent scaling doesn't get applied, but it's simple
            return
        bone.bone.use_inherit_rotation = False
        # inherit from controller
        inheritConstraint = bone.constraints.new('CHILD_OF')
        inheritConstraint.name = "Segment Scale Compensate (Main)"
        inheritConstraint.target = self._sscControl(parent)
        inheritConstraint.inverse_matrix = bone.bone.parent.matrix_local.inverted()
        # that inheritance messes up translation, which we just want to be the original translation
        # we could fix this by disabling translation in the child of constraint we just added,
        # but that has side-effects to shear and rotation because of constraints' limits (relevant:
        # https://devtalk.blender.org/t/fundamental-flaws-in-blenders-constraint-philosophy/13921)
        # so instead we just set the translation using drivers
        # (which doesn't have any of those side effects, thankfully)
        fixTransConstraint = bone.constraints.new('LIMIT_LOCATION')
        fixTransConstraint.name = "Segment Scale Compensate (Location Fix)"
        fixTransConstraint.owner_space = 'LOCAL'
        for comp in "xyz":
            for bound in ("min", "max"):
                setattr(fixTransConstraint, f"use_{bound}_{comp}", True)
                driver = fixTransConstraint.driver_add(f"{bound}_{comp}").driver
                driver.type = 'SUM'
                var = driver.variables.new()
                var.type = 'TRANSFORMS'
                target = var.targets[0]
                target.id = rigObj
                target.bone_target = bone.name
                target.transform_type = f'LOC_{comp.capitalize()}'
                target.transform_space = 'TRANSFORM_SPACE'


# SOME NOTES ABOUT ANIMATION DOMAINS & ACTION IMPORING
# brres animations are generic at the model level, and can be applied to any model with the right
# data paths (material names, joint names, etc)
# blender animations (actions) are generic too, but at the id level - for instance, while brres
# material animations are collections of animations for a full model's materials, a blender material
# animation is an action applied to a single material
# because of discrepancies like this, animation import is difficult. the number of blender actions
# needed for each subfile type to get proper import goes something like this:
# 1 chr file -> 1 action (obj domain) per armature
# 1 clr file -> 1 action (mat domain) per material
# 1 pat file -> 1 action (mat domain) per material
# 1 srt file -> 1 action (mat domain) per material
# 1 vis file -> 1 action (obj domain)
# since you can only have one action active on an id at a time (or only one action per nla strip),
# the number of actions created for a given animation type must be the highest number for its domain
# so actually, 1 vis file -> 1 action (obj domain) per armature
# this means we make some actions with duplicate data. unfortunately, this is the best
# solution i've found to this problem so far, so it's what we've got for now.


class BRRESAnimImporter(Generic[ANIM_SUBFILE_T]):

    def __init__(self, parentImporter: "BRRESImporter", anim: ANIM_SUBFILE_T):
        self.parentImporter = parentImporter
        self.anim = anim

    def _loadAction(self, animSubfile: animation.AnimSubfile, subAnimName: str = None):
        """Get the Blender action for a BRRES animation (creating one if not yet created)."""
        actionName = f"{animSubfile.name} ({subAnimName})" if subAnimName else animSubfile.name
        try:
            return bpy.data.actions[self.parentImporter.actions[actionName]]
        except KeyError:
            action = bpy.data.actions.new(actionName)
            self.parentImporter.actions[actionName] = action.name
            action.use_fake_user = False
            action.use_frame_range = True
            action.frame_start = self.parentImporter.settings.frameStart
            # subtract 1 as brres "length" is number of frames (including both endpoints),
            # as opposed to length of frame span (which doesn't include one endpoint, and is what
            # we want here)
            action.frame_end = action.frame_start + animSubfile.length - 1
            action.use_cyclic = animSubfile.enableLoop
            return action

    def _genTrack(self, data: bpy.types.ID, action: bpy.types.Action, name: str):
        """If it doesn't already exist, generate an NLA track on an ID for an animation.

        Additionally, if it doesn't already exist, generate a rest action on the ID to back up
        anything modified by the provided action.
        """
        self._genRestAction(data, action)
        animData = data.animation_data
        # make action active temporarily to update its id_root (id type to which it can be applied)
        initialAction = animData.action
        animData.action = action
        animData.action = initialAction
        # generate track
        tracks: bpy.types.NlaTracks = animData.nla_tracks
        try:
            track = tracks[name]
        except KeyError:
            track = tracks.new()
            track.name = name
            track.mute = True
        if action.name not in track.strips.keys():
            strip = track.strips.new("", self.parentImporter.settings.frameStart, action)
            # crop strip to full action range
            strip.action_frame_start = action.frame_start
            strip.action_frame_end = action.frame_end

    def _genRestAction(self, data: bpy.types.ID, animAction: bpy.types.Action):
        """Generate a rest action for some ID, or update an existing one w/ more values.

        Another action is required to determine which properties of the ID are backed up (anything
        modified by that action).
        """
        frameStart = self.parentImporter.settings.frameStart
        rest: bpy.types.Action = None
        if data.animation_data and data.animation_data.action:
            rest = data.animation_data.action
        else:
            rest = bpy.data.actions.new(f"BRRES Initial State ({data.name})")
            rest.use_fake_user = True
            animData = data.animation_data_create()
            animData.action = rest
            # track = animData.nla_tracks.new()
            # track.name = "BRRES Initial State"
            # track.strips.new("", self.parentImporter.settings.frameStart, rest)
        existingRestFcPaths = {(fc.data_path, fc.array_index) for fc in rest.fcurves}
        # go through paths modified by anim action to determine what to back up
        for fc in animAction.fcurves:
            path = (fc.data_path, fc.array_index)
            if path in existingRestFcPaths:
                continue # don't back up stuff that's already been backed up
            dataPath, dataIdx = path
            # get property value from id & set up rest entry
            try:
                restVal = data.path_resolve(dataPath)
            except ValueError:
                continue # id doesn't have this property
            try:
                # data might have an index if it points to a component of a vector
                restVal = restVal[dataIdx]
            except TypeError:
                pass
            rest.fcurves.new(dataPath, index=dataIdx).keyframe_points.insert(frameStart, restVal)

    def _genFCurve(self, kfs: np.ndarray, action: bpy.types.Action, path: str, idx = 0):
        """Generate an fcurve for some action and datapath/index based on BRRES keyframe data."""
        # make keyframes
        numKfs = len(kfs)
        fc = action.fcurves.new(path, index=idx)
        fc.color_mode = 'AUTO_RGB'
        fc.keyframe_points.add(numKfs)
        coords = kfs[:, :2].copy()
        coords[:, 0] += self.parentImporter.settings.frameStart
        fc.keyframe_points.foreach_set("co", coords.flatten())
        # convert hermite to bezier for handles
        # https://math.stackexchange.com/questions/4128882/nonparametric-hermite-cubic-to-bezier-curve
        # note: leftmost left handle and rightmost right handle have undefined lengths
        # so we just make them reflect their corresponding right/left handles
        # (i.e., the first left handle reflects the first right handle, and
        # the last right handle reflects the last left handle)
        tans = kfs[:, 2]
        kfDists = coords[1:, 0] - coords[:-1, 0]
        kfDists /= 3 # dx is divided by constant 3 for conversion calculations
        # left handles
        handlesL = coords.copy()
        handlesL[1:, 0] -= kfDists
        handlesL[1:, 1] -= kfDists * tans[1:]
        # right handles
        handlesR = coords.copy()
        handlesR[:-1, 0] += kfDists
        handlesR[:-1, 1] += kfDists * tans[:-1]
        # reflect edge handles
        handlesL[0] = -handlesR[0] + 2 * coords[0]
        handlesR[-1] = -handlesL[-1] + 2 * coords[-1]
        # finally, actually set handles & verify fcurve
        for kfp in fc.keyframe_points: # for some reason foreach_set() does nothing for this
            kfp.handle_left_type = kfp.handle_right_type = 'FREE'
        fc.keyframe_points.foreach_set("handle_left", handlesL.flatten())
        fc.keyframe_points.foreach_set("handle_right", handlesR.flatten())
        fc.update()


class BRRESChrImporter(BRRESAnimImporter[chr0.CHR0]):

    def __init__(self, parentImporter: "BRRESImporter", anim: chr0.CHR0):
        super().__init__(parentImporter, anim)
        settings = parentImporter.settings
        frameStart = settings.frameStart
        importScale = settings.scale
        boneLen = settings.boneLen
        forExist = settings.animsForExisting
        mtxBoneToBRRES = np.array(parentImporter.mtxBoneToBRRES)
        mtxBoneFromBRRES = np.array(parentImporter.mtxBoneFromBRRES)
        jointAnims = {a.jointName: a for a in anim.jointAnims}
        objs = bpy.data.objects if forExist else (o.rigObj for o in parentImporter.models.values())
        for rigObj in objs:
            if rigObj.pose is None: # not an armature
                continue
            action = self._loadAction(anim, rigObj.name)
            for poseBone in rigObj.pose.bones:
                try:
                    jAnim = jointAnims[poseBone.name]
                except KeyError:
                    continue
                # now, fill out action data for this bone
                vecProps = ("scale", "rotation_euler", "location")
                scalePath, rotPath, transPath = [poseBone.path_from_id(p) for p in vecProps]
                # scale
                if jAnim.scale:
                    fcInfo: dict[int, np.ndarray] = {}
                    for cIdx, cAnim in enumerate(jAnim.scale):
                        kfs = cAnim.keyframes.copy()
                        cIdx = transformAxis(cIdx, mtxBoneFromBRRES)[0]
                        fcInfo[cIdx] = kfs
                    # sort to add fcurves in the correct order, since it may get
                    # messed up by axis transformations
                    # (this order is purely a visual thing within blender)
                    for cIdx, kfs in sorted(fcInfo.items(), key=lambda item: item[0]):
                        self._genFCurve(kfs, action, scalePath, cIdx)
                # rotation
                # rest pose compensation is done by interpolating rotation for each frame &
                # adjusting each one based on rest rotation
                # (this interpolation method is required because a rotational difference may require
                # changes in several components, which means that if you maintain the original kfs,
                # you can't always interpolate between adjusted rotations w/o data loss)
                if jAnim.rot:
                    usedFrameIdcs = np.concatenate([a.keyframes[:, 0] for a in jAnim.rot])
                    fMin = np.floor(usedFrameIdcs.min()).astype(int)
                    fMax = np.ceil(usedFrameIdcs.max()).astype(int)
                    fRange = np.arange(fMin, fMax + 1)
                    numFrames = fMax - fMin + 1
                    frames = [a.interpolate(fRange) for a in jAnim.rot]
                    frames = np.array(frames, dtype=float).T
                    mtcs = tf.Rotation.mtx(frames)[..., :-1, :-1]
                    invRotMtx = np.linalg.inv(poseBone.bone.matrix)
                    adjustMtx = mtxBoneFromBRRES if poseBone.parent else MTX_FROM_BONE
                    mtcs = invRotMtx @ adjustMtx @ mtcs @ mtxBoneToBRRES
                    frames = np.deg2rad(tf.decompose3DRotation(mtcs))
                    for i, compVals in enumerate(frames.swapaxes(0, 1)):
                        coords = np.stack((fRange + frameStart, compVals), axis=-1)
                        fc = action.fcurves.new(rotPath, index=i)
                        fc.color_mode = 'AUTO_RGB'
                        fc.keyframe_points.add(numFrames)
                        fc.keyframe_points.foreach_set("co", coords.flatten())
                        linear = enumVal(bpy.types.Keyframe, "interpolation", 'LINEAR')
                        fc.keyframe_points.foreach_set("interpolation", (linear, ) * numFrames)
                        fc.update()
                # translation
                # compensation is done by subtracting rest translation, applying import scale,
                # and accounting for bone axis conversions when necessary
                if jAnim.trans:
                    # if bone has a parent, translation must be converted based on bone axes
                    # if bone has no parent, translation must be be converted from bone
                    # coordinates to standard blender coordinates
                    convertMtx = mtxBoneFromBRRES if poseBone.parent else np.array(MTX_FROM_BONE)
                    restTrans = np.array(poseBone.bone.head * importScale)
                    if poseBone.parent: # head is relative to parent tail; adjust for that
                        restTrans[1] += importScale * boneLen
                    restTrans = restTrans @ convertMtx
                    fcInfo: dict[int, np.ndarray] = {}
                    for cIdx, (cAnim, rest) in enumerate(zip(jAnim.trans, restTrans)):
                        kfs = cAnim.keyframes.copy()
                        kfs[:, 1] -= rest
                        cIdx, kfScalar = transformAxis(cIdx, convertMtx)
                        kfs[:, 1:] *= kfScalar / importScale
                        fcInfo[cIdx] = kfs
                    # sort to add fcurves in the correct order, since it may get
                    # messed up by axis transformations
                    # (this order is purely a visual thing within blender)
                    for cIdx, kfs in sorted(fcInfo.items(), key=lambda item: item[0]):
                        self._genFCurve(kfs, action, transPath, cIdx)
            if action.fcurves:
                # if anything was generated, this model supports this animation, so create track
                self._genTrack(rigObj, action, anim.name)
            else:
                # otherwise, delete the action we created
                bpy.data.actions.remove(action)


class BRRESClrImporter(BRRESAnimImporter[clr0.CLR0]):

    def __init__(self, parentImporter: "BRRESImporter", anim: clr0.CLR0):
        super().__init__(parentImporter, anim)
        frameStart = parentImporter.settings.frameStart
        regPaths = (
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
            "brres.colorRegs.constant4",
        )
        for matAnim in anim.matAnims:
            # create action
            action = self._loadAction(anim, matAnim.matName)
            for regPath, regAnim in zip(regPaths, matAnim.allRegs):
                if regAnim is None:
                    continue
                frameIdcs = np.arange(len(regAnim.colors)) + frameStart
                for i, (mask, chan) in enumerate(zip(regAnim.mask, regAnim.normalized.T)):
                    if not mask:
                        fc = action.fcurves.new(regPath, index=i)
                        fc.color_mode = 'AUTO_RGB'
                        fc.keyframe_points.add(len(chan))
                        coords = np.stack((frameIdcs, chan), axis=-1)
                        fc.keyframe_points.foreach_set("co", coords.flatten())
                        linear = enumVal(bpy.types.Keyframe, "interpolation", 'LINEAR')
                        fc.keyframe_points.foreach_set("interpolation", (linear, ) * len(chan))
                        fc.update()
            # create nla tracks for materials referenced by this animation
            for mdl in parentImporter.models.values():
                try:
                    bMatName = next(n for m, n in mdl.mats.items() if m.name == matAnim.matName)
                    self._genTrack(bpy.data.materials[bMatName], action, anim.name)
                except StopIteration: # model doesn't have this material
                    pass
            if parentImporter.settings.animsForExisting:
                try:
                    self._genTrack(bpy.data.materials[matAnim.matName], action, anim.name)
                except KeyError:
                    pass


class BRRESPatImporter(BRRESAnimImporter[pat0.PAT0]):

    def __init__(self, parentImporter: "BRRESImporter", anim: pat0.PAT0):
        super().__init__(parentImporter, anim)
        self.matActions: dict[str, str] = {}
        """Maps material names to action names for this animation's material anims."""
        frameStart = self.parentImporter.settings.frameStart
        allTexImgSlots = parentImporter.texImgSlots
        # create map from blender image names to brres sources (tex/plt name tuple)
        imgInfo = {img: info for info, img in parentImporter.images.items()}
        # loop through material anims and fill out animation data
        for matAnim in anim.matAnims:
            # create action
            matName = matAnim.matName
            action = self._loadAction(anim, matName)
            self.matActions[matName] = action.name
            if matName not in allTexImgSlots:
                allTexImgSlots[matName] = {}
            matTexImgSlots = allTexImgSlots[matName]
            for texIdx, texAnim in matAnim.texAnims.items():
                if texIdx not in matTexImgSlots:
                    matTexImgSlots[texIdx] = []
                texImgSlots = matTexImgSlots[texIdx]
                texImgs = {imgInfo[s]: i for i, s in enumerate(texImgSlots)}
                kfs = texAnim.keyframes
                # process keyframes for texture's image slots, adding new ones if required
                # this little thing is not efficient but pat0 animations are usually very
                # small so it's fine for now, i got other bigger fish to fry
                texNames = texAnim.texNames
                pltNames = texAnim.pltNames
                numTex = len(texNames)
                numPlt = len(pltNames)
                processedKfs = []
                for f, (t, p) in zip(kfs[:, :1], kfs[:, 1:].astype(np.uint16)):
                    fTexName = texNames[t] if t < numTex else None
                    fPltName = pltNames[p] if p < numPlt else None
                    fImgInfo = (fTexName, fPltName)
                    if fImgInfo not in texImgs:
                        frameImgName = parentImporter.images.get(fImgInfo)
                        texImgs[fImgInfo] = len(texImgs)
                        texImgSlots.append(frameImgName)
                    processedKfs.append((f + frameStart, texImgs[fImgInfo] + 1))
                fc = action.fcurves.new(f"brres.textures.coll_[{texIdx}].activeImgSlot")
                fc.keyframe_points.add(len(kfs))
                fc.keyframe_points.foreach_set("co", [v for kf in processedKfs for v in kf])
                constant = enumVal(bpy.types.Keyframe, "interpolation", 'CONSTANT')
                fc.keyframe_points.foreach_set("interpolation", (constant, ) * len(kfs))
                fc.update()
            # create nla tracks for materials referenced by this animation
            for mdl in self.parentImporter.models.values():
                try:
                    bMatName = next(n for m, n in mdl.mats.items() if m.name == matAnim.matName)
                    self._genTrack(bpy.data.materials[bMatName], action, anim.name)
                except StopIteration: # model doesn't have this material
                    pass
            if parentImporter.settings.animsForExisting:
                try:
                    self._genTrack(bpy.data.materials[matAnim.matName], action, anim.name)
                except KeyError:
                    pass


class BRRESSrtImporter(BRRESAnimImporter[srt0.SRT0]):

    def __init__(self, parentImporter: "BRRESImporter", anim: srt0.SRT0):
        super().__init__(parentImporter, anim)
        for matAnim in anim.matAnims:
            # create action
            action = self._loadAction(anim, matAnim.matName)
            texAnims = (matAnim.texAnims, matAnim.indAnims)
            basePaths = ("brres.textures", "brres.indSettings.transforms")
            for texAnimDict, basePath in zip(texAnims, basePaths):
                for texIdx, texAnim in texAnimDict.items():
                    texPath = f"{basePath}.coll_[{texIdx}].transform"
                    vecPaths = (f"{texPath}.scale", f"{texPath}.rotation", f"{texPath}.translation")
                    vecAnims = (texAnim.scale, texAnim.rot, texAnim.trans)
                    for vecPath, vecAnim in zip(vecPaths, vecAnims):
                        for compIdx, compAnim in enumerate(vecAnim):
                            frames = compAnim.keyframes
                            if vecPath.endswith("rotation"):
                                np.deg2rad(frames[:, 1:], out=frames[:, 1:])
                            self._genFCurve(frames, action, vecPath, compIdx)
            # create nla tracks for materials referenced by this animation
            for mdl in self.parentImporter.models.values():
                try:
                    bMatName = next(n for m, n in mdl.mats.items() if m.name == matAnim.matName)
                    self._genTrack(bpy.data.materials[bMatName], action, anim.name)
                except StopIteration: # model doesn't have this material
                    pass
            if parentImporter.settings.animsForExisting:
                try:
                    self._genTrack(bpy.data.materials[matAnim.matName], action, anim.name)
                except KeyError:
                    pass


class BRRESVisImporter(BRRESAnimImporter[vis0.VIS0]):

    def __init__(self, parentImporter: "BRRESImporter", anim: vis0.VIS0):
        super().__init__(parentImporter, anim)
        forExist = parentImporter.settings.animsForExisting
        mdls = parentImporter.models.values()
        arms = bpy.data.armatures if forExist else {mdl.rigObj.data for mdl in mdls}
        action = self._loadAction(anim)
        animatedBones = {jointAnim.jointName for jointAnim in anim.jointAnims}
        for jointAnim in anim.jointAnims:
            frames = np.logical_not(jointAnim.frames)
            frameIdcs = np.arange(len(frames)) + parentImporter.settings.frameStart
            dataPath = f"bones[\"{jointAnim.jointName}\"].hide"
            fc = action.fcurves.new(dataPath)
            coords = np.stack((frameIdcs, frames), axis=-1)
            mask = np.insert(coords[:-1, 1] != coords[1:, 1], 0, True) # remove duplicate frames
            coords = coords[mask]
            numKfs = len(coords)
            fc.keyframe_points.add(numKfs)
            fc.keyframe_points.foreach_set("co", coords.flatten())
            constant = enumVal(bpy.types.Keyframe, "interpolation", 'CONSTANT')
            fc.keyframe_points.foreach_set("interpolation", (constant, ) * numKfs)
            fc.update()
        for arm in arms:
            if any(bone.name in animatedBones for bone in arm.bones):
                # if anything was generated, this model supports this animation, so create track
                self._genTrack(arm, action, anim.name)


class BRRESImporter():

    def __init__(self, context: bpy.types.Context, res: brres.BRRES, settings: "ImportBRRES"):
        self.context = context
        self.res = res
        self.settings = settings
        # set up bone axis conversion matrices
        self.mtxBoneToBRRES: Matrix = axis_conversion(
            from_forward='X',
            from_up='Y',
            to_forward=settings.secondaryBoneAxis,
            to_up=settings.primaryBoneAxis
        )
        self.mtxBoneFromBRRES: Matrix = self.mtxBoneToBRRES.inverted()
        # load images
        self.images: dict[tuple[str, str], str] = {}
        """Maps a TEX0 and optionally PLT0 name to a Blender image name."""
        for tex in res.folder(tex0.TEX0):
            if tex.isPaletteIndices:
                # for palette images, just import a different image for each usable palette
                # (maybe blender has a better analog to this, but for now this is fine)
                for plt in res.folder(plt0.PLT0):
                    if plt.isCompatible(tex):
                        self._loadImg(tex, plt)
            else:
                self._loadImg(tex, None)
        # load models
        self.models: dict[mdl0.MDL0, BRRESMdlImporter] = {}
        for model in res.folder(mdl0.MDL0):
            self._loadModel(model)
        # load animations in reverse-alphabetical order
        # (order reversed because nla tracks are added from bottom to top; by reversing, we get
        # alphabetical order from top to bottom)
        self.anims: dict[animation.AnimSubfile, BRRESAnimImporter] = {}
        self.actions: dict[str, str] = {}
        self.texImgSlots: dict[str, dict[int, list[str]]] = {}
        animFilter = settings.animFilter
        if settings.doAnim:
            animSubfileTypes = (chr0.CHR0, clr0.CLR0, pat0.PAT0, srt0.SRT0, vis0.VIS0)
            anims = [f for t in animSubfileTypes for f in res.folder(t)]
            anims.sort(key=lambda f: f.name, reverse=True)
            for anim in anims:
                if animFilter in anim.name:
                    self._loadAnim(anim)
            self._processPatAnims()

    def _processPatAnims(self):
        """Generate & sort the PAT0 texture image slots for this BRRES, after PAT0 loading."""
        # pat0 loading creates actions, but doesn't create the image slots, and also doesn't do any
        # sorting, which is nice to have since the slots can import in a messy order by default
        # so, this function handles all that after the pat0 animations are loaded
        # it's not very efficient but pat0 anims are typically very small so it's fine for now
        # this first bit gets the pat0 actions for each material together
        patAnims = {a for a in self.anims.values() if isinstance(a, BRRESPatImporter)}
        matActions: dict[str, set[str]] = {}
        for anim in patAnims:
            for matName, actionName in anim.matActions.items():
                try:
                    matActions[matName].add(actionName)
                except KeyError:
                    matActions[matName] = {actionName}
        # now, process actions & material texture image slots corresponding to each brres material
        for name, matTexImgSlots in self.texImgSlots.items():
            actions = [bpy.data.actions[a] for a in matActions[name]]
            # first, sort & determine new order
            # images are sorted alphabetically, unless they end with a period followed by numbers
            # in that case, they're sorted alphabetically for everything up to the dot, then by
            # the numbers if it comes down to that (e.g., img.1, img.2, and img.10 are sorted in
            # numerical order, rather than alphabetically, which would be img.1, img.10, img.2)
            slotSortMap: dict[int, dict[int, int]] = {}
            for texIdx, texImgSlots in matTexImgSlots.items():
                slotIdcs = {s: i for i, s in enumerate(texImgSlots)}
                nameSortKeys = {}
                for imgName in texImgSlots:
                    if imgName is None:
                        nameSortKeys[imgName] = ("", -1)
                    else:
                        dotIdx = imgName.rfind(".")
                        postDot = imgName[dotIdx + 1:]
                        if postDot.isdigit():
                            nameSortKeys[imgName] = (imgName[:dotIdx], int(postDot))
                        else:
                            nameSortKeys[imgName] = (imgName, -1)
                sortedNames = sorted(texImgSlots, key=lambda n, k=nameSortKeys: k[n])
                texImgMap = {slotIdcs[n]: i for i, n in enumerate(sortedNames)}
                slotSortMap[texIdx] = texImgMap
                texPath = f"brres.textures.coll_[{texIdx}].activeImgSlot"
                # fix order in fcurves using map from old indices to new
                for action in actions:
                    for fc in action.fcurves:
                        if fc.data_path == texPath:
                            for kf in fc.keyframe_points:
                                frameIdx, texImgSlot = kf.co
                                kf.co = (frameIdx, texImgMap[texImgSlot - 1] + 1)
            # now, add the actual texture images within the materials
            # (and adjust the rest action if the original is moved)
            names = {n for m in self.models.values() for mt, n in m.mats.items() if mt.name == name}
            for blendName in names: # for each blender material corresponding to this brres mat name
                mat = bpy.data.materials[blendName]
                textures = mat.brres.textures
                restAction = mat.animation_data.action
                restFCurves = {fc.data_path: fc for fc in restAction.fcurves}
                for texIdx, sortedTexImgs in slotSortMap.items():
                    texImgSlots = matTexImgSlots[texIdx]
                    tex = textures[texIdx]
                    texImgs = tex.imgs
                    for imgIdx in sortedTexImgs:
                        texImg = texImgs.add(False)
                        imgName = texImgSlots[imgIdx]
                        if imgName is not None:
                            texImg.img = bpy.data.images[texImgSlots[imgIdx]]
                    # if the original image (from the mdl0, not pat0) has been re-added,
                    # make the new version the active one and remove the original
                    # if it hasn't, move it to the end so that the fcurves will work
                    # (but keep it active)
                    # also, adjust for this removal or movement in the rest action
                    originalImg = texImgs[0].img
                    restFc = restFCurves[f"brres.textures.coll_[{texIdx}].activeImgSlot"]
                    if originalImg and originalImg.name in texImgSlots:
                        newIdx = sortedTexImgs[texImgSlots.index(originalImg.name)]
                        texImgs.remove(texImgs.activeIdx)
                        texImgs.activeIdx = newIdx
                        restFc.keyframe_points[0].co[1] = newIdx + 1
                    else:
                        newIdx = len(texImgs) - 1
                        texImgs.move(0, newIdx)
                        restFc.keyframe_points[0].co[1] = newIdx + 1

    def _loadModel(self, model: mdl0.MDL0):
        self.models[model] = BRRESMdlImporter(self, model)

    def _loadImg(self, img: tex0.TEX0, plt: plt0.PLT0 = None):
        if self.settings.existingImages and img.name in bpy.data.images:
            self.images[(img.name, plt.name if plt else None)] = img.name
            return bpy.data.images[img.name]
        fmts = {
            tex0.I4: 'I4',
            tex0.I8: 'I8',
            tex0.IA4: 'IA4',
            tex0.IA8: 'IA8',
            tex0.RGB565: 'RGB565',
            tex0.RGB5A3: 'RGB5A3',
            tex0.RGBA8: 'RGBA8',
            tex0.CMPR: 'CMPR'
        }
        blenderImage = bpy.data.images.new(img.name, *img.dims, alpha=True)
        self.images[(img.name, plt.name if plt else None)] = blenderImage.name
        images = img.images
        if img.isPaletteIndices:
            blenderImage.brres.fmt = fmts[plt.fmt]
            try:
                images = [plt.colors[data] for data in images]
            except AttributeError as e:
                raise ValueError("To import a palette image, a palette must be provided") from e
        else:
            blenderImage.brres.fmt = fmts[img.fmt]
        blenderImage.pixels[:] = images[0][::-1].flatten()
        for i, mipmap in enumerate(images[1:]):
            mmSlot = blenderImage.brres.mipmaps.add(False)
            mmName = f"{img.name} (Mipmap {i + 1})"
            mmSlot.img = mmImg = bpy.data.images.new(mmName, *img.mipmapDims(i + 1), alpha=True)
            mmImg.brres.fmt = blenderImage.brres.fmt
            mmImg.pixels[:] = mipmap[::-1].flatten()
            mmImg.pack()
        blenderImage.pack()
        return blenderImage

    def _loadAnim(self, anim: ANIM_SUBFILE_T):
        self.anims[anim] = {
            chr0.CHR0: BRRESChrImporter,
            clr0.CLR0: BRRESClrImporter,
            pat0.PAT0: BRRESPatImporter,
            srt0.SRT0: BRRESSrtImporter,
            vis0.VIS0: BRRESVisImporter
        }[type(anim)](self, anim)


def drawOp(self, context: bpy.types.Context):
    self.layout.operator(ImportBRRES.bl_idname, text="Binary Revolution Resource (.brres)")


class ImportBRRES(bpy.types.Operator, ImportHelper):
    """Read a BRRES file"""

    bl_idname = "import_scene.brres"
    bl_label = "Import BRRES"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".brres"
    filter_glob: bpy.props.StringProperty(
        default="*.brres",
        options={'HIDDEN'},
    )

    mergeMats: bpy.props.BoolProperty(
        name="Merge Duplicate Materials",
        description="Import identical materials with the same name as one",
        default=True
    )

    multiMatMeshes: bpy.props.BoolProperty(
        name="Multi-Material Meshes",
        description="Join parts of meshes that are detected to be split into multiple materials based on naming conventions", # pylint: disable=line-too-long
        default=True
    )

    useAttrNames: bpy.props.BoolProperty(
        name="Attribute Names",
        description="Import the names of color & UV attributes, if possible", # pylint: disable=line-too-long
        default=False
    )

    customNormals: bpy.props.BoolProperty(
        name="Custom Normals",
        description="Use imported normals (otherwise, Blender will calculate them)",
        default=True
    )

    existingImages: bpy.props.BoolProperty(
        name="Use Existing Images",
        description="When an image being imported has the same name as one that already exists, use that one instead", # pylint: disable=line-too-long
        default=False
    )

    sscMode: bpy.props.EnumProperty(
        name="Scale Compensate Mode",
        description="Method for replicating the Segment Scale Compensate feature of BRRES bones.",
        items=(
            ('ACCURATE', "Accurate", "Replicate SSC through drivers & constraints that match BRRES behavior with total accuracy"), # pylint: disable=line-too-long
            ('SIMPLE', "Simple", "Replicate SSC by disabling scale inheritance, which doesn't match BRRES behavior in all cases but is often sufficient & makes editing easier"), # pylint: disable=line-too-long
        ),
        default='ACCURATE'
    )

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Import is scaled down by this factor. Recommended values are 16 for most models and 30 for worldmaps", # pylint: disable=line-too-long
        default=16
    )

    boneLen: bpy.props.FloatProperty(
        name="Bone Length",
        default=.1
    )

    primaryBoneAxis: bpy.props.EnumProperty(
        name="Primary Bone Axis",
        description="Destination primary bone axis (Blender uses Y)",
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
        description="Destination secondary bone axis (Blender uses X)",
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

    doAnim: bpy.props.BoolProperty(
        name="Import Animations",
        default=True
    )

    animsForExisting: bpy.props.BoolProperty(
        name="For Existing Assets",
        description="Apply animations to existing scene data (useful for NSMBW's player animations)", # pylint: disable=line-too-long
        default=False
    )

    animFilter: bpy.props.StringProperty(
        name="Filter",
        description="Only animations with names containing this string are imported (leave blank to import all animations)" # pylint: disable=line-too-long
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
        # self.report({'INFO'}, "Importing BRRES...")
        with open(self.filepath, "rb") as f:
            fileData = f.read()
            if not fileData:
                raise ValueError("File is empty")
            BRRESImporter(context, brres.BRRES.unpack(fileData), self)
        self.report({'INFO'}, f"Imported \"{os.path.basename(self.filepath)}\"")
        context.window.cursor_set('DEFAULT')
        restoreView(restoreShading)
        profiler.disable()
        with open(LOG_PATH, "w", encoding="utf-8") as logFile:
            Stats(profiler, stream=logFile).sort_stats(SortKey.CUMULATIVE).print_stats()
        return {'FINISHED'}

    def draw(self, context: bpy.types.Context):
        pass # leave menu drawing up to the import panels


class ImportPanel(bpy.types.Panel):
    bl_space_type = 'FILE_BROWSER'
    bl_region_type = 'TOOL_PROPS'
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        return context.space_data.active_operator.bl_idname == "IMPORT_SCENE_OT_brres"


class GeneralPanel(ImportPanel):
    bl_idname = "BRRES_PT_import_general"
    bl_label = "General"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ImportBRRES = context.space_data.active_operator
        layout.prop(settings, "mergeMats")
        layout.prop(settings, "multiMatMeshes")
        layout.prop(settings, "useAttrNames")
        layout.prop(settings, "customNormals")
        layout.prop(settings, "existingImages")
        layout.prop(settings, "scale")


class ArmPanel(ImportPanel):
    bl_idname = "BRRES_PT_import_arm"
    bl_label = "Armatures"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ImportBRRES = context.space_data.active_operator
        layout.prop(settings, "boneLen")
        layout.prop(settings, "sscMode", expand=True)
        layout.prop(settings, "primaryBoneAxis", expand=True)
        layout.prop(settings, "secondaryBoneAxis", expand=True)


class AnimPanel(ImportPanel):
    bl_idname = "BRRES_PT_import_anim"
    bl_label = "Animations"

    def draw_header(self, context: bpy.types.Context):
        settings: ImportBRRES = context.space_data.active_operator
        self.layout.prop(settings, "doAnim", text="")

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings: ImportBRRES = context.space_data.active_operator
        layout.enabled = settings.doAnim
        layout.prop(settings, "animsForExisting")
        layout.prop(settings, "animFilter")
        layout.prop(settings, "frameStart")
