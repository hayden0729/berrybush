# 3rd party imports
import bpy
import numpy as np
# internal imports
from .common import usedMatSlots
from .limiter import ObjectLimiter, ObjectLimiterFactory
from ..wii import gx


def verifyBRRES(op: "VerifyBRRES", context: bpy.types.Context, limiter: ObjectLimiter):
    """Verify the scene's BRRES settings, reporting warnings.

    The number of warnings reported and number of warnings suppressed are returned."""
    numProblems = 0
    numSuppressed = 0
    usedMats = set()
    images: set[bpy.types.Image] = set()
    # invalid attribute references
    for rigObj in bpy.data.objects:
        if rigObj.type == 'ARMATURE' and limiter.includes(rigObj):
            for obj in bpy.data.objects:
                if limiter.includes(obj) and obj.parent is rigObj:
                    try:
                        mesh = obj.to_mesh()
                    except RuntimeError:
                        continue
                    for attr in (*mesh.brres.meshAttrs.clrs, *mesh.brres.meshAttrs.uvs):
                        if attr == "" or attr in mesh.uv_layers or attr in mesh.attributes:
                            continue
                        numProblems += 1
                        e = (f"Mesh '{mesh.name}' references an attribute '{attr}' "
                             f"in its BRRES settings, but has no such attribute")
                        op.report({'INFO'}, e)
                    # grab used materials for use in material verification
                    objUsedMats = {matSlot.material for matSlot in usedMatSlots(obj, mesh)}
                    if None in objUsedMats:
                        numProblems += 1
                        e = (f"Mesh '{mesh.name}' has geometry without an assigned material "
                              "(this will not be exported)")
                        op.report({'INFO'}, e)
                    usedMats |= objUsedMats
    # material problems
    usedTevs = set()
    usedMats.discard(None)
    for mat in usedMats:
        # textures w/o images & images w/ dimensions that aren't powers of 2
        for tex in mat.brres.textures:
            for slot, texImg in enumerate(tex.imgs, 1):
                img = texImg.img
                if img is None:
                    numProblems += 1
                    e = (f"Animation slot {slot} for texture '{tex.name}' of material '{mat.name}' "
                         f"is empty")
                    op.report({'INFO'}, e)
                    continue
                images.add(img)
        # materials w/o tev configs
        try:
            tev = context.scene.brres.tevConfigs[mat.brres.tevID]
        except KeyError:
            numProblems += 1
            op.report({'INFO'}, f"Material '{mat.name}' lacks a TEV configuration")
            continue
        # tev problems
        usedTevs.add(tev)
        numTextures = len(mat.brres.textures)
        numIndTransforms = len(mat.brres.indSettings.transforms)
        numLightChans = len(mat.brres.lightChans)
        for stage in tev.stages:
            # indirect problems
            if stage.indSettings.enable:
                # invalid texture references
                indTexSlot = tev.indTexSlots[stage.indSettings.slot - 1]
                if indTexSlot > numTextures:
                    numProblems += 1
                    e = (f"Stage '{stage.name}' of TEV config '{tev.name}' references texture slot "
                         f"{indTexSlot} for its indirect texture, but material '{mat.name}', which "
                         f"uses this config, lacks enough textures for this")
                    op.report({'INFO'}, e)
                # invalid transform references
                mtxSlot = stage.indSettings.mtxSlot
                if mtxSlot > numIndTransforms:
                    numProblems += 1
                    e = (f"Stage '{stage.name}' of TEV config '{tev.name}' references indirect "
                         f"texture transform slot {mtxSlot}, but material '{mat.name}', which uses "
                         f"this config, lacks enough indirect texture transforms for this")
                    op.report({'INFO'}, e)
            # invalid references in main tev args
            texSlot = stage.sels.texSlot
            rasSel = stage.sels.rasterSel
            args = (*stage.colorParams.args, *stage.alphaParams.args)
            # invalid texture references
            if texSlot > numTextures and ('TEX_COLOR' in args or 'TEX_ALPHA' in args):
                numProblems += 1
                e = (f"Stage '{stage.name}' of TEV config '{tev.name}' references texture slot "
                     f"{texSlot}, but material '{mat.name}', which uses this config, lacks enough "
                     f"textures for this")
                op.report({'INFO'}, e)
            # invalid light channel references
            if rasSel.startswith('COLOR'):
                rasSlot = int(rasSel[-1]) + 1
                if rasSlot > numLightChans and ('RASTER_COLOR' in args or 'RASTER_ALPHA' in args):
                    numProblems += 1
                    e = (f"Stage '{stage.name}' of TEV config '{tev.name}' references lighting "
                         f"channel slot {rasSlot}, but material '{mat.name}', which uses this "
                         f"config, lacks enough lighting channels for this")
                    op.report({'INFO'}, e)
    for tev in usedTevs:
        # no stages
        if len([stage for stage in tev.stages if not stage.hide]) == 0:
            numProblems += 1
            op.report({'INFO'}, f"TEV config '{tev.name}' doesn't have any enabled stages")
        # referencing standard color slot 1 before it's set
        chanInfo = (
            ("RGB", "colorParams", 'STANDARD_0_COLOR'),
            ("Alpha", "alphaParams", 'STANDARD_0_ALPHA')
        )
        reg0SetStages = {}
        for chanName, paramsName, argName in chanInfo: # do this first for color, then for alpha
            # first, get first stage where standard slot 1 is set
            firstStageSet = 0
            for stage in tev.stages:
                params = getattr(stage, paramsName)
                if params.output == '0':
                    break
                firstStageSet += 1
            reg0SetStages[chanName] = firstStageSet
            # then, find stages before that where standard slot 1 is used
            for stage in tev.stages[:firstStageSet + 1]: # stage where it's set gets included
                params = getattr(stage, paramsName)
                if argName in stage.colorParams.args or argName in stage.alphaParams.args:
                    numProblems += 1
                    e = (f"Stage '{stage.name}' of TEV config '{tev.name}' references Standard "
                         f"Color Slot 1 {chanName} before a value is written to that register (its "
                         f"initial state is undefined)")
                    op.report({'INFO'}, e)
    # image problems
    imgMax = gx.MAX_TEXTURE_SIZE
    for img in images:
        # non-power of 2 dims
        dims = np.array(img.size, dtype=int)
        if any(bin(dim).count("1") > 1 for dim in dims):
            if not op.includeSuppressed and img.brres.warnSupPow2:
                numSuppressed += 1
            else:
                numProblems += 1
                e = f"Image '{img.name}' has dimensions that aren't both powers of 2: {tuple(dims)}"
                op.report({'INFO'}, e)
        # excessive size
        if np.any(dims > imgMax):
            if not op.includeSuppressed and img.brres.warnSupSize:
                numSuppressed += 1
            else:
                numProblems += 1
                e = f"Image '{img.name}' has dimensions that aren't both <= {imgMax}: {tuple(dims)}"
                op.report({'INFO'}, e)
        # improper mipmap dims
        for mm in img.brres.mipmaps:
            dims //= 2
            mmImg: bpy.types.Image = mm.img
            if mmImg is not None:
                mmDims = np.array(mmImg.size, dtype=int)
                if np.any(mmDims != dims):
                    numProblems += 1
                    e = (f"Image '{img.name}' has a mipmap '{mmImg.name}' with improper dimensions "
                         f"(should be {tuple(dims)}, but are {tuple(mmDims)})")
                    op.report({'INFO'}, e)
    return (numProblems, numSuppressed)


def WarningSuppressionProperty():
    """Property for suppressing BRRES verifier warnings."""
    return bpy.props.BoolProperty(
        name="BRRES Warning",
        description="Click to suppress so that this will be ignored by the verifier",
        default=False,
        options=set()
    )


def drawWarningUI(layout: bpy.types.UILayout, text: str,
                  suppressionData = None, suppressionProp = ""):
    """Draw a BRRES verifier warning, optionally with a suppression button."""
    warnRow = layout.row()
    warnRow.alignment = 'CENTER'
    labelRow = warnRow.row()
    labelRow.alignment = 'CENTER'
    labelRow.label(text="", icon='ERROR')
    labelRow.label(text=text)
    if suppressionData:
        suppressionEnabled = getattr(suppressionData, suppressionProp)
        labelRow.enabled = not suppressionEnabled
        supIcon = 'HIDE_ON' if suppressionEnabled else 'HIDE_OFF'
        warnRow.prop(suppressionData, suppressionProp, text="", icon=supIcon, emboss=False)


class VerifyBRRES(bpy.types.Operator):
    """Verify the scene's BRRES settings"""

    bl_idname = "brres.verify"
    bl_label = "Verify BRRES Settings"

    limitTo: bpy.props.EnumProperty(
        name="Limit To",
        description="Data to verify (only includes assets that would be used by the corresponding export setting)", # pylint: disable=line-too-long
        items=(
            ('ALL', "All", ""),
            ('SELECTED', "Selected", ""),
            ('VISIBLE', "Visible", ""),
        ),
        default='ALL'
    )

    includeSuppressed: bpy.props.BoolProperty(
        name="Bypass Warning Suppression",
        description="Report all detected problems, including those flagged to be ignored",
        default=False
    )

    def execute(self, context: bpy.types.Context):
        self.report({'INFO'}, "Searching for BRRES warnings...")
        limiter = ObjectLimiterFactory.create(context, self.limitTo)
        warns, suppressed = verifyBRRES(self, context, limiter)
        if warns:
            plural = "s" if warns > 1 else ""
            sup = f", plus {suppressed} suppressed" if suppressed else ""
            e = f"{warns} BRRES warning{plural} detected{sup}. Check the Info Log for details"
            self.report({'WARNING'}, e)
        elif suppressed:
            plural = "s" if suppressed > 1 else ""
            self.report({'INFO'}, f"{suppressed} BRRES warning{plural} detected & suppressed")
        else:
            self.report({'INFO'}, "No BRRES warnings detected")
        return {'FINISHED'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        self.layout.prop(self, "limitTo")
        self.layout.prop(self, "includeSuppressed")


def drawOp(self, context: bpy.types.Context):
    layout: bpy.types.UILayout = self.layout
    layout.separator()
    layout.operator(VerifyBRRES.bl_idname, text="Verify BRRES Settings", icon='CHECKMARK')
