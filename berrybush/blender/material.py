# 3rd party imports
import bpy
# internal imports
from .common import (
    UNDOCUMENTED, PropertyPanel,
    drawColumnSeparator, drawCheckedProp, drawProp, getPropName, filterMats
)
from .proputils import CloneablePropertyGroup, CustomIDPropertyGroup
from .texture import TexSettings, TextureTransform
from ..wii import gx, transform as tf
from ..wii.alias import alias


COMPARE_OP_ENUM_ITEMS = (
    ('NEVER', "Fail", ""),
    ('LESS', "<", ""),
    ('EQUAL', "==", ""),
    ('LEQUAL', "<=", ""),
    ('GREATER', ">", ""),
    ('NEQUAL', "!=", ""),
    ('GEQUAL', ">=", ""),
    ('ALWAYS', "Pass", ""),
)


_texTransformGens = {
    'MAYA': tf.MayaMtxGen2D,
    'XSI': tf.XSIMtxGen2D,
    'MAX': tf.MaxMtxGen2D
}


class IndTexIndividualSettings(bpy.types.PropertyGroup):

    _scaleItems = (
        ('0', "1", ""),
        ('1', "2", ""),
        ('2', "4", ""),
        ('3', "8", ""),
        ('4', "16", ""),
        ('5', "32", ""),
        ('6', "64", ""),
        ('7', "128", ""),
        ('8', "256", ""),
        ('9', "512", ""),
        ('10', "1,024", ""),
        ('11', "2,048", ""),
        ('12', "4,096", ""),
        ('13', "8,196", ""),
        ('14', "16,392", ""),
        ('15', "32,768", ""),
    )

    mode: bpy.props.EnumProperty(
        name="Indirect Mode",
        description="Distortion method for this indirect texture (all rendered as Warp in BerryBush)", # pylint: disable=line-too-long
        items=(
            ('WARP', "Warp", UNDOCUMENTED),
            ('NORMAL_MAP', "Normal Mapping", UNDOCUMENTED),
            ('NORMAL_MAP_SPEC', "Normal Mapping (Specular)", UNDOCUMENTED),
            ('USER_0', "Custom 1", UNDOCUMENTED),
            ('USER_1', "Custom 2", UNDOCUMENTED),
        ),
        default='WARP',
        options=set()
    )

    lightSlot: bpy.props.IntProperty(
        name="Light Slot",
        description=UNDOCUMENTED,
        default=1,
        min=1,
        max=128,
        options=set()
    )

    scaleU: bpy.props.EnumProperty(
        name="U Coordinate Scale",
        description="Scaling applied to coordinates for indirect texture sampling (U axis)",
        items=(
            ('DIV_1', "U Unscaled", ""),
            ('DIV_2', "U ÷ 2", ""),
            ('DIV_4', "U ÷ 4", ""),
            ('DIV_8', "U ÷ 8", ""),
            ('DIV_16', "U ÷ 16", ""),
            ('DIV_32', "U ÷ 32", ""),
            ('DIV_64', "U ÷ 64", ""),
            ('DIV_128', "U ÷ 128", ""),
            ('DIV_256', "U ÷ 256", ""),
        ),
        default='DIV_1'
    )

    scaleV: bpy.props.EnumProperty(
        name="V Coordinate Scale",
        description="Scaling applied to coordinates for indirect texture sampling (V axis)",
        items=(
            ('DIV_1', "V Unscaled", ""),
            ('DIV_2', "V ÷ 2", ""),
            ('DIV_4', "V ÷ 4", ""),
            ('DIV_8', "V ÷ 8", ""),
            ('DIV_16', "V ÷ 16", ""),
            ('DIV_32', "V ÷ 32", ""),
            ('DIV_64', "V ÷ 64", ""),
            ('DIV_128', "V ÷ 128", ""),
            ('DIV_256', "V ÷ 256", ""),
        ),
        default='DIV_1'
    )


class IndTransform(CustomIDPropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {f"{m.name}/{t.name}": t for m in mats for t in m.brres.indSettings.transforms}

    @classmethod
    def defaultName(cls):
        return "Indirect Transform"

    def onNameChange(self, newName: str):
        self.transform.name = newName # this makes fcurves display the ind transform name

    transform: bpy.props.PointerProperty(type=TextureTransform)


class IndTexSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {m.name: m.brres.indSettings for m in mats}

    tex1: bpy.props.PointerProperty(type=IndTexIndividualSettings)
    tex2: bpy.props.PointerProperty(type=IndTexIndividualSettings)
    tex3: bpy.props.PointerProperty(type=IndTexIndividualSettings)
    tex4: bpy.props.PointerProperty(type=IndTexIndividualSettings)
    transforms: IndTransform.CustomIDCollectionProperty(gx.MAX_INDIRECT_MTCS)

    texConfigs = alias("tex1", "tex2", "tex3", "tex4")


class ColorRegSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {m.name: m.brres.colorRegs for m in mats}

    constant1: bpy.props.FloatVectorProperty(
        name="Constant Slot 1",
        description="Color register whose value can be read by TEV stages",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    constant2: bpy.props.FloatVectorProperty(
        name="Constant Slot 2",
        description="Color register whose value can be read by TEV stages",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    constant3: bpy.props.FloatVectorProperty(
        name="Constant Slot 3",
        description="Color register whose value can be read by TEV stages",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    constant4: bpy.props.FloatVectorProperty(
        name="Constant Slot 4",
        description="Color register whose value can be read by TEV stages",
        default=(0, 0, 0, 1),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    standard1: bpy.props.FloatVectorProperty(
        name="Standard Slot 1",
        description="Color register whose value can be read and written to by TEV stages",
        default=(1, 1, 1, 1),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
        options=set(),
        get=lambda self: (1, 1, 1, 1), # slot 1 is locked to solid white
        set=lambda self, value: None
    )

    standard2: bpy.props.FloatVectorProperty(
        name="Standard Slot 2",
        description="Color register whose value can be read and written to by TEV stages",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    standard3: bpy.props.FloatVectorProperty(
        name="Standard Slot 3",
        description="Color register whose value can be read and written to by TEV stages",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    standard4: bpy.props.FloatVectorProperty(
        name="Standard Slot 4",
        description="Color register whose value can be read and written to by TEV stages",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    constant = alias("constant1", "constant2", "constant3", "constant4")
    standard = alias("standard1", "standard2", "standard3", "standard4")


class LightChannelColorAlphaSettings(bpy.types.PropertyGroup):

    difFromReg: bpy.props.BoolProperty(
        name="Use Diffuse Register",
        description="Use a constant register instead of vertex colors for the diffuse color",
        default=True,
        options=set()
    )

    ambFromReg: bpy.props.BoolProperty(
        name="Use Ambient Register",
        description="Use a constant color instead of vertex colors for the ambient color",
        default=True,
        options=set()
    )

    enableDiffuse: bpy.props.BoolProperty(
        name="Enable Diffuse Lighting",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    diffuseMode: bpy.props.EnumProperty(
        name="Diffuse Lighting",
        description=UNDOCUMENTED,
        items=(
            ('NONE', "None", UNDOCUMENTED),
            ('SIGNED', "Signed", UNDOCUMENTED),
            ('CLAMPED', "Clamped", UNDOCUMENTED),
        ),
        default='NONE',
        options=set()
    )

    enableAttenuation: bpy.props.BoolProperty(
        name="Enable Attenuation",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    attenuationMode: bpy.props.EnumProperty(
        name="Attenuation",
        description=UNDOCUMENTED,
        items=(
            ('SPECULAR', "Specular", UNDOCUMENTED),
            ('SPOTLIGHT', "Spotlight", UNDOCUMENTED),
        ),
        default='SPOTLIGHT',
        options=set()
    )

    enabledLights: bpy.props.BoolVectorProperty(
        name="Enabled Lights",
        description=UNDOCUMENTED,
        default=[False] * 8,
        size=8,
        options=set()
    )


class LightChannelSettings(CustomIDPropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {f"{m.name}/{lc.name}": lc for m in mats for lc in m.brres.lightChans}

    @classmethod
    def defaultName(cls):
        return "Lighting Channel"

    def drawListItem(self: bpy.types.UIList, context: bpy.types.Context, layout: bpy.types.UILayout,
                     data, item, icon: int, active_data, active_property: str, index=0, flt_flag=0):
        # pylint: disable=attribute-defined-outside-init
        self.use_filter_show = False
        layout.prop(item, "name", text="", emboss=False, icon='LIGHT')

    def usesVertClrs(self):
        """Whether this light channel relies on vertex colors for its output."""
        return not all((
            self.colorSettings.difFromReg,
            self.colorSettings.ambFromReg,
            self.alphaSettings.difFromReg,
            self.alphaSettings.ambFromReg
        ))

    difColor: bpy.props.FloatVectorProperty(
        name="Diffuse Register",
        description="Constant diffuse color if not taken from vertex colors",
        default=(0, 0, 0, 1),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    ambColor: bpy.props.FloatVectorProperty(
        name="Ambient Register",
        description="Constant ambient color if not taken from vertex colors",
        default=(0, 0, 0, 0),
        min=0,
        max=1,
        size=4,
        subtype='COLOR_GAMMA',
    )

    colorSettings: bpy.props.PointerProperty(
        name="Color Settings",
        type=LightChannelColorAlphaSettings
    )

    alphaSettings: bpy.props.PointerProperty(
        name="Alpha Settings",
        type=LightChannelColorAlphaSettings
    )


class AlphaSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {m.name: m.brres.alphaSettings for m in mats}

    enableBlendOp: bpy.props.BoolProperty(
        name="Enable Blending Operation",
        description="Blend colors being written to the framebuffer with the colors already present",
        default=False,
        options=set()
    )

    blendSrcFactor: bpy.props.EnumProperty(
        name="Source Factor",
        description="Factor by which the source color (color being written) should be multiplied",
        items=(
            ('ZERO', "0", ""),
            ('ONE', "1", ""),
            ('DST_COLOR', "Destination Color", ""),
            ('INV_DST_COLOR', "1 - Destination Color", ""),
            ('SRC_ALPHA', "Source Alpha", ""),
            ('INV_SRC_ALPHA', "1 - Source Alpha", ""),
            ('DST_ALPHA', "Destination Alpha", "Warning: Destination alpha not available for some framebuffer formats"), # pylint: disable=line-too-long
            ('INV_DST_ALPHA', "1 - Destination Alpha", "Warning: Destination alpha not available for some framebuffer formats"), # pylint: disable=line-too-long
        ),
        default='SRC_ALPHA',
        options=set()
    )

    blendOp: bpy.props.EnumProperty(
        name="Blending Operation",
        description="Equation to be used for blending",
        items=(
            ('+', "Src * Fac + Dst * Fac", ""),
            ('-', "Dst - Src", ""),
        ),
        default='+',
        options=set()
    )

    blendDstFactor: bpy.props.EnumProperty(
        name="Destination Factor",
        description="Factor by which the destination color (color being overwritten) should be multiplied", # pylint: disable=line-too-long
        items=(
            ('ZERO', "0", ""),
            ('ONE', "1", ""),
            ('SRC_COLOR', "Source Color", ""),
            ('INV_SRC_COLOR', "1 - Source Color", ""),
            ('SRC_ALPHA', "Source Alpha", ""),
            ('INV_SRC_ALPHA', "1 - Source Alpha", ""),
            ('DST_ALPHA', "Destination Alpha", "Warning: Destination alpha not available for some framebuffer formats"), # pylint: disable=line-too-long
            ('INV_DST_ALPHA', "1 - Destination Alpha", "Warning: Destination alpha not available for some framebuffer formats"), # pylint: disable=line-too-long
        ),
        default='INV_SRC_ALPHA',
        options=set()
    )

    enableLogicOp: bpy.props.BoolProperty(
        name="Enable Logical Operation",
        description="Before writing colors to the framebuffer, perform a bitwise logical operation between the new and old colors", # pylint: disable=line-too-long
        default=False,
        options=set()
    )

    logicOp: bpy.props.EnumProperty(
        name="Logical Operation",
        description="Value to write",
        items=(
            ('CLEAR', "0", ""),
            ('AND', "Source & Destination", ""),
            ('REVAND', "Source & ~Destination", ""),
            ('COPY', "Source", ""),
            ('INVAND', "~Source & Destination", ""),
            ('NOOP', "Destination", ""),
            ('XOR', "Source ^ Destination", ""),
            ('OR', "Source | Destination", ""),
            ('NOR', "~(Source | Destination)", ""),
            ('EQUIV', "~(Source ^ Destination)", ""),
            ('INV', "~Destination", ""),
            ('REVOR', "Source | ~Destination", ""),
            ('INVCOPY', "~Source", ""),
            ('INVOR', "~Source | Destination", ""),
            ('NAND', "~(Source & Destination)", ""),
            ('SET', "1", ""),
        ),
        default='COPY',
        options=set()
    )

    enableDither: bpy.props.BoolProperty(
        name="Enable Dithering",
        default=False,
        options=set()
    )

    enableColorUpdate: bpy.props.BoolProperty(
        name="Update Color Buffer",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    enableAlphaUpdate: bpy.props.BoolProperty(
        name="Update Alpha Buffer",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    cullMode: bpy.props.EnumProperty(
        name="Culling",
        description="Disable drawing of certain faces",
        items=(
            ('NONE', "None", ""),
            ('FRONT', "Front", ""),
            ('BACK', "Back", ""),
            ('BOTH', "Both", ""),
        ),
        default='NONE',
        options=set()
    )

    isXlu: bpy.props.BoolProperty(
        name="Translucent Render Group",
        description="Draw this object after all objects with this setting unchecked (takes precedence over draw priority)", # pylint: disable=line-too-long
        default=False,
        options=set()
    )

    testLogic: bpy.props.EnumProperty(
        name="",
        items=(
            ('AND', "And", ""),
            ('OR', "Or", ""),
            ('XOR', "Xor", ""),
            ('XNOR', "Xnor", ""),
        ),
        default='AND',
        options=set()
    )

    testComp1: bpy.props.EnumProperty(
        name="",
        items=COMPARE_OP_ENUM_ITEMS,
        default='ALWAYS',
        options=set()
    )

    testComp2: bpy.props.EnumProperty(
        name="",
        items=COMPARE_OP_ENUM_ITEMS,
        default='ALWAYS',
        options=set()
    )

    testComps = alias("testComp1", "testComp2")

    testVal1: bpy.props.FloatProperty(
        name="",
        default=0,
        min=0,
        max=1,
        subtype='FACTOR',
        options=set()
    )

    testVal2: bpy.props.FloatProperty(
        name="",
        default=0,
        min=0,
        max=1,
        subtype='FACTOR',
        options=set()
    )

    testVals = alias("testVal1", "testVal2")

    enableConstVal: bpy.props.BoolProperty(
        name="Enable Constant Alpha",
        description="Write a constant alpha value to the framebuffer (has no effect on blending, alpha test, etc. for this material). Warning: Only available for framebuffer formats that support reading destination alpha", # pylint: disable=line-too-long
        default=False,
        options=set()
    )

    constVal: bpy.props.FloatProperty(
        name="Send Constant Alpha",
        description="Constant alpha value to write (has no effect on blending, alpha test, etc. for this material). Warning: Only available for framebuffer formats that support reading destination alpha", # pylint: disable=line-too-long
        default=0,
        min=0,
        max=1,
        subtype='FACTOR',
        options=set()
    )


class DepthSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {m.name: m.brres.depthSettings for m in mats}

    enableDepthTest: bpy.props.BoolProperty(
        name="Depth Testing",
        description="Test fragments' depth values against those in the depth buffer and discard those that fail", # pylint: disable=line-too-long
        default=True,
        options=set()
    )

    enableDepthUpdate: bpy.props.BoolProperty(
        name="Update Buffer",
        description="Update the depth buffer when the test is passed",
        default=True,
        options=set()
    )

    depthFunc: bpy.props.EnumProperty(
        name="Function",
        description="Depth comparison test",
        items=COMPARE_OP_ENUM_ITEMS,
        default='LEQUAL',
        options=set()
    )


class MiscMatSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {m.name: m.brres.miscSettings for m in mats}

    useLightSet: bpy.props.BoolProperty(
        name="Use Light Set",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    lightSet: bpy.props.IntProperty(
        name="Light Set",
        description=UNDOCUMENTED,
        default=1,
        min=1,
        max=128,
        options=set()
    )

    useFogSet: bpy.props.BoolProperty(
        name="Use Fog Set",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    fogSet: bpy.props.IntProperty(
        name="Fog Set",
        description=UNDOCUMENTED,
        default=1,
        min=1,
        max=128,
        options=set()
    )

    texTransformMode: bpy.props.EnumProperty(
        name="Transform Mode",
        description="Set of conventions used for transforming this material's textures",
        items=(
            ('MAYA', "Maya", "Translation along rotated axes, rotation clockwise about center, scaling from bottom-left"), # pylint: disable=line-too-long
            ('XSI', "XSI", "Translation along absolute axes, rotation counterclockwise about bottom-left,  scaling from bottom-left"), # pylint: disable=line-too-long
            ('MAX', "3DS Max", "Translation along absolute axes, rotation clockwise about center,  scaling from center") # pylint: disable=line-too-long
        ),
        default='MAYA',
        options=set()
    )

    def getTexTransformGen(self):
        return _texTransformGens[self.texTransformMode]


class MatSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {m.name: m.brres for m in mats}

    tevID: bpy.props.StringProperty(options=set())
    textures: TexSettings.CustomIDCollectionProperty(gx.MAX_TEXTURES)
    indSettings: bpy.props.PointerProperty(type=IndTexSettings)
    colorRegs: bpy.props.PointerProperty(type=ColorRegSettings)
    lightChans: LightChannelSettings.CustomIDCollectionProperty(gx.MAX_CLR_ATTRS)
    alphaSettings: bpy.props.PointerProperty(type=AlphaSettings)
    depthSettings: bpy.props.PointerProperty(type=DepthSettings)
    miscSettings: bpy.props.PointerProperty(type=MiscMatSettings)


class MatPanel(PropertyPanel):
    bl_idname = "BRRES_PT_mat"
    bl_label = "BRRES Settings"
    bl_context = "material"
    bl_options = set()

    @classmethod
    def poll(cls, context: bpy.types.Context):
        mat = context.material
        return mat is not None and context.engine == "BERRYBUSH" and not mat.grease_pencil

    def draw(self, context: bpy.types.Context):
        matSettings = context.material.brres
        layout = self.layout
        matSettings.drawCloneUI(layout)


class IndTexPanel(PropertyPanel):
    bl_idname = "BRRES_PT_ind_tex"
    bl_label = "Indirect Textures"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        matSettings = context.material.brres
        try:
            tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        except KeyError:
            tevSettings = None
        indSettings = matSettings.indSettings
        indSettings.drawCloneUI(layout)
        for i in range(1, gx.MAX_INDIRECTS + 1):
            indTexSettings = getattr(indSettings, f"tex{i}")
            row = layout.row().split(factor=.15)
            labelCol = row.column()
            labelCol.alignment = 'RIGHT'
            labelCol.label(text=f"Slot {i}")
            dataRow = row.column().row().split(factor=.67)
            slot = tevSettings.indTexSlots[i - 1] if tevSettings else gx.MAX_TEXTURES
            dataRow.column().prop(indTexSettings, "mode", text="")
            lightRow = dataRow.column().row()
            lightRow.enabled = indTexSettings.mode in {'NORMAL_MAP', 'NORMAL_MAP_SPEC'}
            lightRow.prop(indTexSettings, "lightSlot", text="")
            lightRow.label(icon='LIGHT')
            scaleRow = layout.row().split(factor=.15)
            iconCol = scaleRow.column().row()
            iconCol.alignment = 'RIGHT'
            try:
                img = matSettings.textures[slot - 1].activeImg
                img.preview_ensure()
                iconCol.label(icon_value=img.preview.icon_id)
            except (IndexError, AttributeError):
                iconCol.label(icon='TEXTURE')
            scaleRow = scaleRow.column().row().split(factor=.5)
            scaleRow.column().prop(indTexSettings, "scaleU", text="")
            scaleRow.column().prop(indTexSettings, "scaleV", text="")
            if i < gx.MAX_INDIRECTS:
                # draw a little separator between slots
                # (i could alternatively put each one in a box but this is prettier imo)
                layout.box()


class IndTransformPanel(PropertyPanel):
    bl_idname = "BRRES_PT_ind_transforms"
    bl_label = "Transforms"
    bl_parent_id = "BRRES_PT_ind_tex"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        matSettings = context.material.brres
        indSettings = matSettings.indSettings
        indSettings.transforms.drawList(layout)
        try:
            transform = indSettings.transforms.activeItem().transform
            layout.prop(transform, "translation")
            layout.prop(transform, "rotation")
            layout.prop(transform, "scale")
        except IndexError:
            pass


class ColorRegPanel(PropertyPanel):
    bl_idname = "BRRES_PT_color_regs"
    bl_label = "Color Registers"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        matSettings = context.material.brres
        clrRegs = matSettings.colorRegs
        clrRegs.drawCloneUI(layout)
        labels = ("Standard Colors", "Constant Colors")
        icons = ('RESTRICT_COLOR_ON', 'RESTRICT_COLOR_OFF')
        propNames = ("standard", "constant")
        for label, icon, propName in zip(labels, icons, propNames):
            layout.label(text=label, icon=icon)
            for i in range(1, 5):
                sublayout = layout
                if propName == "standard" and i == 1:
                    sublayout = layout.row()
                    sublayout.enabled = False
                drawProp(sublayout, clrRegs, f"{propName}{i}", factor=.2, text=f"Slot {i}")


class LightChannelPanel(PropertyPanel):
    bl_idname = "BRRES_PT_light_chans"
    bl_label = "Lighting Channels"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context):
        layout = self.layout
        matSettings = context.material.brres
        matSettings.lightChans.drawList(layout)
        try:
            lightChan = matSettings.lightChans.activeItem()
            lightChan.drawCloneUI(layout)
            activeData = context.active_object.data
            if lightChan.usesVertClrs():
                hasAttr = False
                if isinstance(activeData, bpy.types.Mesh):
                    clrIdx = matSettings.lightChans.activeIdx
                    clrAttr = activeData.brres.meshAttrs.clrs[clrIdx]
                    hasAttr = clrAttr in activeData.uv_layers or clrAttr in activeData.attributes
                labelRow = layout.row()
                labelRow.alignment = 'CENTER'
                if hasAttr:
                    labelRow.label(text=f"Vertex Colors: {clrAttr}", icon='GROUP_VCOL')
                else:
                    labelRow.label(text="No associated vertex colors for this object", icon='ERROR')
            settingsNames = ("colorSettings", "alphaSettings")
            icons = ('IMAGE_RGB', 'IMAGE_ALPHA')
            for settingsName, icon in zip(settingsNames, icons):
                s = getattr(lightChan, settingsName) # settings for color or alpha
                box = layout.box()
                labelRow = box.row()
                labelRow.alignment = 'CENTER'
                labelRow.label(text=getPropName(lightChan, settingsName), icon=icon)
                drawCheckedProp(box, s, "difFromReg", lightChan, "difColor", decorate=True)
                drawCheckedProp(box, s, "ambFromReg", lightChan, "ambColor", decorate=True)
                drawCheckedProp(box, s, "enableDiffuse", s, "diffuseMode")
                drawCheckedProp(box, s, "enableAttenuation", s, "attenuationMode")
                lightRow = box.row().split(factor=.4)
                labelCol = lightRow.column()
                labelCol.alignment = 'RIGHT'
                labelCol.label(text=getPropName(s, "enabledLights"))
                lightRow.column().row().prop(s, "enabledLights", text="")
        except IndexError:
            pass


class AlphaSettingsPanel(PropertyPanel):
    bl_idname = "BRRES_PT_mat_alpha"
    bl_label = "Blending & Discarding"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        matSettings = context.material.brres
        alphaSettings = matSettings.alphaSettings
        alphaSettings.drawCloneUI(layout)
        drawCheckedProp(layout, alphaSettings, "enableBlendOp", alphaSettings, "blendOp")
        col = layout.column()
        col.enabled = alphaSettings.enableBlendOp and alphaSettings.blendOp == "+"
        col.prop(alphaSettings, "blendSrcFactor")
        drawColumnSeparator(col)
        col.prop(alphaSettings, "blendDstFactor")
        row = layout.row()
        row.enabled = not alphaSettings.enableBlendOp
        drawCheckedProp(row, alphaSettings, "enableLogicOp", alphaSettings, "logicOp")
        layout.prop(alphaSettings, "enableDither")
        layout.prop(alphaSettings, "enableColorUpdate")
        layout.prop(alphaSettings, "enableAlphaUpdate")
        drawCheckedProp(layout, alphaSettings, "enableConstVal", alphaSettings, "constVal")
        layout.prop(alphaSettings, "cullMode", expand=True)
        layout.prop(alphaSettings, "isXlu")
        alphaTestLabelRow = layout.row()
        alphaTestLabelRow.alignment = 'CENTER'
        alphaTestLabelRow.label(text="Discard pixels where alpha value fails this test:")
        alphaTestRow = layout.row(align=True)
        specialComps = {'NEVER', 'ALWAYS'}
        for i in range(1, 3):
            propStr = f"testComp{i}"
            alphaTestRow.prop(alphaSettings, propStr, text="")
            valCol = alphaTestRow.column(align=True)
            valCol.enabled = getattr(alphaSettings, propStr) not in specialComps
            valCol.prop(alphaSettings, f"testVal{i}", text="")
            if i == 1:
                alphaTestRow.prop(alphaSettings, "testLogic", text="")


class DepthSettingsPanel(PropertyPanel):
    bl_idname = "BRRES_PT_mat_depth"
    bl_label = "Depth"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        matSettings = context.material.brres
        depthSettings = matSettings.depthSettings
        depthSettings.drawCloneUI(layout)
        layout.prop(depthSettings, "enableDepthTest")
        col = layout.column()
        col.enabled = depthSettings.enableDepthTest
        col.prop(depthSettings, "enableDepthUpdate")
        drawColumnSeparator(col)
        col.prop(depthSettings, "depthFunc")


class MatMiscPanel(PropertyPanel):
    bl_idname = "BRRES_PT_mat_misc"
    bl_label = "Misc."
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        matSettings = context.material.brres
        miscSettings = matSettings.miscSettings
        miscSettings.drawCloneUI(layout)
        drawCheckedProp(layout, miscSettings, "useLightSet", miscSettings, "lightSet")
        drawCheckedProp(layout, miscSettings, "useFogSet", miscSettings, "fogSet")
