# third party imports
import bpy
# internal imports
from .common import UNDOCUMENTED, UI_COL_SEP, PropertyPanel, drawProp, drawIcon
from .proputils import CustomIDPropertyGroup
from ..wii import gx
from ..wii.alias import alias


class TevStageSelSettings(bpy.types.PropertyGroup):

    def constColorItems(self, context: bpy.types.Context):
        from .. import ICONS # pylint: disable=import-outside-toplevel
        return (
            ('VAL_8_8', "1", "", ICONS['VAL_8_8'].icon_id, 0),
            ('VAL_7_8', "0.875", "", ICONS['VAL_7_8'].icon_id, 1),
            ('VAL_6_8', "0.75", "", ICONS['VAL_6_8'].icon_id, 2),
            ('VAL_5_8', "0.625", "", ICONS['VAL_5_8'].icon_id, 3),
            ('VAL_4_8', "0.5", "", ICONS['VAL_4_8'].icon_id, 4),
            ('VAL_3_8', "0.375", "", ICONS['VAL_3_8'].icon_id, 5),
            ('VAL_2_8', "0.25", "", ICONS['VAL_2_8'].icon_id, 6),
            ('VAL_1_8', "0.125", "", ICONS['VAL_1_8'].icon_id, 7),
            ('VAL_0_8', "0", "", ICONS['VAL_0_8'].icon_id, 8),
            ('RGB_0', "Slot 1 RGB", "RGB from this material's first constant color slot", ICONS['RGB'].icon_id, 9), # pylint: disable=line-too-long
            ('RGB_1', "Slot 2 RGB", "RGB from this material's second constant color slot", ICONS['RGB'].icon_id, 10), # pylint: disable=line-too-long
            ('RGB_2', "Slot 3 RGB", "RGB from this material's third constant color slot", ICONS['RGB'].icon_id, 11), # pylint: disable=line-too-long
            ('RGB_3', "Slot 4 RGB", "RGB from this material's fourth constant color slot", ICONS['RGB'].icon_id, 12), # pylint: disable=line-too-long
            ('R_0', "Slot 1 R", "R from this material's first constant color slot", 'COLORSET_01_VEC', 13), # pylint: disable=line-too-long
            ('R_1', "Slot 2 R", "R from this material's second constant color slot", 'COLORSET_01_VEC', 14), # pylint: disable=line-too-long
            ('R_2', "Slot 3 R", "R from this material's third constant color slot", 'COLORSET_01_VEC', 15), # pylint: disable=line-too-long
            ('R_3', "Slot 4 R", "R from this material's fourth constant color slot", 'COLORSET_01_VEC', 16), # pylint: disable=line-too-long
            ('G_0', "Slot 1 G", "G from this material's first constant color slot", 'COLORSET_03_VEC', 17), # pylint: disable=line-too-long
            ('G_1', "Slot 2 G", "G from this material's second constant color slot", 'COLORSET_03_VEC', 18), # pylint: disable=line-too-long
            ('G_2', "Slot 3 G", "G from this material's third constant color slot", 'COLORSET_03_VEC', 19), # pylint: disable=line-too-long
            ('G_3', "Slot 4 G", "G from this material's fourth constant color slot", 'COLORSET_03_VEC', 20), # pylint: disable=line-too-long
            ('B_0', "Slot 1 B", "B from this material's first constant color slot", 'COLORSET_04_VEC', 21), # pylint: disable=line-too-long
            ('B_1', "Slot 2 B", "B from this material's second constant color slot", 'COLORSET_04_VEC', 22), # pylint: disable=line-too-long
            ('B_2', "Slot 3 B", "B from this material's third constant color slot", 'COLORSET_04_VEC', 23), # pylint: disable=line-too-long
            ('B_3', "Slot 4 B", "B from this material's fourth constant color slot", 'COLORSET_04_VEC', 24), # pylint: disable=line-too-long
            ('A_0', "Slot 1 A", "A from this material's first constant color slot", 'COLORSET_13_VEC', 25), # pylint: disable=line-too-long
            ('A_1', "Slot 2 A", "A from this material's second constant color slot", 'COLORSET_13_VEC', 26), # pylint: disable=line-too-long
            ('A_2', "Slot 3 A", "A from this material's third constant color slot", 'COLORSET_13_VEC', 27), # pylint: disable=line-too-long
            ('A_3', "Slot 4 A", "A from this material's fourth constant color slot", 'COLORSET_13_VEC', 28), # pylint: disable=line-too-long
        )

    def constAlphaItems(self, context: bpy.types.Context):
        from .. import ICONS # pylint: disable=import-outside-toplevel
        return (
            ('VAL_8_8', "1", "", ICONS['VAL_8_8'].icon_id, 0),
            ('VAL_7_8', "0.875", "", ICONS['VAL_7_8'].icon_id, 1),
            ('VAL_6_8', "0.75", "", ICONS['VAL_6_8'].icon_id, 2),
            ('VAL_5_8', "0.625", "", ICONS['VAL_5_8'].icon_id, 3),
            ('VAL_4_8', "0.5", "", ICONS['VAL_4_8'].icon_id, 4),
            ('VAL_3_8', "0.375", "", ICONS['VAL_3_8'].icon_id, 5),
            ('VAL_2_8', "0.25", "", ICONS['VAL_2_8'].icon_id, 6),
            ('VAL_1_8', "0.125", "", ICONS['VAL_1_8'].icon_id, 7),
            ('VAL_0_8', "0", "", ICONS['VAL_0_8'].icon_id, 8),
            ('R_0', "Slot 1 R", "R from this material's first constant color slot", 'COLORSET_01_VEC', 9), # pylint: disable=line-too-long
            ('R_1', "Slot 2 R", "R from this material's second constant color slot", 'COLORSET_01_VEC', 10), # pylint: disable=line-too-long
            ('R_2', "Slot 3 R", "R from this material's third constant color slot", 'COLORSET_01_VEC', 11), # pylint: disable=line-too-long
            ('R_3', "Slot 4 R", "R from this material's fourth constant color slot", 'COLORSET_01_VEC', 12), # pylint: disable=line-too-long
            ('G_0', "Slot 1 G", "G from this material's first constant color slot", 'COLORSET_03_VEC', 13), # pylint: disable=line-too-long
            ('G_1', "Slot 2 G", "G from this material's second constant color slot", 'COLORSET_03_VEC', 14), # pylint: disable=line-too-long
            ('G_2', "Slot 3 G", "G from this material's third constant color slot", 'COLORSET_03_VEC', 15), # pylint: disable=line-too-long
            ('G_3', "Slot 4 G", "G from this material's fourth constant color slot", 'COLORSET_03_VEC', 16), # pylint: disable=line-too-long
            ('B_0', "Slot 1 B", "B from this material's first constant color slot", 'COLORSET_04_VEC', 17), # pylint: disable=line-too-long
            ('B_1', "Slot 2 B", "B from this material's second constant color slot", 'COLORSET_04_VEC', 18), # pylint: disable=line-too-long
            ('B_2', "Slot 3 B", "B from this material's third constant color slot", 'COLORSET_04_VEC', 19), # pylint: disable=line-too-long
            ('B_3', "Slot 4 B", "B from this material's fourth constant color slot", 'COLORSET_04_VEC', 20), # pylint: disable=line-too-long
            ('A_0', "Slot 1 A", "A from this material's first constant color slot", 'COLORSET_13_VEC', 21), # pylint: disable=line-too-long
            ('A_1', "Slot 2 A", "A from this material's second constant color slot", 'COLORSET_13_VEC', 22), # pylint: disable=line-too-long
            ('A_2', "Slot 3 A", "A from this material's third constant color slot", 'COLORSET_13_VEC', 23), # pylint: disable=line-too-long
            ('A_3', "Slot 4 A", "A from this material's fourth constant color slot", 'COLORSET_13_VEC', 24), # pylint: disable=line-too-long
        )

    def rasterSelItems(self, context: bpy.types.Context):
        lcNames = [f"Channel {i}" for i in range(1, gx.MAX_CLR_ATTRS + 1)]
        try:
            for i, lc in enumerate(context.material.brres.lightChans):
                lcNames[i] += f" ({lc.name}) "
        except AttributeError:
            pass
        return (
            ('COLOR_0', lcNames[0], "First light channel of this material"),
            ('COLOR_1', lcNames[1], "Second light channel of this material"),
            ('ALPHA_BUMP', "Bump Alpha", UNDOCUMENTED),
            ('NORMALIZED_ALPHA_BUMP', "Normalized Bump Alpha", UNDOCUMENTED),
            ('ZERO', "Zero", "All channels set to 0"),
        )

    texSlot: bpy.props.IntProperty(
        name="Texture Slot",
        description="Material texture slot for this stage's texture color",
        min=1,
        max=gx.MAX_TEXTURES,
        default=1,
        options=set()
    )

    texSwapSlot: bpy.props.IntProperty(
        name="Texture Swap Slot",
        description="Color swap slot to be applied to the texture color for this stage",
        min=1,
        max=gx.MAX_COLOR_SWAPS,
        default=1,
        options=set()
    )

    rasterSel: bpy.props.EnumProperty(
        name="Raster Color",
        description="Source for this stage's raster color",
        items=rasterSelItems,
        default=4,
        options=set()
    )

    rasSwapSlot: bpy.props.IntProperty(
        name="Raster Swap Slot",
        description="Color swap slot to be applied to the raster color for this stage",
        min=1,
        max=gx.MAX_COLOR_SWAPS,
        default=1,
        options=set()
    )

    constColor: bpy.props.EnumProperty(
        name="Constant Color",
        description="Value to use for the 'Constant RGB' input option",
        items=constColorItems,
        default=9,
        options=set()
    )

    constAlpha: bpy.props.EnumProperty(
        name="Constant Alpha",
        description="Value to use for the 'Constant Alpha' input option",
        items=constAlphaItems,
        default=21,
        options=set()
    )


class TevStageIndSettings(bpy.types.PropertyGroup):

    enable: bpy.props.BoolProperty(
        name="Enable",
        description="Perform indirect texturing for this stage",
        default=False,
        options=set()
    )

    slot: bpy.props.IntProperty(
        name="Indirect Texture Slot",
        description="Source to use for indirect texturing",
        min=1,
        max=gx.MAX_INDIRECTS,
        default=1,
        options=set()
    )

    fmt: bpy.props.EnumProperty(
        name="Color Bits Used",
        description="Undocumented (all options except 8 not accurately rendered in BerryBush)",
        items=(
            ('ALL_8', "8", UNDOCUMENTED),
            ('LOWER_5', "5", UNDOCUMENTED),
            ('LOWER_4', "4", UNDOCUMENTED),
            ('LOWER_3', "3", UNDOCUMENTED),
        ),
        default='ALL_8',
        options=set()
    )

    enableBias: bpy.props.BoolVectorProperty(
        name="Enable",
        description=UNDOCUMENTED,
        size=3,
        default=(True, True, True),
        options=set()
    )

    bumpAlphaComp: bpy.props.EnumProperty(
        name="Bump Alpha Channel",
        description=UNDOCUMENTED,
        items=(
            ('NONE', "None", UNDOCUMENTED),
            ('S', "Alpha", UNDOCUMENTED),
            ('T', "Blue", UNDOCUMENTED),
            ('U', "Green", UNDOCUMENTED),
        ),
        default='NONE',
        options=set()
    )

    mtxType: bpy.props.EnumProperty(
        name="Transform Type",
        description=UNDOCUMENTED,
        items=(
            ('STATIC', "Static", UNDOCUMENTED),
            ('S', "Dynamic (U)", UNDOCUMENTED),
            ('T', "Dynamic (V)", UNDOCUMENTED),
        ),
        default='STATIC',
        options=set()
    )

    mtxSlot: bpy.props.IntProperty(
        name="Transform Slot",
        description=UNDOCUMENTED,
        min=1,
        max=gx.MAX_INDIRECT_MTCS,
        default=1,
        options=set()
    )

    wrapItems = (
        ('OFF', "Off", ""),
        ('ON_256', "256", ""),
        ('ON_128', "128", ""),
        ('ON_64', "64", ""),
        ('ON_32', "32", ""),
        ('ON_16', "16", ""),
        ('ON_0', "0", ""),
    )

    wrapU: bpy.props.EnumProperty(
        name="U Wrap",
        description="Distance before repeating texture (U axis) (not reflected in BerryBush)",
        items=wrapItems,
        default='OFF',
        options=set()
    )

    wrapV: bpy.props.EnumProperty(
        name="V Wrap",
        description="Distance before repeating texture (V axis) (not reflected in BerryBush)",
        items=wrapItems,
        default='OFF',
        options=set()
    )

    utcLOD: bpy.props.BoolProperty(
        name="Apply for Mipmap Calculations",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    addPrev: bpy.props.BoolProperty(
        name="Add Previous Stage",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )


class TevStageCalcParams(bpy.types.PropertyGroup):

    compMode: bpy.props.BoolProperty(
        name="Comparison Mode",
        description="Use the comparison-based equation for this computation",
        default=False,
        options=set()
    )

    op: bpy.props.EnumProperty(
        name="Operator",
        description="Arithmetic operator between D and lerp(A, B, C) in this computation",
        items=(
            ('ADD_OR_GREATER', "+", ""),
            ('SUBTRACT_OR_EQUALS', "-", ""),
        ),
        default='ADD_OR_GREATER',
        options=set()
    )

    scale: bpy.props.EnumProperty(
        name="Scale",
        description="Scale applied to the result of this computation, before clamping",
        items=(
            ('HALF_OR_BY_CHANNEL_OR_A', "0.5", ""),
            ('ONE_OR_R', "1", ""),
            ('TWO_OR_RG', "2", ""),
            ('FOUR_OR_RGB', "4", ""),
        ),
        default='ONE_OR_R',
        options=set()
    )

    bias: bpy.props.EnumProperty(
        name="Bias",
        description="Bias added to the result of this computation, before scaling & clamping",
        items=(
            ('ZERO', "0", ""),
            ('HALF', "0.5", ""),
            ('NEGATIVE_HALF', "-0.5", ""),
        ),
        default='ZERO',
        options=set()
    )

    compOp: bpy.props.EnumProperty(
        name="Operator",
        description="Operator used to compare inputs A and B",
        items=(
            ('ADD_OR_GREATER', ">", ""),
            ('SUBTRACT_OR_EQUALS', "==", ""),
        ),
        default='ADD_OR_GREATER',
        options=set()
    )

    output: bpy.props.EnumProperty(
        name="Output",
        description="Destination for storing this computation's result",
        items=(
            ('0', "Standard Color Slot 1", ""),
            ('1', "Standard Color Slot 2", ""),
            ('2', "Standard Color Slot 3", ""),
            ('3', "Standard Color Slot 4", ""),
        ),
        default='0',
        options=set()
    )

    clamp: bpy.props.BoolProperty(
        name="Clamp",
        description="Clamp the result of this computation between 0 and 1",
        default=True,
        options=set()
    )


class TevStageColorParams(TevStageCalcParams):

    def argItems(self, context: bpy.types.Context):
        from .. import ICONS # pylint: disable=import-outside-toplevel
        try:
            matSettings = context.material.brres
            tevSettings = context.scene.brres.tevConfigs[matSettings.tevID]
            texIcon = tevSettings.stages.activeItem().texIcon(context)
        except AttributeError:
            texIcon = 'TEXTURE'
        return (
            ('STANDARD_0_COLOR', "Standard Slot 1 RGB", "RGB from this material's first standard color slot", 'RESTRICT_COLOR_ON', 0), # pylint: disable=line-too-long
            ('STANDARD_0_ALPHA', "Standard Slot 1 Alpha", "Alpha from this material's first standard color slot", 'RESTRICT_COLOR_ON', 1), # pylint: disable=line-too-long
            ('STANDARD_1_COLOR', "Standard Slot 2 RGB", "RGB from this material's second standard color slot", 'RESTRICT_COLOR_ON', 2), # pylint: disable=line-too-long
            ('STANDARD_1_ALPHA', "Standard Slot 2 Alpha", "Alpha from this material's second standard color slot", 'RESTRICT_COLOR_ON', 3), # pylint: disable=line-too-long
            ('STANDARD_2_COLOR', "Standard Slot 3 RGB", "RGB from this material's third standard color slot", 'RESTRICT_COLOR_ON', 4), # pylint: disable=line-too-long
            ('STANDARD_2_ALPHA', "Standard Slot 3 Alpha", "Alpha from this material's third standard color slot", 'RESTRICT_COLOR_ON', 5), # pylint: disable=line-too-long
            ('STANDARD_3_COLOR', "Standard Slot 4 RGB", "RGB from this material's fourth standard color slot", 'RESTRICT_COLOR_ON', 6), # pylint: disable=line-too-long
            ('STANDARD_3_ALPHA', "Standard Slot 4 Alpha", "Alpha from this material's fourth standard color slot", 'RESTRICT_COLOR_ON', 7), # pylint: disable=line-too-long
            ('CONSTANT', "Constant RGB", "Value from this computation's Constant Color selection", 'RESTRICT_COLOR_OFF', 14), # pylint: disable=line-too-long
            ('TEX_COLOR', "Texture RGB", "RGB from this stage's texture selection", texIcon, 8),
            ('TEX_ALPHA', "Texture Alpha", "Alpha from this stage's texture selection", texIcon, 9),
            ('RASTER_COLOR', "Raster RGB", "RGB from this stage's raster selection", 'LIGHT', 10), # pylint: disable=line-too-long
            ('RASTER_ALPHA', "Raster Alpha", "Alpha from this stage's raster selection", 'LIGHT', 11), # pylint: disable=line-too-long
            ('ONE', "1", "All channels set to 1", ICONS['VAL_8_8'].icon_id, 12),
            ('HALF', ".5", "All channels set to .5", ICONS['VAL_4_8'].icon_id, 13),
            ('ZERO', "0", "All channels set to 0", ICONS['VAL_0_8'].icon_id, 15),
        )

    a: bpy.props.EnumProperty(
        name="Input A", description="Input A", items=argItems, default=15, options=set()
    )
    b: bpy.props.EnumProperty(
        name="Input B", description="Input B", items=argItems, default=15, options=set()
    )
    c: bpy.props.EnumProperty(
        name="Input C", description="Input C", items=argItems, default=15, options=set()
    )
    d: bpy.props.EnumProperty(
        name="Input D", description="Input D", items=argItems, default=15, options=set()
    )

    args = alias("a", "b", "c", "d")

    compChan: bpy.props.EnumProperty(
        name="Subject",
        description="Basis of comparison",
        items=(
            ('ONE_OR_R', "R", "Compare red"),
            ('TWO_OR_RG', "RG", "Compare (red + 255 * green)"),
            ('FOUR_OR_RGB', "RGB", "Compare (red + 255 * green + 65,025 * blue)"),
            ('HALF_OR_BY_CHANNEL_OR_A', "By Channel", "Compare ecah channel individually"),
        ),
        default='HALF_OR_BY_CHANNEL_OR_A',
        options=set()
    )


class TevStageAlphaParams(TevStageCalcParams):

    def argItems(self, context: bpy.types.Context):
        from .. import ICONS # pylint: disable=import-outside-toplevel
        try:
            matSettings = context.material.brres
            tevSettings = context.scene.brres.tevConfigs[matSettings.tevID]
            texIcon = tevSettings.stages.activeItem().texIcon(context)
        except AttributeError:
            texIcon = 'TEXTURE'
        return (
            ('STANDARD_0_ALPHA', "Standard Slot 1 Alpha", "Alpha from this stage's first standard color slot", 'RESTRICT_COLOR_ON', 0), # pylint: disable=line-too-long
            ('STANDARD_1_ALPHA', "Standard Slot 2 Alpha", "Alpha from this stage's second standard color slot", 'RESTRICT_COLOR_ON', 1), # pylint: disable=line-too-long
            ('STANDARD_2_ALPHA', "Standard Slot 3 Alpha", "Alpha from this stage's third standard color slot", 'RESTRICT_COLOR_ON', 2), # pylint: disable=line-too-long
            ('STANDARD_3_ALPHA', "Standard Slot 4 Alpha", "Alpha from this stage's fourth standard color slot", 'RESTRICT_COLOR_ON', 3), # pylint: disable=line-too-long
            ('TEX_ALPHA', "Texture Alpha", "Alpha from this stage's texture selection", texIcon, 4),
            ('RASTER_ALPHA', "Raster Alpha", "Alpha from this stage's raster selection", 'LIGHT', 5), # pylint: disable=line-too-long
            ('CONSTANT', "Constant Alpha", "Value from this computation's Constant Alpha selection", 'RESTRICT_COLOR_OFF', 6), # pylint: disable=line-too-long
            ('ZERO', "0", "Alpha set to 0", ICONS['VAL_0_8'].icon_id, 7),
        )

    a: bpy.props.EnumProperty(
        name="Input A", description="Input A", items=argItems, default=7, options=set()
    )
    b: bpy.props.EnumProperty(
        name="Input B", description="Input B", items=argItems, default=7, options=set()
    )
    c: bpy.props.EnumProperty(
        name="Input C", description="Input C", items=argItems, default=7, options=set()
    )
    d: bpy.props.EnumProperty(
        name="Input D", description="Input D", items=argItems, default=7, options=set()
    )

    args = alias("a", "b", "c", "d")

    compChan: bpy.props.EnumProperty(
        name="Subject",
        description="Basis of comparison",
        items=(
            ('ONE_OR_R', "R", "Compare red"),
            ('TWO_OR_RG', "RG", "Compare (red + 255 * green)"),
            ('FOUR_OR_RGB', "RGB", "Compare (red + 255 * green + 65,025 * blue)"),
            ('HALF_OR_BY_CHANNEL_OR_A', "A", "Compare alpha"),
        ),
        default='HALF_OR_BY_CHANNEL_OR_A',
        options=set()
    )


class TevStageSettings(CustomIDPropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        return {f"{t.name}/{s.name}": s for t in context.scene.brres.tevConfigs for s in t.stages}

    @classmethod
    def defaultName(cls):
        return "Stage"

    def drawListItem(self: bpy.types.UIList, context: bpy.types.Context, layout: bpy.types.UILayout,
                     data, item, icon: int, active_data, active_property: str, index=0, flt_flag=0):
        # pylint: disable=attribute-defined-outside-init
        self.use_filter_show = False
        row = layout.row(align=True)
        drawIcon(row, item.texIcon(context))
        # drawIcon(row, item.indIcon(context))
        row.prop(item, "name", text="", emboss=False)
        hideIcon = 'HIDE_ON' if item.hide else 'HIDE_OFF'
        row.prop(item, "hide", text="", icon=hideIcon, emboss=False)

    def texIcon(self, context: bpy.types.Context):
        """Texture icon ID for this stage. Varies based on active material."""
        try:
            texImg = context.material.brres.textures[self.sels.texSlot - 1].activeImg
            texImg.preview_ensure()
            return texImg.preview.icon_id
        except (AttributeError, IndexError):
            return 'TEXTURE'

    def indIcon(self, context: bpy.types.Context):
        """Indirect texture icon ID for this stage. Varies based on active material."""
        try:
            tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
            texIdx = tevSettings.indTexSlots[self.indSettings.slot - 1] - 1
            texImg = context.material.brres.textures[texIdx].activeImg
            texImg.preview_ensure()
            return texImg.preview.icon_id
        except (IndexError, AttributeError):
            return 'TEXTURE'

    hide: bpy.props.BoolProperty(
        name="Hide Stage",
        description="Disable stage in the TEV process",
        default=False,
        options=set()
    )

    sels: bpy.props.PointerProperty(type=TevStageSelSettings)
    indSettings: bpy.props.PointerProperty(type=TevStageIndSettings)
    colorParams: bpy.props.PointerProperty(type=TevStageColorParams)
    alphaParams: bpy.props.PointerProperty(type=TevStageAlphaParams)


class ColorSwapSettings(bpy.types.PropertyGroup):

    items = (('R', "R", "", 'COLORSET_01_VEC', 0),
             ('G', "G", "", 'COLORSET_03_VEC', 1),
             ('B', "B", "", 'COLORSET_04_VEC', 2),
             ('A', "A", "", 'COLORSET_13_VEC', 3))

    r: bpy.props.EnumProperty(name="R", description="Red source", items=items, default='R', options=set()) # pylint: disable=line-too-long
    g: bpy.props.EnumProperty(name="G", description="Green source", items=items, default='G', options=set()) # pylint: disable=line-too-long
    b: bpy.props.EnumProperty(name="B", description="Blue source", items=items, default='B', options=set()) # pylint: disable=line-too-long
    a: bpy.props.EnumProperty(name="A", description="Alpha source", items=items, default='A', options=set()) # pylint: disable=line-too-long


class TevSettings(CustomIDPropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        return {f"{s.name}/{t.name}": t for s in bpy.data.scenes for t in s.brres.tevConfigs}

    @classmethod
    def defaultName(cls):
        return "TEV Config"

    def initialize(self):
        super().initialize()
        # set up default color swap settings
        for i in range(gx.MAX_COLOR_SWAPS):
            swapSettings = self.colorSwaps.add()
            if i > 0:
                swapSettings.r = swapSettings.g = swapSettings.b = gx.ColorChannel(i - 1).name
        # add a stage
        self.stages.add()

    stages: TevStageSettings.CustomIDCollectionProperty(gx.MAX_TEV_STAGES, 1)

    colorSwaps: bpy.props.CollectionProperty(type=ColorSwapSettings)

    indTexSlots: bpy.props.IntVectorProperty(
        name="Texture Slots",
        description="Material textures to use for each indirect texture",
        size=gx.MAX_INDIRECTS,
        min=1,
        max=gx.MAX_TEXTURES,
        default=[gx.MAX_TEXTURES] * gx.MAX_INDIRECTS,
        options=set()
    )


class TevPanel(PropertyPanel):
    bl_idname = "BRRES_PT_tev"
    bl_label = "TEV Configuration"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context: bpy.types.Context):
        matSettings = context.material.brres
        layout = self.layout
        context.scene.brres.tevConfigs.drawAccessor(layout, matSettings, "tevID")


class TevSubPanel(PropertyPanel):
    bl_parent_id = "BRRES_PT_tev"
    @classmethod
    def poll(cls, context: bpy.types.Context):
        try:
            tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
            return True
        except KeyError:
            return False


class TevColorSwapPanel(TevSubPanel):
    bl_idname = "BRRES_PT_tev_color_swap"
    bl_label = "Color Swap Table"
    bl_parent_id = "BRRES_PT_tev"

    def draw(self, context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        for i in range(gx.MAX_COLOR_SWAPS):
            row = layout.row().split(factor=.15)
            labelCol = row.column()
            labelCol.alignment = 'RIGHT'
            labelCol.label(text=f"Slot {i + 1}")
            for chan in "rgba":
                row.prop(tevSettings.colorSwaps[i], chan, text="")


class TevIndSelPanel(TevSubPanel):
    bl_idname = "BRRES_PT_tev_ind_sel"
    bl_label = "Indirect Texture Selections"

    def draw(self, context):
        layout = self.layout
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        textures = context.material.brres.textures
        for i, slot in enumerate(tevSettings.indTexSlots):
            icon = 'TEXTURE'
            try:
                img = textures[slot - 1].activeImg
                img.preview_ensure()
                icon = img.preview.icon_id
            except (IndexError, AttributeError):
                pass
            label = f"Indirect Source {i + 1}"
            drawProp(layout, tevSettings, "indTexSlots", factor=.35, text=label, icon=icon, index=i)


class TevStagePanel(TevSubPanel):
    bl_idname = "BRRES_PT_tev_stage"
    bl_label = "Stages"

    def draw(self, context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        layout = self.layout
        tevSettings.stages.drawList(layout)


class TevStageSubPanel(PropertyPanel):
    bl_parent_id = "BRRES_PT_tev_stage"
    @classmethod
    def poll(cls, context: bpy.types.Context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        try:
            tevSettings.stages.activeItem()
            return True
        except IndexError:
            return False


class TevStageSelPanel(TevStageSubPanel):
    bl_idname = "BRRES_PT_tev_stage_sel"
    bl_label = "Selections"

    def draw(self, context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        stageSettings = tevSettings.stages.activeItem()
        sels = stageSettings.sels
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        drawProp(layout, sels, "texSlot", icon=stageSettings.texIcon(context))
        texSwapRow = layout.row()
        texSwapRow.prop(sels, "texSwapSlot")
        texSwap = tevSettings.colorSwaps[sels.texSwapSlot - 1]
        texSwapRow.separator()
        for c in "rgba":
            texSwapRow.label(icon_value=texSwapRow.enum_item_icon(texSwap, c, getattr(texSwap, c)))
        layout.prop(sels, "rasterSel")
        rasSwapRow = layout.row()
        rasSwapRow.prop(sels, "rasSwapSlot")
        rasSwap = tevSettings.colorSwaps[sels.rasSwapSlot - 1]
        rasSwapRow.separator()
        for c in "rgba":
            rasSwapRow.label(icon_value=rasSwapRow.enum_item_icon(rasSwap, c, getattr(rasSwap, c)))


class TevStageIndPanel(TevStageSubPanel):
    bl_idname = "BRRES_PT_tev_stage_ind"
    bl_label = "Indirect Texturing"

    def draw(self, context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        stageSettings = tevSettings.stages.activeItem()
        indSettings = stageSettings.indSettings
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(indSettings, "enable")
        layout = layout.column()
        layout.enabled = indSettings.enable
        drawProp(layout, indSettings, "slot", icon=stageSettings.indIcon(context))
        layout.separator(factor=UI_COL_SEP)
        layout.row().prop(indSettings, "fmt", expand=True)
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "enableBias", index=0, text="Alpha Bias")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "enableBias", index=1, text="Blue Bias")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "enableBias", index=2, text="Green Bias")
        layout.separator(factor=UI_COL_SEP)
        layout.row().prop(indSettings, "bumpAlphaComp", expand=True)
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "mtxType")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "mtxSlot")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "wrapU")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "wrapV")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "utcLOD")
        layout.separator(factor=UI_COL_SEP)
        layout.prop(indSettings, "addPrev")


def drawTevStageCalcParams(layout: bpy.types.UILayout, params: TevStageCalcParams):
    """Draw the color/alpha calculation parameters for one TEV stage."""
    layout.prop(params, "a")
    layout.prop(params, "b")
    layout.prop(params, "c")
    layout.prop(params, "d")
    layout.prop(params, "compMode")
    if params.compMode:
        layout.prop(params, "compOp", expand=True)
        layout.prop(params, "compChan")
    else:
        layout.prop(params, "op", expand=True)
        layout.prop(params, "scale", expand=True)
        layout.prop(params, "bias", expand=True)
    layout.prop(params, "output")
    layout.prop(params, "clamp")
    labelRow = layout.box().row()
    labelRow.alignment = 'CENTER'
    label = ""
    if params.compMode:
        opLbl = layout.enum_item_name(params, "compOp", params.compOp)
        chanLbl = layout.enum_item_name(params, "compChan", params.compChan)
        label = f"D + (A {opLbl} B {chanLbl} ? C : 0)"
    else:
        scaleLbl = layout.enum_item_name(params, "scale", params.scale)
        biasLbl = layout.enum_item_name(params, "bias", params.bias)
        opLbl = layout.enum_item_name(params, "op", params.op)
        label = f"{scaleLbl} * ({biasLbl} + D {opLbl} lerp(A, B, C))"
    clampedLabel = f"Clamp({label})" if params.clamp else label
    labelRow.label(text=f"Output = {clampedLabel}")


class TevStageColorPanel(TevStageSubPanel):
    bl_idname = "BRRES_PT_tev_stage_color"
    bl_label = "Color"

    def draw(self, context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        stageSettings = tevSettings.stages.activeItem()
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(stageSettings.sels, "constColor")
        drawTevStageCalcParams(layout, stageSettings.colorParams)


class TevStageAlphaPanel(TevStageSubPanel):
    bl_idname = "BRRES_PT_tev_stage_alpha"
    bl_label = "Alpha"

    def draw(self, context):
        tevSettings = context.scene.brres.tevConfigs[context.material.brres.tevID]
        stageSettings = tevSettings.stages.activeItem()
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(stageSettings.sels, "constAlpha")
        drawTevStageCalcParams(layout, stageSettings.alphaParams)
