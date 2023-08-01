# 3rd party imports
import bpy
import numpy as np
# internal imports
from .common import UNDOCUMENTED, UI_COL_SEP, PropertyPanel, drawCheckedProp, drawProp, filterMats
from .proputils import CustomIDPropertyGroup
from .verify import WarningSuppressionProperty, drawWarningUI
from ..wii import gx


class TextureTransform(bpy.types.PropertyGroup):

    translation: bpy.props.FloatVectorProperty(
        name="Translation",
        subtype='XYZ',
        size=2,
    )

    rotation: bpy.props.FloatProperty(
        name="Rotation",
        subtype='ANGLE',
    )

    scale: bpy.props.FloatVectorProperty(
        name="Scale",
        subtype='XYZ',
        size=2,
        default=(1, 1),
    )


class TexImg(CustomIDPropertyGroup):

    img: bpy.props.PointerProperty(type=bpy.types.Image)

    def drawListItem(self: bpy.types.UIList, context: bpy.types.Context, layout: bpy.types.UILayout,
                     data, item, icon: int, active_data, active_property: str, index=0, flt_flag=0):
        # pylint: disable=attribute-defined-outside-init
        self.use_filter_show = False
        img: bpy.types.Image = item.img
        try:
            img.preview_ensure()
            layout.label(text=img.name, icon_value=img.preview.icon_id)
        except AttributeError:
            # text is space rather than empty to keep active indicator right-aligned when used
            layout.label(text=" ", icon='IMAGE_DATA')
        if index == context.material.brres.textures.activeItem().activeImgSlot - 1:
            # this is the active image, so draw a little indicator
            layout.label(icon='RESTRICT_RENDER_OFF')


class TexSettings(CustomIDPropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        mats = filterMats(includeGreasePencil=False)
        return {f"{m.name}/{t.name}": t for m in mats for t in m.brres.textures}

    @classmethod
    def defaultName(cls):
        return "Texture"

    def drawListItem(self: bpy.types.UIList, context: bpy.types.Context, layout: bpy.types.UILayout,
                     data, item, icon: int, active_data, active_property: str, index=0, flt_flag=0):
        # pylint: disable=attribute-defined-outside-init
        self.use_filter_show = False
        img: bpy.types.Image = item.activeImg
        try:
            img.preview_ensure()
            layout.prop(item, "name", text="", emboss=False, icon_value=img.preview.icon_id)
        except AttributeError:
            layout.prop(item, "name", text="", emboss=False, icon='TEXTURE')

    activeImgSlot: bpy.props.IntProperty(
        name="Active Slot",
        description="Current image to use for this texture",
        min=1,
        default=1
    )

    imgs: TexImg.CustomIDCollectionProperty()

    uiImgTab: bpy.props.EnumProperty(
        name="Image UI Tab",
        items=(
            ('IMG', "Active Image", "Settings for the active image"),
            ('SLOTS', "Animation Slots", "Image slots for PAT animations"),
        ),
        default='IMG',
        options=set()
    )

    def _addImgFromAdder(self, context: bpy.types.Context):
        """Add a new image slot to this texture from the adder property.

        Make this new image the active one, and reset the adder.
        If the adder is None, do nothing.
        """
        newImg = self.imgAdder
        if newImg:
            self.imgs.add().img = newImg
            self.activeImgSlot = len(self.imgs)
            self.imgAdder = None

    imgAdder: bpy.props.PointerProperty(
        type=bpy.types.Image,
        update=_addImgFromAdder,
        options=set()
    )

    @property
    def activeImg(self) -> bpy.types.Image:
        try:
            return self.imgs[self.activeImgSlot - 1].img
        except IndexError:
            return None

    transform: bpy.props.PointerProperty(type=TextureTransform)

    def coordSrcItems(self, context: bpy.types.Context):
        """Items for the coordinate source enum."""
        uvNames = [f"UV {i}" for i in range(1, gx.MAX_UV_ATTRS + 1)]
        try:
            activeData = context.active_object.data
            if isinstance(activeData, bpy.types.Mesh):
                attrs = activeData.brres.meshAttrs
                uvNames[:] = [f"{n} ({a})" if a else n for n, a in zip(uvNames, attrs.uvs)]
        except AttributeError:
            pass # no active object
        return (
            ('UV_1', uvNames[0], "Coordinates from the mesh's 1st UV slot", 'GROUP_UVS', 0),
            ('UV_2', uvNames[1], "Coordinates from the mesh's 2nd UV slot", 'GROUP_UVS', 1),
            ('UV_3', uvNames[2], "Coordinates from the mesh's 3rd UV slot", 'GROUP_UVS', 2),
            ('UV_4', uvNames[3], "Coordinates from the mesh's 4th UV slot", 'GROUP_UVS', 3),
            ('UV_5', uvNames[4], "Coordinates from the mesh's 5th UV slot", 'GROUP_UVS', 4),
            ('UV_6', uvNames[5], "Coordinates from the mesh's 6th UV slot", 'GROUP_UVS', 5),
            ('UV_7', uvNames[6], "Coordinates from the mesh's 7th UV slot", 'GROUP_UVS', 6),
            ('UV_8', uvNames[7], "Coordinates from the mesh's 8th UV slot", 'GROUP_UVS', 7),
            ('PROJECTION', "Projection Map", "Coordinates based on positions & camera", 'VIEW_CAMERA', 8), # pylint: disable=line-too-long
            ('ENV_CAM', "Environment Map (Camera)", "Coordinates based on normals & camera", 'VIEW_CAMERA', 9), # pylint: disable=line-too-long
            ('ENV_LIGHT', "Environment Map (Light)", "Coordinates based on normals & lights (rendered the same as camera mode in BerryBush)", 'LIGHT', 10), # pylint: disable=line-too-long
            ('ENV_SPEC', "Environment Map (Specular)", "Coordinates based on normals & lights (rendered the same as camera mode in BerryBush)", 'LIGHT', 11), # pylint: disable=line-too-long
        )

    mapMode: bpy.props.EnumProperty(
        name="Coordinate Source",
        description="Texture mapping method",
        items=coordSrcItems,
        default=0,
        options=set()
    )

    wrapModeItems = (
         ('CLAMP', "Clamp", ""),
         ('REPEAT', "Repeat", ""),
         ('MIRROR', "Mirror", ""),
     )

    wrapModeU: bpy.props.EnumProperty(
        name="U Wrap",
        description="Out of bounds extrapolation behavior (U axis)",
        items=wrapModeItems,
        default='CLAMP',
        options=set()
    )

    wrapModeV: bpy.props.EnumProperty(
        name="V Wrap",
        description="Out of bounds extrapolation behavior (V axis)",
        items=wrapModeItems,
        default='CLAMP',
        options=set()
    )

    minFilter: bpy.props.EnumProperty(
        name="Min Filter",
        description="Filter applied for image downscaling (minification)",
        items=(
            ('NEAREST', "Nearest", "Nearest neighbor filtering (sharper)"),
            ('LINEAR', "Linear", "Bilinear filtering (smoother)"),
        ),
        default='LINEAR',
        options=set()
    )

    magFilter: bpy.props.EnumProperty(
        name="Mag Filter",
        description="Filter applied for image upscaling (magnification)",
        items=(
            ('NEAREST', "Nearest", "Nearest neighbor filtering (sharper)"),
            ('LINEAR', "Linear", "Bilinear filtering (smoother)"),
        ),
        default='LINEAR',
        options=set()
    )

    mipFilter: bpy.props.EnumProperty(
        name="Mip Filter",
        description="Filter applied between different mipmaps",
        items=(
            ('NEAREST', "Nearest", "Nearest neighbor filtering (sharper)"),
            ('LINEAR', "Linear", "Bilinear filtering (smoother)"),
        ),
        default='LINEAR',
        options=set()
    )

    lodBias: bpy.props.FloatProperty(
        name="LOD Bias",
        description="Bias added to the calculation that determines which mipmap to use",
        options=set()
    )

    maxAnisotropy: bpy.props.EnumProperty(
        name="Max Anisotropy",
        description="Maximum ratio of anisotropy to be used during filtering (not reflected in BerryBush)", # pylint: disable=line-too-long
        items=(
            ('ONE', "1", ""),
            ('TWO', "2", ""),
            ('FOUR', "4", ""),
        ),
        default='ONE',
        options=set()
    )

    clampBias: bpy.props.BoolProperty(
        name="Clamp Bias",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    texelInterpolate: bpy.props.BoolProperty(
        name="Texel Interpolate",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    useCam: bpy.props.BoolProperty(
        name="Use Camera",
        description=UNDOCUMENTED,
        default=False,
        options=set()
    )

    camSlot: bpy.props.IntProperty(
        name="Camera Slot",
        description=UNDOCUMENTED,
        default=1,
        min=1,
        max=128,
        options=set()
    )

    useLight: bpy.props.BoolProperty(
        name="Use Light",
        description=UNDOCUMENTED,
        default=False,
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


class MipmapSlot(CustomIDPropertyGroup):

    @classmethod
    def defaultName(cls):
        return "Mipmap"

    def drawListItem(self: bpy.types.UIList, context: bpy.types.Context, layout: bpy.types.UILayout,
                     data, item, icon: int, active_data, active_property: str, index=0, flt_flag=0):
        # pylint: disable=attribute-defined-outside-init
        self.use_filter_show = False
        img: bpy.types.Image = item.img
        try:
            img.preview_ensure()
            layout.label(text=img.name, icon_value=img.preview.icon_id)
        except AttributeError:
            layout.label(text="", icon='IMAGE_DATA')

    img: bpy.props.PointerProperty(type=bpy.types.Image)


class ImgSettings(bpy.types.PropertyGroup):

    fmt: bpy.props.EnumProperty(
        name="Format",
        description="Method used to compress this texture",
        items=(
            ('I4', "I4", "4 bits/pixel; intensity (one value for red, green, blue, & alpha) with 4 bits of precision"), # pylint: disable=line-too-long
            ('I8', "I8", "8 bits/pixel; intensity (one value for red, green, blue, & alpha) with 8 bits of precision"), # pylint: disable=line-too-long
            ('IA4', "IA4", "8 bits/pixel; intensity (one value for red, green, & blue) & alpha with 4 bits of precision each"), # pylint: disable=line-too-long
            ('IA8', "IA8", "16 bits/pixel; intensity (one value for red, green, & blue) & alpha with 8 bits of precision each"), # pylint: disable=line-too-long
            ('RGB565', "RGB565", "16 bits/pixel; red, green, & blue with 5/6/5 bits of precision respectively (alpha always 1)"), # pylint: disable=line-too-long
            ('RGB5A3', "RGB5A3", "16 bits/pixel; red, green, blue, & 3-bit alpha with 5 bits of color precision when alpha = 1 or 4 otherwise"), # pylint: disable=line-too-long
            ('RGBA8', "RGBA8", "32 bits/pixel; red, green, blue, & alpha with 8 bits of precision each (best quality, worst size)"), # pylint: disable=line-too-long
            ('CMPR', "CMPR", "4 bits/pixel; red, green, blue, & binary alpha stored using DXT1 compression"), # pylint: disable=line-too-long
        ),
        default='RGB5A3',
        options=set()
    )

    mipmaps: MipmapSlot.CustomIDCollectionProperty()

    warnSupPow2: WarningSuppressionProperty()

    warnSupSize: WarningSuppressionProperty()


class TexPanel(PropertyPanel):
    bl_idname = "BRRES_PT_tex"
    bl_label = "Textures"
    bl_parent_id = "BRRES_PT_mat"

    def draw(self, context: bpy.types.Context):
        matSettings = context.material.brres
        layout = self.layout
        drawProp(layout, matSettings.miscSettings, "texTransformMode", .33)
        matSettings.textures.drawList(layout)
        try:
            matSettings.textures.activeItem().drawCloneUI(layout)
        except IndexError:
            pass


class TexSubPanel(PropertyPanel):
    bl_parent_id = "BRRES_PT_tex"
    @classmethod
    def poll(cls, context: bpy.types.Context):
        matSettings = context.material.brres
        try:
            matSettings.textures.activeItem()
            return True
        except IndexError:
            return False


class TexTransformPanel(TexSubPanel):
    bl_idname = "BRRES_PT_tex_transform"
    bl_label = "Transform"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        matSettings = context.material.brres
        texSettings = matSettings.textures.activeItem()
        transform = texSettings.transform
        layout.prop(transform, "translation")
        layout.prop(transform, "rotation")
        layout.prop(transform, "scale")


class TexSettingsPanel(TexSubPanel):
    bl_idname = "BRRES_PT_tex_settings"
    bl_label = "Settings"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        matSettings = context.material.brres
        texSettings = matSettings.textures.activeItem()
        layout.prop(texSettings, "mapMode")
        layout.prop(texSettings, "wrapModeU", expand=True)
        layout.prop(texSettings, "wrapModeV", expand=True)
        layout.prop(texSettings, "minFilter", expand=True)
        layout.prop(texSettings, "magFilter", expand=True)
        # mipmapping stuff
        col = layout.column()
        activeImg = texSettings.activeImg
        col.enabled = activeImg is not None and len(activeImg.brres.mipmaps) > 0
        col.row().prop(texSettings, "mipFilter", expand=True)
        col.separator(factor=UI_COL_SEP)
        col.prop(texSettings, "lodBias")
        col.separator(factor=UI_COL_SEP)
        col.row().prop(texSettings, "maxAnisotropy", expand=True)
        col.separator(factor=UI_COL_SEP)
        col.prop(texSettings, "clampBias")
        col.separator(factor=UI_COL_SEP)
        col.prop(texSettings, "texelInterpolate")
        # everything else
        drawCheckedProp(layout, texSettings, "useCam", texSettings, "camSlot")
        drawCheckedProp(layout, texSettings, "useLight", texSettings, "lightSlot")


class ImgPanel(TexSubPanel):
    bl_idname = "BRRES_PT_img"
    bl_label = "Image"

    def draw(self, context):
        layout = self.layout
        matSettings = context.material.brres
        texSettings = matSettings.textures.activeItem()
        layout.row().prop(texSettings, "uiImgTab", expand=True)
        activeImg: bpy.types.Image = None
        if texSettings.uiImgTab == 'IMG':
            # show active image & its format (also mipmaps, in mipmap panel)
            try:
                texImg = texSettings.imgs[texSettings.activeImgSlot - 1]
                activeImg = texImg.img
                imgSettings = activeImg.brres
                if activeImg:
                    dims = np.array(activeImg.size, dtype=int)
                    if any(bin(dim).count("1") > 1 for dim in dims):
                        drawWarningUI(
                            layout, imgSettings, "warnSupPow2",
                            "Dimensions aren't both powers of 2"
                        )
                    if np.any(dims > gx.MAX_TEXTURE_SIZE):
                        drawWarningUI(
                            layout, imgSettings, "warnSupSize",
                            f"Dimensions aren't both <= {gx.MAX_TEXTURE_SIZE}"
                        )
                layout.template_ID_preview(texImg, "img", new="image.new", open="image.open")
                if activeImg:
                    imgCol = layout.column()
                    imgCol.use_property_split = True
                    imgCol.use_property_decorate = False
                    imgCol.prop(imgSettings, "fmt")
            except IndexError: # active slot out of bounds
                layout.template_ID_preview(texSettings, "imgAdder",
                                           new="image.new", open="image.open")
        else:
            # show image slot ui
            layout.use_property_split = True
            layout.prop(texSettings, "activeImgSlot")
            texSettings.imgs.drawList(layout)
            try:
                texImg = texSettings.imgs.activeItem()
                layout.template_ID(texImg, "img", new="image.new", open="image.open")
            except IndexError: # texture has no images
                pass


class MipmapPanel(PropertyPanel):
    bl_idname = "BRRES_PT_mipmaps"
    bl_label = "Mipmaps"
    bl_parent_id = "BRRES_PT_img"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        matSettings = context.material.brres
        texSettings = matSettings.textures.activeItem()
        try:
            return texSettings.uiImgTab == 'IMG' and texSettings.activeImg is not None
        except IndexError:
            return False

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        matSettings = context.material.brres
        texSettings = matSettings.textures.activeItem()
        imgSettings = texSettings.activeImg.brres
        mipmaps = imgSettings.mipmaps
        mipmaps.drawList(layout)
        try:
            layout.template_ID(mipmaps.activeItem(), "img", new="image.new", open="image.open")
        except IndexError:
            pass
