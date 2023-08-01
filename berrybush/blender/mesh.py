# 3rd party imports
import bpy
# internal imports
from .common import PropertyPanel, drawCheckedProp, drawProp
from .proputils import CloneablePropertyGroup
from ..wii import gx
from ..wii.alias import alias


class MeshAttrSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        return {m.name: m.brres.meshAttrs for m in bpy.data.meshes}

    _desc = "Attribute layer or UV map to use for this BRRES attribute"

    clr1: bpy.props.StringProperty(name="Slot 1", description=_desc, options=set())
    clr2: bpy.props.StringProperty(name="Slot 2", description=_desc, options=set())
    uv1: bpy.props.StringProperty(name="Slot 1", description=_desc, options=set(), default="UVMap")
    uv2: bpy.props.StringProperty(name="Slot 2", description=_desc, options=set())
    uv3: bpy.props.StringProperty(name="Slot 3", description=_desc, options=set())
    uv4: bpy.props.StringProperty(name="Slot 4", description=_desc, options=set())
    uv5: bpy.props.StringProperty(name="Slot 5", description=_desc, options=set())
    uv6: bpy.props.StringProperty(name="Slot 6", description=_desc, options=set())
    uv7: bpy.props.StringProperty(name="Slot 7", description=_desc, options=set())
    uv8: bpy.props.StringProperty(name="Slot 8", description=_desc, options=set())

    clrs = alias("clr1", "clr2")
    uvs = alias("uv1", "uv2", "uv3", "uv4", "uv5", "uv6", "uv7", "uv8")


class MeshSettings(CloneablePropertyGroup):

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context):
        return {m.name: m.brres for m in bpy.data.meshes}

    meshAttrs: bpy.props.PointerProperty(type=MeshAttrSettings)

    enableDrawPrio: bpy.props.BoolProperty(
        name="Enable Draw Priority",
        description="Give this object a special priority during rendering (if disabled, object has minimum priority)", # pylint: disable=line-too-long
        default=False,
        options=set()
    )

    drawPrio: bpy.props.IntProperty(
        name="Draw Priority",
        description="Priority for rendering this object (higher -> draw later than lower-priority others in the same render group)", # pylint: disable=line-too-long
        min=1,
        max=255,
        default=1,
        options=set()
    )


class MeshPanel(PropertyPanel):
    bl_idname = "BRRES_PT_mesh_attrs"
    bl_label = "BRRES Settings"
    bl_context = "data"
    bl_options = set()

    @classmethod
    def poll(cls, context):
        return context.mesh is not None

    def draw(self, context):
        layout = self.layout
        meshSettings = context.active_object.data.brres
        meshAttrs = meshSettings.meshAttrs
        meshSettings.drawCloneUI(layout)
        labels = ("Color Attributes", "UV Maps")
        icons = ('GROUP_VCOL', 'GROUP_UVS')
        propNames = ("clr", "uv")
        propCounts = (gx.MAX_CLR_ATTRS, gx.MAX_UV_ATTRS)
        for label, icon, propName, propCount in zip(labels, icons, propNames, propCounts):
            layout.label(text=label, icon=icon)
            for i in range(1, propCount + 1):
                drawProp(layout, meshAttrs, f"{propName}{i}", factor=.2)
        layout.label(text="Other Settings", icon='SETTINGS')
        drawCheckedProp(layout, meshSettings, "enableDrawPrio", meshSettings, "drawPrio", .34)
