# standard imports
from itertools import chain
# 3rd party imports
import bpy


class PreviewAnimation(bpy.types.Operator):
    """Preview an animation by soloing NLA tracks by name"""

    bl_idname = "brres.preview_anim"
    bl_label = "Preview Animation"

    name: bpy.props.StringProperty(
        name="Name",
        description="All NLA tracks with this name will be soloed, and any others will be un-soloed", # pylint: disable=line-too-long
    )

    def execute(self, context: bpy.types.Context):
        for obj in chain(bpy.data.objects.values(), bpy.data.materials.values()):
            obj: bpy.types.Object | bpy.types.Material
            if obj.animation_data is not None:
                for track in obj.animation_data.nla_tracks:
                    track: bpy.types.NlaTrack
                    track.is_solo = track.name == self.name
        return {'FINISHED'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "name")


def drawOp(self, context: bpy.types.Context):
    layout: bpy.types.UILayout = self.layout
    layout.separator()
    layout.operator(PreviewAnimation.bl_idname, text="BerryBush: Preview Animation")
