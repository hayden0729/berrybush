# 3rd party imports
import bpy
# internal imports
from .common import PropertyPanel, UNDOCUMENTED


class BoneSettings(bpy.types.PropertyGroup):

    bbMode: bpy.props.EnumProperty(
        name="Billboard Mode",
        description="Mode of moving this bone based on the camera",
        items=(
            ('OFF', "None", ""),
            ('STANDARD', "Standard", "No rotation restrictions, affected by parent rotation, Z axis parallel to camera's Z axis"), # pylint: disable=line-too-long
            ('STANDARD_PERSP', "Standard (Perspective)", "No rotation restrictions, affected by parent rotation, Z axis pointing at camera"), # pylint: disable=line-too-long
            ('ROTATION', "Rotation", "Y axis parallel to camera's Y axis, unaffected by parent rotation, Z axis parallel to camera's Z axis"), # pylint: disable=line-too-long
            ('ROTATION_PERSP', "Rotation (Perspective)", "Y axis parallel to camera's Y axis, unaffected by parent rotation, Z axis pointing at camera"), # pylint: disable=line-too-long
            ('Y_ROTATION', "Y Rotation", "Only Y rotation allowed, affected by parent rotation, Z axis parallel to camera's Z axis"), # pylint: disable=line-too-long
            ('Y_ROTATION_PERSP', "Y Rotation (Perspective)", "Only Y rotation allowed, affected by parent rotation, Z axis pointing at camera"), # pylint: disable=line-too-long
        ),
        default='OFF',
        options=set()
    )

    bbParent: bpy.props.StringProperty(
        name="Billboard Parent",
        description=UNDOCUMENTED,
        options=set()
    )


class BonePanel(PropertyPanel):
    bl_idname = "BRRES_PT_bone_attrs"
    bl_label = "BRRES Settings"
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        return (context.bone or context.edit_bone)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        bone = context.bone if context.bone else context.armature.bones[context.edit_bone.name]
        boneSettings = bone.brres
        col = layout.column()
        col.prop(boneSettings, "bbMode")
        col.prop_search(boneSettings, "bbParent", context.armature, "bones")
