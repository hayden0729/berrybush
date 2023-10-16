# 3rd party imports
import bpy
# internal imports
from .tev import TevSettings


class SceneSettings(bpy.types.PropertyGroup):
    tevConfigs: TevSettings.CustomIDCollectionProperty()
    renderIgnoreBG: bpy.props.BoolProperty(
        name="Ignore Background",
        description="For drawing operations that reference existing framebuffer colors (e.g., blending), use the source color instead when nothing has been written yet. Only affects transparent renders (not viewport)", # pylint: disable=line-too-long
        default=True
    )
    # version default of (0, 0, 0) just behaves as "most recent" & gets updated on blendfile save
    # this is done because we want the default to be the most recent version, but can't do that
    # directly because:
    # 1) we can't import it from bl_info here because that would make a circular dependency
    # 2) we can't store the version here and then import it when creating bl_info because bl_info
    # is parsed manually by blender or something and seemingly can't use variables for its fields
    version: bpy.props.IntVectorProperty(default=(0, 0, 0))
