# 3rd party imports
import bpy
# internal imports
from .tev import TevSettings


class SceneSettings(bpy.types.PropertyGroup):

    # the version getter & setter setup here lets us use a default version retrieved from bl_info
    # (we have to import in a function to prevent circular dependency)
    # (we can't store the version here and then import it when creating bl_info because bl_info
    # is parsed manually by blender or something and can't use variables for its fields)

    def _getVer(self):
        if self.version_ == (0, 0, 0):
            from .. import bl_info # pylint: disable=import-outside-toplevel
            self.version_ = bl_info["version"]
        return self.version_

    def _setVer(self, v: tuple[int, int, int]):
        self.version_ = v

    tevConfigs: TevSettings.CustomIDCollectionProperty()
    version_: bpy.props.IntVectorProperty(default=(0, 0, 0))
    version: bpy.props.IntVectorProperty(get=_getVer, set=_setVer)
