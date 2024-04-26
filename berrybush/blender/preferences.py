# 3rd party imports
import bpy
import bpy.utils.previews
# internal imports
from .common import drawColumnSeparator, drawCheckedProp


ADDON_IDNAME = __name__.split(".", maxsplit=1)[0]


class BerryBushPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_IDNAME

    doBackups: bpy.props.BoolProperty(
        name="Back Up Overwritten Files",
        description="Create backups when BRRES files are overwritten during export",
        default=True
    )

    backupDir: bpy.props.StringProperty(
        name="Backup Directory",
        description="Directory for backing up overwritten BRRES files",
        subtype='FILE_PATH',
        default=bpy.utils.user_resource('SCRIPTS', path="berrybush_backups")
    )

    doMaxBackups: bpy.props.BoolProperty(
        name="Backup Limit",
        description="Whether to automatically delete old backups once the backup directory reaches a certain capacity", # pylint:disable=line-too-long
        default=True
    )

    maxBackups: bpy.props.IntProperty(
        name="Backup Limit",
        description="Maximum number of backups that can be stored before old ones are automatically deleted", # pylint:disable=line-too-long
        min=1,
        default=100
    )

    doUpdateChecks: bpy.props.BoolProperty(
        name="Check For Updates",
        default=True
    )

    skipThisVersion: bpy.props.BoolProperty(
        name="Skip This Version"
    )

    latestKnownVersion: bpy.props.IntVectorProperty(size=3)

    def draw(self, context):
        # backup stuff
        self.layout.prop(self, "doBackups")
        col = self.layout.column()
        col.enabled = self.doBackups
        col.prop(self, "backupDir")
        drawColumnSeparator(col)
        drawCheckedProp(col, self, "doMaxBackups", self, "maxBackups")
        self.layout.operator("brres.clear_backups")
        # update stuff
        row = self.layout.row().split(factor=.5)
        row.prop(self, "doUpdateChecks")
        skipRow = row.row()
        skipRow.enabled = self.doUpdateChecks
        skipRow.prop(self, "skipThisVersion")


def getPrefs(context: bpy.types.Context) -> BerryBushPreferences:
    """Get the BerryBush preferences from a Blender context."""
    return context.preferences.addons[ADDON_IDNAME].preferences
