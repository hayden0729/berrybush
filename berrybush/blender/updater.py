# 3rd party imports
import bpy


@bpy.app.handlers.persistent
def update(_):
    """Update the active Blend file to a new BerryBush version."""
    from .. import bl_info # pylint: disable=import-outside-toplevel
    latestVer = bl_info["version"]
    currentVer = tuple(bpy.data.scenes[0].brres.version)
    if currentVer != latestVer:
        if currentVer == (1, 0, 0):
            pass
        for scene in bpy.data.scenes:
            scene.brres.version = latestVer
