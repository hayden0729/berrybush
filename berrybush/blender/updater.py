# 3rd party imports
import bpy
# internal imports
from .common import paragraphLabel, getLayerData, setLayerData


@bpy.app.handlers.persistent
def update(_):
    """Update the active Blend file if it uses an outdated BerryBush version."""
    bpy.ops.brres.update()


@bpy.app.handlers.persistent
def saveVer(_):
    """Before saving, update the BerryBush version for every scene in the Blend file."""
    latestVer = addonVer()
    for scene in bpy.data.scenes:
        scene.brres.version = latestVer


def addonVer() -> tuple[int, int, int]:
    """Current version of the BerryBush addon, respresented as a tuple."""
    # this is all needed for the version setup descriped in scene.py
    from .. import bl_info # pylint: disable=import-outside-toplevel
    return bl_info["version"]


def verStr(ver: tuple[int, int, int]):
    """Get a string representation for an addon version."""
    return ".".join(str(i) for i in ver)


def compareVers(verA: tuple[int, int, int], verB: tuple[int, int, int]):
    """Return -1 if verA < verB, 0 if verA == verB, and 1 if verA > verB."""
    for a, b in zip(verA, verB):
        if a < b:
            return -1
        elif a > b:
            return 1
    return 0


class UpdateBRRES(bpy.types.Operator):
    """Update the active Blend file if it uses an outdated BerryBush version."""

    bl_idname="brres.update"
    bl_label="Update BRRES Settings"

    def execute(self, context: bpy.types.Context):
        # get latest version & blendfile version from scene settings
        sceneSettings = bpy.data.scenes[0].brres
        currentVer = tuple(sceneSettings.version)
        latestVer = addonVer()
        # special 1.0.0 detection (it used a different version system)
        if "version_" in sceneSettings:
            currentVer = (1, 0, 0)
            for scene in bpy.data.scenes:
                del scene.brres["version_"]
        # perform updates
        if currentVer != latestVer and currentVer != (0, 0, 0):
            curVerStr = verStr(currentVer)
            self.report({'INFO'}, f"Loading file saved using BerryBush version {curVerStr}")
            # 1.1.0
            if compareVers(currentVer, (1, 1, 0)) < 0:
                # open vertex color gamma updater
                bpy.ops.brres.update_vert_colors_1_1_0('INVOKE_DEFAULT')
                currentVer = (1, 1, 0)
            # 1.1.4
            if compareVers(currentVer, (1, 2, 0)) < 0:
                # update texture transform names for fcurves
                for mat in bpy.data.materials:
                    for tex in mat.brres.textures:
                        tex.transform.name = tex.name
                    for indTf in mat.brres.indSettings.transforms:
                        indTf.transform.name = indTf.name
            for scene in bpy.data.scenes:
                scene.brres.version = latestVer
        return {'FINISHED'}


class UpdateVertColors1_1_0(bpy.types.Operator):
    """Update all vertex colors in the scene from BerryBush 1.0.0 conventions to 1.1.0."""

    bl_idname = "brres.update_vert_colors_1_1_0"
    bl_label = "Update Vertex Colors for BerryBush 1.1.0"
    bl_options = {'UNDO'}

    def execute(self, context: bpy.types.Context):
        layerData = {}
        for mesh in bpy.data.meshes:
            clrs = mesh.brres.meshAttrs.clrs
            meshLayerData = getLayerData(mesh, clrs, unique=False, doProcessing=False)
            for layer, data, indices in meshLayerData:
                if layer is not None and layer not in layerData:
                    data[:, :3] **= 2.2
                    layerData[layer] = data
        setLayerData(layerData)
        return {'FINISHED'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: bpy.types.Context):
        paragraphLabel(self.layout, "This file was saved using an older version of BerryBush, which had different conventions for importing, exporting, and rendering vertex colors. Click OK to update all BRRES vertex color layers to the new conventions (or click elsewhere to keep them unchanged, which will alter how they're rendered and exported).") # pylint: disable=line-too-long
