# standard imports
import json
import urllib
import urllib.request
# 3rd party imports
import bpy
# internal imports
from .common import paragraphLabel, getLayerData, setLayerData
from .preferences import BerryBushPreferences, getPrefs


class LatestVersionChecker():

    def __init__(self):
        self._hasChecked = False

    def _retrieveLatestRelease(self) -> tuple[tuple[int, int, int], str]:
        """Get the tag & release URL for the latest available version of BerryBush from GitHub."""
        url = "https://api.github.com/repos/hayden0729/berrybush/releases/latest"
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
            return (tuple(int(v) for v in data["tag_name"].split(".")), data["html_url"])

    def check(self, currentVer: tuple[int, int, int], prefs: BerryBushPreferences):
        """Display a popup if a newer version of BerryBush than the installed one is available.
        
        (If this has already been done, do nothing)
        """
        if prefs.doUpdateChecks and not self._hasChecked:
            self._hasChecked = True
            try:
                latestVer, url = self._retrieveLatestRelease()
            except:
                print("Failed to retrieve the latest BerryBush version from GitHub")
                return
            if not (prefs.skipThisVersion and latestVer == tuple(prefs.latestKnownVersion)):
                # if the latest version isn't set to be skipped, always reset relevant prefs
                prefs.latestKnownVersion = latestVer
                prefs.skipThisVersion = False
                # finally, do the actual check
                if compareVers(currentVer, latestVer) < 0:
                    bpy.ops.brres.show_latest_version('INVOKE_DEFAULT',
                                                      current=currentVer, latest=latestVer, url=url)


LATEST_VERSION_CHECKER = LatestVersionChecker()


@bpy.app.handlers.persistent
def checkLatestVer(_):
    LATEST_VERSION_CHECKER.check(addonVer(), getPrefs(bpy.context))


class ShowLatestVersion(bpy.types.Operator):
    """Link to the newest version of BerryBush."""

    bl_idname = "brres.show_latest_version"
    bl_label = "BerryBush Update Available!"

    current: bpy.props.IntVectorProperty(size=3)
    latest: bpy.props.IntVectorProperty(size=3)
    url: bpy.props.StringProperty()

    def execute(self, context: bpy.types.Context):
        bpy.ops.wm.url_open(url=self.url)
        return {'FINISHED'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: bpy.types.Context):
        preferences = getPrefs(context)
        currentStr = verStr(self.current)
        latestStr = verStr(self.latest)
        label = f"Click OK to view the latest release ({currentStr} → {latestStr})"
        self.layout.label(text=label)
        row = self.layout.row().split(factor=.5)
        row.prop(preferences, "doUpdateChecks", invert_checkbox=True, text="Disable Update Checks")
        skipRow = row.row()
        skipRow.enabled = preferences.doUpdateChecks
        skipRow.prop(preferences, "skipThisVersion")


@bpy.app.handlers.persistent
def update(_):
    """Update the active Blend file if it uses an outdated BerryBush version."""
    bpy.ops.brres.update()


@bpy.app.handlers.persistent
def saveVer(_):
    """Before saving, update the BerryBush version for every scene in the Blend file."""
    currentVer = addonVer()
    for scene in bpy.data.scenes:
        scene.brres.version = currentVer


def addonVer() -> tuple[int, int, int]:
    """Current installed version of BerryBush, respresented as a tuple."""
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

    bl_idname = "brres.update"
    bl_label = "Update BRRES Settings"

    def execute(self, context: bpy.types.Context):
        # get latest version & blendfile version from scene settings
        sceneSettings = bpy.data.scenes[0].brres
        sceneVer = tuple(sceneSettings.version)
        currentVer = addonVer()
        # special 1.0.0 detection (it used a different version system)
        if "version_" in sceneSettings:
            sceneVer = (1, 0, 0)
            for scene in bpy.data.scenes:
                del scene.brres["version_"]
        # perform updates
        if sceneVer != currentVer and sceneVer != (0, 0, 0):
            self.report({'INFO'}, f"Loading file saved using BerryBush version {verStr(sceneVer)}"
                                  f" (current version: {verStr(currentVer)})")
            # 1.1.0
            if compareVers(sceneVer, (1, 1, 0)) < 0:
                # open vertex color gamma updater
                bpy.ops.brres.update_vert_colors_1_1_0('INVOKE_DEFAULT')
                sceneVer = (1, 1, 0)
            # 1.2.0
            if compareVers(sceneVer, (1, 2, 0)) < 0:
                # update texture transform names for fcurves
                for mat in bpy.data.materials:
                    for tex in mat.brres.textures:
                        tex.transform.name = tex.name
                    for indTf in mat.brres.indSettings.transforms:
                        indTf.transform.name = indTf.name
            # 1.4.0
            if compareVers(sceneVer, (1, 4, 0)) < 0:
                # update texture transform names for fcurves
                for scene in bpy.data.scenes:
                    for tevConfig in scene.brres.tevConfigs:
                        if not tevConfig.users:
                            tevConfig.fakeUser = True
            for scene in bpy.data.scenes:
                scene.brres.version = currentVer
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
