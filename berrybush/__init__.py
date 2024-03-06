# BerryBush - BRRES support for Blender focused on New Super Mario Bros. Wii
# Copyright (C) 2023 hayden0729
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# standard imports
import pathlib
# 3rd party imports
import bpy
import bpy.utils.previews
# internal imports
from .blender import (
    backup,
    bone,
    brresexport,
    brresimport,
    material,
    mesh,
    preferences,
    proputils,
    render,
    scene,
    tev,
    texture,
    updater,
    verify
)


ICONS: bpy.utils.previews.ImagePreviewCollection
"""BerryBush's custom icons. Access by importing within a function to avoid circular imports."""


bl_info = {
    "name" : "BRRES format (BerryBush)",
    "author": "hayden0729",
    "version": (1, 3, 4),
    "blender" : (3, 3, 0),
    "location": "File > Import-Export",
    "description": "NSMBW focused BRRES support",
    "doc_url": "https://github.com/hayden0729/berrybush/wiki",
    "category" : "Import-Export"
}


classes = (
    # data
    tev.TevStageSelSettings,
    tev.TevStageIndSettings,
    tev.TevStageColorParams,
    tev.TevStageAlphaParams,
    tev.TevStageSettings,
    tev.ColorSwapSettings,
    tev.TevSettings,
    texture.TextureTransform,
    texture.TexImg,
    texture.TexSettings,
    texture.MipmapSlot,
    texture.ImgSettings,
    material.IndTexIndividualSettings,
    material.IndTransform,
    material.IndTexSettings,
    material.ColorRegSettings,
    material.LightChannelColorAlphaSettings,
    material.LightChannelSettings,
    material.AlphaSettings,
    material.DepthSettings,
    material.MiscMatSettings,
    material.MatSettings,
    mesh.MeshAttrSettings,
    mesh.MeshSettings,
    bone.BoneSettings,
    scene.SceneSettings,
    # ui
    material.MatPanel,
    tev.TevPanel,
    tev.TevColorSwapPanel,
    tev.TevIndSelPanel,
    tev.TevStagePanel,
    tev.TevStageSelPanel,
    tev.TevStageIndPanel,
    tev.TevStageColorPanel,
    tev.TevStageAlphaPanel,
    texture.TexPanel,
    texture.TexTransformPanel,
    texture.TexSettingsPanel,
    texture.ImgPanel,
    texture.MipmapPanel,
    material.IndTexPanel,
    material.IndTransformPanel,
    material.ColorRegPanel,
    material.LightChannelPanel,
    material.AlphaSettingsPanel,
    material.DepthSettingsPanel,
    material.MatMiscPanel,
    mesh.MeshPanel,
    bone.BonePanel,
    brresimport.GeneralPanel,
    brresimport.ArmPanel,
    brresimport.AnimPanel,
    brresexport.GeneralPanel,
    brresexport.ArmGeoPanel,
    brresexport.ImagePanel,
    brresexport.AnimPanel,
    render.FilmPanel,
    # operators
    backup.ClearBackups,
    proputils.CustomIDCollOpAdd,
    proputils.CustomIDCollOpRemove,
    proputils.CustomIDCollOpClone,
    proputils.CustomIDCollOpClearSelection,
    proputils.CustomIDCollOpChoose,
    proputils.CustomIDCollOpMoveUp,
    proputils.CustomIDCollOpMoveDown,
    brresimport.ImportBRRES,
    brresexport.ExportBRRES,
    verify.VerifyBRRES,
    updater.UpdateBRRES,
    updater.UpdateVertColors1_1_0,
    # preferences
    preferences.BerryBushPreferences,
    # render engine
    render.BRRESRenderEngine
)


def register():
    global ICONS # pylint: disable=global-statement
    ICONS = bpy.utils.previews.new()
    for f in (pathlib.Path(__file__).parent / "blender" / "icons").iterdir():
        if f.is_file():
            ICONS.load(f.stem.upper(), str(f.resolve()), 'IMAGE')
    for cls in classes:
        bpy.utils.register_class(cls)
        if issubclass(cls, proputils.DynamicPropertyGroup):
            cls.registerDynamicClasses()
    bpy.types.Scene.brres = bpy.props.PointerProperty(type=scene.SceneSettings)
    bpy.types.Bone.brres = bpy.props.PointerProperty(type=bone.BoneSettings)
    bpy.types.Mesh.brres = bpy.props.PointerProperty(type=mesh.MeshSettings)
    bpy.types.Material.brres = bpy.props.PointerProperty(type=material.MatSettings)
    bpy.types.Image.brres = bpy.props.PointerProperty(type=texture.ImgSettings)
    bpy.types.TOPBAR_MT_file_import.append(brresimport.drawOp)
    bpy.types.TOPBAR_MT_file_export.append(brresexport.drawOp)
    bpy.types.VIEW3D_MT_object.append(verify.drawOp)
    bpy.app.handlers.load_post.append(updater.update)
    bpy.app.handlers.save_pre.append(updater.saveVer)
    render.BRRESRenderEngine.registerOnPanels()


def unregister():
    render.BRRESRenderEngine.unregisterOnPanels()
    bpy.app.handlers.save_pre.remove(updater.saveVer)
    bpy.app.handlers.load_post.remove(updater.update)
    bpy.types.VIEW3D_MT_object.remove(verify.drawOp)
    bpy.types.TOPBAR_MT_file_export.remove(brresexport.drawOp)
    bpy.types.TOPBAR_MT_file_import.remove(brresimport.drawOp)
    del bpy.types.Image.brres
    del bpy.types.Material.brres
    del bpy.types.Mesh.brres
    del bpy.types.Bone.brres
    del bpy.types.Scene.brres
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        if issubclass(cls, proputils.DynamicPropertyGroup):
            cls.unregisterDynamicClasses()
    bpy.utils.previews.remove(ICONS)


if __name__ == "__main__":
    register()
