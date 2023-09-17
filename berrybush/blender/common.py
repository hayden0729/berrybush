# standard imports
from ast import literal_eval
import pathlib
import re
import textwrap
# 3rd party imports
import bpy
from bpy_extras.io_utils import axis_conversion
import numpy as np


UNDOCUMENTED = "Undocumented"
LOG_PATH = pathlib.Path(__file__).parent.parent / "log.txt"
UI_COL_SEP = .2 # extra separation between rows that must be added for column layouts


# in general, blender uses z for up and -y for forward, while brres uses y for up and z for forward
MTX_TO_BRRES = axis_conversion(from_forward='-Y', from_up='Z', to_forward='Z', to_up='Y').to_4x4()
MTX_FROM_BRRES = MTX_TO_BRRES.inverted()

# finally, this converts between blender's world space conventions and bone conventions
# (note: this is the same conversion as MTX_TO_BRRES and MTX_FROM_BRRES by pure coincidence)
MTX_TO_BONE = axis_conversion(from_forward='-Y', from_up='Z', to_forward='Z', to_up='Y')
MTX_FROM_BONE = MTX_TO_BONE.inverted()


ATTR_COMP_COUNTS = {'FLOAT_VECTOR': 3, 'FLOAT_COLOR': 4, 'BYTE_COLOR': 4, 'FLOAT2': 2}


class PropertyPanel(bpy.types.Panel):
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_options = {'DEFAULT_CLOSED'}


def transformAxis(axis: int, mtx: np.ndarray) -> tuple[int, int]:
    """Given an axis index and an axis conversion matrix, get the corresponding output axis index.

    The returned tuple contains this index & a scale factor applied to the axis (usually 1 or -1).

    The matrix is assumed to be an invertible composition of 90 degree rotations (on any axes) and
    scalings. Anything else will lead to undefined results. 
    """
    vec = np.array((0, ) * len(mtx))
    vec[axis] = 1
    outputVec = mtx @ vec
    outputAxis = np.flatnonzero(outputVec)[0]
    return (outputAxis, int(outputVec[outputAxis]))


def solidView(context: bpy.types.Context):
    """Switch every 3D viewport space to solid view.

    Return a dict mapping each space to its original view type.
    """
    spaceShadingTypes: dict[bpy.types.Space, str] = {}
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        spaceShadingTypes[space] = space.shading.type
                        space.shading.type = 'SOLID'
    return spaceShadingTypes


def restoreView(spaceShadingTypes: dict[bpy.types.Space, str]):
    """From a dict mapping 3D viewport spaces to shading view types, set the view types."""
    for space, shadingType in spaceShadingTypes.items():
        space.shading.type = shadingType


def limitIncludes(limitTo: bool, obj: bpy.types.Object):
    """Return True if an object should be included in a group based on a "limit to" setting."""
    if obj.type == 'ARMATURE' and any(limitIncludes(limitTo, c) for c in obj.children_recursive):
        return True
    if limitTo == 'SELECTED':
        return obj.select_get()
    if limitTo == 'VISIBLE':
        return not obj.hide_get()
    return True


def enumVal(data, propName: str, enumItem: str = None, callback = None):
    """Get the value for an enum property. If no enum item is provided, the active one is used.

    For enum properties that rely on a callback function for their items, it must be provided.
    """
    if enumItem is None:
        enumItem = getattr(data, propName)
    if callback:
        # enum_items doesn't work, so we have to run the callback and find value manually
        items = callback(data, bpy.context)
        if len(items[0]) > 3: # numeric values are provided in items
            return next(item[-1] for item in items if item[0] == enumItem)
        else: # numeric values are automatic
            return next(i for i, item in enumerate(items) if item[0] == enumItem)
    return data.bl_rna.properties[propName].enum_items[enumItem].value


def paragraphLabel(layout: bpy.types.UILayout, text: str, widthChars = 55):
    """Draw a multi-line label w/ automatic wrapping on a layout."""
    # https://blender.stackexchange.com/questions/74052/wrap-text-within-a-panel
    wrapp = textwrap.TextWrapper(width=widthChars)
    wList = wrapp.wrap(text)
    for textRow in wList:
        row = layout.row(align=True)
        row.alignment = 'EXPAND'
        row.scale_y = 0.6
        row.label(text=textRow)


def usedMatSlots(obj: bpy.types.Object, mesh: bpy.types.Mesh):
    """Get the material slots of a mesh object that are used by its mesh."""
    usedSlots = set(f.material_index for f in mesh.polygons)
    return (matSlot for i, matSlot in enumerate(obj.material_slots) if i in usedSlots)


def filterMats(includeActive = True, includeGreasePencil = True):
    """Return a filtered generator of Blender's materials."""
    tests = []
    if not includeActive and bpy.context.active_object is not None:
        tests.append(lambda m: m is not bpy.context.active_object.active_material)
    if not includeGreasePencil:
        tests.append(lambda m: not m.is_grease_pencil)
    return (m for m in bpy.data.materials if all(test(m) for test in tests))


def parseDataPath(path: str, parentLevel = 0):
    """Parse a full Blender data path without exec or eval (avoiding security concerns).

    Optionally, you can use the parentLevel parameter to go further up the path. Setting it to 0
    evaluates the full path, but -1 gets the parent of that result, -2 gets the parent of the -1
    result, etc. Positive values are invalid.
    """
    data = [bpy]
    items = path.split(".")[1:] # no need to parse "bpy" in beginning
    while items:
        item = items.pop(0)
        if "[" in item: # dict key or list index
            # blender's dict keys, when represented as strings, use different quote chars
            # (either ' or ") depending on what kinds of quotes are used within the string
            # if both kinds are used, it just puts backslashes on one kind
            # this system parses out the key (allowed to contain periods, quotes, backslahes, etc)
            # and hopefully produces correct output 100% of the time
            itemBase, key = item.split("[", 1)
            quote = key[0]
            if quote not in "\'\"": # list index
                data.append(getattr(data[-1], itemBase)[int(key[:-1])])
            else: # dict key
                # as long as the key doesn't end w/ a real (not escaped) quote and right bracket,
                # keep going through the initial split on periods
                while not (key[-2:] == f"{quote}]" and (key[-3] != "\\" or key[-4:-2] == "\\\\")):
                    key += "." + items.pop(0)
                data.append(getattr(data[-1], itemBase)[literal_eval(key[:-1])])
        else: # regular property
            data.append(getattr(data[-1], item))
    return data[parentLevel - 1]


def makeUniqueName(name: str, usedNames: set[str]):
    """Make a name unique by adding numbers to it, the way Blender does for its data blocks."""
    # based on this:
    # https://blender.stackexchange.com/questions/15122/collectionproperty-avoid-duplicate-names
    # get current number if value's already in number format
    stem, n = name, 0
    match = re.match(r"(.*)\.(\d{3,})", name)
    if match is not None:
        stem, n = match.groups()
        n = int(n)
    # go through names and increment number until a vacant spot's found
    while name in usedNames:
        n += 1
        name = f"{stem}.{n:03d}"
    return name


def getPropName(data, p: str):
    """Get the name of a property from its parent data and ID."""
    return data.rna_type.properties[p].name # use rna_type over bl_rna bc it works w/ operator props


def drawIcon(layout: bpy.types.UILayout, icon: str | int):
    if isinstance(icon, str):
        layout.label(icon=icon)
    elif isinstance(icon, int):
        layout.label(icon_value=icon)


def drawProp(layout: bpy.types.UILayout, data, p: str,
             factor: float = .4, text = "", icon: str | int = None, index = -1):
    """Draw a property on a layout with its label & data split according to some factor."""
    row = layout.row().split(factor=factor)
    labelCol = row.column()
    labelCol.alignment = 'RIGHT'
    labelCol.label(text=getPropName(data, p) if not text else text)
    propRow = row.column().row()
    propRow.prop(data, p, text="", index=index)
    drawIcon(propRow, icon)


def drawCheckedProp(layout: bpy.types.UILayout, boolData, boolProp: str,
                    mainData, mainProp: str, splitFactor: float = .4, decorate = False):
    """Draw a property on a layout with a checkbox for enabling it based on a bool property."""
    row = layout.row().split(factor=splitFactor)
    row.use_property_split = False
    labelCol = row.column()
    labelCol.alignment = 'RIGHT'
    labelCol.label(text=getPropName(mainData, mainProp))
    dataRow = row.column().row(align=True)
    dataRow.prop(boolData, boolProp, text="")
    fieldRow = dataRow.row()
    fieldRow.use_property_decorate = fieldRow.use_property_split = decorate
    fieldRow.enabled = getattr(boolData, boolProp)
    fieldRow.prop(mainData, mainProp, text="")


def foreachGet(coll: bpy.types.bpy_prop_collection, attr: str,
               numComps = 1, dtype = np.float32) -> np.ndarray:
    """Quickly create an array of some attribute of a bpy_prop_collection.

    For instance, to get a mesh's positions, use foreachGet(mesh.vertices, "co", 3)
    """
    arr = np.empty((len(coll) * numComps), dtype)
    coll.foreach_get(attr, arr)
    return arr.reshape((-1, numComps)) if numComps > 1 else arr


def getLoopVertIdcs(mesh: bpy.types.Mesh):
    """Get the vertex indices for each loop of a mesh."""
    return foreachGet(mesh.loops, "vertex_index", dtype=np.integer)


def getLoopFaceIdcs(mesh: bpy.types.Mesh):
    """Get the face indices for each loop of a mesh."""
    faceLoopCounts = foreachGet(mesh.polygons, "loop_total", dtype=np.integer)
    return np.arange(len(mesh.polygons)).repeat(faceLoopCounts)


def getFaceMatIdcs(mesh: bpy.types.Mesh):
    """Get the material indices for each face of a mesh."""
    return foreachGet(mesh.polygons, "material_index", dtype=np.integer)


def getLayerData(mesh: bpy.types.Mesh, layerNames: tuple[str],
                 isUV = False, unique = True, doProcessing = True):
    """Get the data for a series of mesh layers.

    Each entry of this data contains the layer, its data, and the indices into this data that should
    be used for its domain. If "unique" is disabled, the indices aren't provided, and the data just
    corresponds to each entry of the domain right away.

    The layers are looked up by name, and the returned value is a list of tuples containing layers &
    numpy arrays of their data. Since name collisions can occur between UV layers and generic
    attributes, you can choose which to prioritize.

    Optionally, data can be processed after retrieval. If the UV flag is enabled, this causes
    the Y coordinates of the data to be flipped because of differences in conventions between
    Blender & BRRES; otherwise, this causes the data to be clamped from 0-1 and gamma-corrected.
    """
    uvLayers = mesh.uv_layers.keys()
    genericLayers = mesh.attributes.keys()
    allData: list[tuple[bpy.types.Attribute | None, np.ndarray | None, np.ndarray | None]] = []
    for layerName in layerNames:
        layerData: np.ndarray = None
        layerIdcs: np.ndarray = None
        # deal w/ name collisions by prioritizing either uv or generic layers
        isUVLayer = False
        isGenericLayer = False
        if isUV:
            isUVLayer = layerName in uvLayers
            isGenericLayer = layerName in genericLayers if not isUVLayer else False
        else:
            isGenericLayer = layerName in genericLayers
            isUVLayer = layerName in uvLayers if not isGenericLayer else False
        # actually get data
        layer: bpy.types.Attribute = None
        if isUVLayer:
            layer = mesh.uv_layers[layerName]
            layerData = foreachGet(layer.data, "uv", 2)
        elif isGenericLayer:
            layer = mesh.attributes[layerName]
            layerCompCount = ATTR_COMP_COUNTS.get(layer.data_type, 1)
            for propName in ("value", "color", "vector"):
                try:
                    layerData = foreachGet(layer.data, propName, layerCompCount)
                    break
                except AttributeError:
                    pass
        # then, process data & add to list
        if layerData is not None:
            if doProcessing:
                layerData = simplifyLayerData(layerData)
                if isUV: # flip uvs
                    layerData[:, 1] = 1 - layerData[:, 1]
                else: # correct & clamp colors
                    layerData[:, :3] **= (1 / 2.2)
                    layerData.clip(0, 1, out=layerData)
            if unique:
                layerData, layerIdcs = np.unique(layerData, return_inverse=True, axis=0)
        allData.append((layer, layerData, layerIdcs))
    return allData


def setLayerData(layerData: dict[bpy.types.Attribute, np.ndarray]):
    """Easily set the data for mesh attribute/UV layers. No additional processing is performed."""
    meshes: set[bpy.types.Mesh] = set()
    for layer, data in layerData.items():
        meshes.add(layer.id_data)
        for propName in ("uv", "value", "color", "vector"):
            try:
                layer.data.foreach_set(propName, data.flatten())
                break
            except AttributeError:
                pass
    for mesh in meshes:
        mesh.update()


def simplifyLayerData(data: np.ndarray):
    """Simplify data for a mesh attribute for the sake of file size optimization."""
    scale = 1 << 14
    return (data * scale).round() / scale
