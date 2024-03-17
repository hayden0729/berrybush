# standard imports
from abc import abstractmethod
import json
import re
from typing import TYPE_CHECKING
from uuid import uuid4
# 3rd party imports
import bpy
# internal imports
from .common import makeUniqueName, parseDataPath
# special typing imports
if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = object


def _clonePropertyGroup(src: bpy.types.PropertyGroup, dst: bpy.types.PropertyGroup,
                        keepIntact: set[str] = None):
    """Recursively clone a property group, including data in pointer & collection properties.

    You can use the keepIntact parameter to keep certain properties intact on the initial level.
    Nested property groups will be unaffected by this.
    """
    if keepIntact is None:
        keepIntact = set()
    keepIntact.add("rna_type")
    if src == dst:
        return
    for propName, p in src.bl_rna.properties.items():
        if propName in keepIntact:
            continue
        srcVal = getattr(src, propName)
        if isinstance(p, bpy.types.PointerProperty) and isinstance(srcVal, bpy.types.PropertyGroup):
            _clonePropertyGroup(srcVal, getattr(dst, propName))
            continue
        elif isinstance(p, bpy.types.CollectionProperty):
            dstData = getattr(dst, propName)
            dstData.clear()
            for entry in srcVal:
                clone = dstData.add()
                _clonePropertyGroup(entry, clone)
            continue
        setattr(dst, propName, srcVal)


class DynamicPropertyGroup(bpy.types.PropertyGroup):
    """Property group that has dynamic classes created with it which must be registered."""

    @classmethod
    @abstractmethod
    def getDynamicClasses(cls) -> set[type]:
        """Get all the of this property group's associated dynamic classes."""
        return set()

    @classmethod
    @abstractmethod
    def generateDynamicClasses(cls):
        """Generate this property group's associated dynamic classes.

        Return this class's name in snake_case, which can be utilized in further class creation.
        """
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def checkDynamicClasses(cls):
        """Generate this property group's associated dynamic classes if they don't yet exist."""
        try:
            cls.getDynamicClasses()
        except AttributeError:
            cls.generateDynamicClasses()

    @classmethod
    def registerDynamicClasses(cls):
        """Register this property group's associated dynamic classes.

        (If not yet generated, generate them first)
        """
        cls.checkDynamicClasses()
        for c in cls.getDynamicClasses():
            bpy.utils.register_class(c)

    @classmethod
    def unregisterDynamicClasses(cls):
        """Unregister this property group's associated dynamic classes."""
        for c in cls.getDynamicClasses():
            bpy.utils.unregister_class(c)


class CloneablePropertyGroup(DynamicPropertyGroup):
    """Property group with an operator for cloning from a dictionary of other options.

    This cloning is recursive, meaning that groups contained within this one (even those that are
    not explicitly cloneable) will be cloned.
    """

    cloneOp: type[bpy.types.Operator]

    @classmethod
    def getCloneSources(cls, context: bpy.types.Context) -> dict[str, Self]:
        """Return a dict with the potential names & data from which this data can be cloned."""
        return {}

    def cloneFrom(self, other: Self, rename = False):
        """Clone another group's settings into this one."""
        keepIntact = {"uuid"}
        if not rename:
            keepIntact.add("name")
        _clonePropertyGroup(other, self, keepIntact)

    def drawCloneUI(self, layout: bpy.types.UILayout, text="Copy From..."):
        """Draw the clone UI for this item on a layout."""
        cloneOp = layout.operator(self.cloneOp.bl_idname, icon='COPYDOWN', text=text)
        cloneOp.dst = repr(self)

    @classmethod
    def getDynamicClasses(cls):
        return super().getDynamicClasses() | {cls.cloneOp}

    @classmethod
    def generateDynamicClasses(cls):
        clsNamePascal = cls.__name__
        clsNameSnake = super().generateDynamicClasses()
        cls.cloneOp = type(
            f"PropertyGroupClone{clsNamePascal}",
             (bpy.types.Operator, ),
             {"bl_idname": f"brres.property_group_clone_{clsNameSnake}",
             "bl_label": "Copy From", "bl_description": "Copy data from another source",
             "bl_options": {'UNDO', 'INTERNAL'}, "bl_property": "src",
             "execute": cls._getCloneOpExecute(), "invoke": cls._getCloneOpInvoke()}
        )
        annotations = cls.cloneOp.__annotations__ # properties must be annotations to work
        annotations["dst"] = bpy.props.StringProperty()
        annotations["src"] = bpy.props.EnumProperty(items=cls._getCloneOpItems())
        return clsNameSnake

    @classmethod
    def _getCloneOpItems(cls):
        """Get the function for the enum items for this class's clone operator."""
        def items(opSelf: bpy.types.Operator, context: bpy.types.Context):
            return ((name, name, "") for name in cls.getCloneSources(context))
        return items

    @classmethod
    def _getCloneOpExecute(cls):
        """Get the execute function for this class's clone operator."""
        def execute(opSelf: bpy.types.Operator, context: bpy.types.Context):
            dst = parseDataPath(opSelf.dst)
            dst.cloneFrom(cls.getCloneSources(context)[opSelf.src])
            context.area.tag_redraw()
            dst.id_data.update_tag()
            return {'FINISHED'}
        return execute

    @classmethod
    def _getCloneOpInvoke(cls):
        """Get the invoke function for this class's clone operator."""
        def invoke(opSelf: bpy.types.Operator, context: bpy.types.Context, event: bpy.types.Event):
            context.window_manager.invoke_search_popup(opSelf)
            return {'FINISHED'}
        return invoke


class _CustomIDCollectionProperty(bpy.types.PropertyGroup):

    activeIdx: bpy.props.IntProperty()
    # this is a CollectionProperty of the ID's type,
    # set in CustomIDPropertyGroup.generateDynamicClasses()
    coll_: None

    def add(self, updateActive = True):
        """Add an item to this collection & return it."""
        initialLen = len(self.coll_)
        if initialLen < self.maxLength():
            if updateActive:
                self.activeIdx = initialLen
            newItem = self.coll_.add()
            newItem.initialize()
            self.id_data.update_tag()
            return newItem

    def duplicate(self, i: int):
        """Duplicate the item at some index in this collection & return the copy."""
        if len(self.coll_) > 0:
            newItem = self.add()
            if newItem is not None:
                newItem.cloneFrom(self.coll_[i], rename=True)
            return newItem

    def index(self, v):
        """Return the index of an item in this collection from its name, UUID, or value."""
        try:
            return next(i for i, item in enumerate(self.coll_) if v in {i, item.name, item.uuid})
        except StopIteration as e:
            raise ValueError("Item not found in collection") from e

    def move(self, oldIdx: int, newIdx: int):
        """Move an item in this collection to a new index.

        The new index is constrained to the collection's bounds.

        The active index is adjusted appropriately so that it still points to the same item.
        """
        newIdx = max(0, min(len(self.coll_) - 1, newIdx))
        self.coll_.move(oldIdx, newIdx)
        self.id_data.update_tag()
        # active index adjustment
        activeIdx = self.activeIdx
        if activeIdx == oldIdx:
            self.activeIdx = newIdx
        elif activeIdx < oldIdx and activeIdx > newIdx:
            self.activeIdx += 1
        elif activeIdx > oldIdx and activeIdx < newIdx:
            self.activeIdx -= 1

    def remove(self, idx: int):
        """Remove an item from this collection."""
        initialLen = len(self.coll_)
        if initialLen > self.minLength() and idx in range(initialLen):
            self.coll_.remove(idx)
            activeIdx = self.activeIdx
            if activeIdx > idx or activeIdx == initialLen - 1:
                self.activeIdx -= 1
            self.id_data.update_tag()

    def activeItem(self):
        """Get the active item in this collection."""
        return self.coll_[self.activeIdx]

    def __len__(self):
        return len(self.coll_)

    def __iter__(self):
        yield from self.coll_

    def __getitem__(self, v):
        try:
            return self.coll_[v]
        except KeyError:
            try:
                return next(item for item in self.coll_ if item.uuid == v)
            except StopIteration as e:
                raise KeyError("Item not found in collection") from e

    @property
    def dtype(self) -> type["CustomIDPropertyGroup"]:
        return self.bl_rna.properties["coll_"].fixed_type

    def rules(self) -> dict[str]:
        """Rules (max & min length), if any, followed by this collection."""
        # rules are stored in the description of the pointer to this wrapper
        # get that through parent property group (ie whatever's pointing to this)
        parent = parseDataPath(repr(self), -1)
        pathToSelf = repr(self)[len(repr(parent)) + 1:]
        return json.loads(parent.bl_rna.properties[pathToSelf].description)

    def maxLength(self) -> int | float:
        """Maximum length of this collection. Either an int or infinity."""
        return self.rules().get("maxLength", float("inf"))

    def minLength(self) -> int:
        """Minimum length of this collection."""
        return self.rules().get("minLength", 0)

    def drawList(self, layout: bpy.types.UILayout):
        row = layout.row()
        row.template_list(self.dtype._uiListType.bl_idname, "", self, "coll_", self, "activeIdx")
        col = row.column(align=True)
        collOps: list[CustomIDCollOp] = []
        collOps.append(col.operator(CustomIDCollOpAdd.bl_idname, icon='ADD', text=""))
        collOps.append(col.operator(CustomIDCollOpRemove.bl_idname, icon='REMOVE', text=""))
        col.separator()
        collOps.append(col.operator(CustomIDCollOpClone.bl_idname, icon='DUPLICATE', text=""))
        col.separator()
        collOps.append(col.operator(CustomIDCollOpMoveUp.bl_idname, icon='TRIA_UP', text=""))
        collOps.append(col.operator(CustomIDCollOpMoveDown.bl_idname, icon='TRIA_DOWN', text=""))
        for op in collOps:
            op.collPath = repr(self)
            try:
                op.refPath = ""
                op.refProp = ""
            except AttributeError:
                pass

    def drawAccessor(self, context: bpy.types.Context, layout: bpy.types.UILayout,
                     refData, refPropName: str):
        row = layout.row(align=True)
        collOps = []
        collOps.append(row.operator(CustomIDCollOpChoose.bl_idname, icon='PRESET', text=""))
        uuid = getattr(refData, refPropName)
        try:
            item = self[uuid]
            collOps.append(row.operator(CustomIDCollOpClearSelection.bl_idname, icon='X', text=""))
            collOps.append(row.operator(CustomIDCollOpAdd.bl_idname, icon='ADD', text=""))
            row.prop(item, "name", text="", expand=True)
            item.drawAccessorExtras(context, row)
            collOps.append(row.operator(CustomIDCollOpClone.bl_idname, icon='DUPLICATE', text=""))
            item.drawCloneUI(row, text="")
            collOps.append(row.operator(CustomIDCollOpRemove.bl_idname, icon='TRASH', text=""))
        except KeyError:
            collOps.append(row.operator(CustomIDCollOpAdd.bl_idname, icon='ADD', text="New"))
        for op in collOps:
            op.collPath = repr(self)
            try:
                op.refPath = repr(refData)
                op.refProp = refPropName
            except AttributeError:
                pass


class CustomIDPropertyGroup(CloneablePropertyGroup):

    def getName(self) -> str:
        """Get this property group's name."""
        try:
            return self["name"]
        except KeyError:
            return ""

    def setName(self, name: str):
        """Set this property group's name, adding numbers to the end if it's already taken."""
        # based on this:
        # https://blender.stackexchange.com/questions/15122/collectionproperty-avoid-duplicate-names
        if name == "":
            return
        names = {item.name for item in self._parentCollection}
        names.remove(self.name)
        newName = makeUniqueName(name, names)
        self["name"] = newName
        self.onNameChange(newName)

    def onNameChange(self, newName: str):
        """Called whenever this property group's name is changed, after the change."""

    _collPropertyType: type[_CustomIDCollectionProperty]
    _uiListType: type[bpy.types.UIList]

    name: bpy.props.StringProperty(name="Name", get=getName, set=setName)
    uuid: bpy.props.StringProperty()

    @classmethod
    def defaultName(cls) -> str:
        """Get the default name for an instance of this property group when it's first created."""
        return ""

    def initialize(self):
        """Initialize this property group after its creation."""
        self.uuid = uuid4().hex
        self.name = self.defaultName()

    def delete(self):
        """Remove this property group from its parent collection."""
        parentColl = self._parentCollection
        parentColl.remove(parentColl.index(self.uuid))

    @property
    def _parentCollection(self) -> _CustomIDCollectionProperty:
        return parseDataPath(repr(self), -1)

    def chooserOptionDisplayName(self, context: bpy.types.Context):
        """Name used by the item selection menu in the accessor UI."""
        return self.name

    def drawListItem(self: bpy.types.UIList, context: bpy.types.Context, layout: bpy.types.UILayout,
                     data, item, icon: int, active_data, active_property: str, index=0, flt_flag=0):
        """Draw a list entry for this property group.

        ("self" refers to the list, as that calls this)"""
        # pylint: disable=attribute-defined-outside-init
        self.use_filter_show = False
        layout.prop(item, "name", text="", emboss=False)

    def drawAccessorExtras(self, context: bpy.types.Context, layout: bpy.types.UILayout):
        """Draw any extra fields for the ID accessor UI."""

    @classmethod
    def generateDynamicClasses(cls):
        clsNamePascal = cls.__name__
        clsNameSnake = super().generateDynamicClasses()
        # collection property wrapper class
        cls._collPropertyType = type(
            f"{clsNamePascal}CollectionProperty",
            (_CustomIDCollectionProperty, ),
            {}
        )
        annotations = cls._collPropertyType.__annotations__
        annotations["coll_"] = bpy.props.CollectionProperty(type=cls)
        # ui list class
        cls._uiListType = type(
            f"{clsNamePascal}UIList",
             (bpy.types.UIList, ),
             {"bl_idname": f"BRRES_UL_{clsNameSnake}_list", "draw_item": cls.drawListItem},
        )
        return clsNameSnake

    @classmethod
    def getDynamicClasses(cls):
        return super().getDynamicClasses() | {cls._collPropertyType, cls._uiListType}

    @classmethod
    def CustomIDCollectionProperty(cls, maxLength: int = None, minLength: int = None):
        """Return a new custom ID collection property definition.

        Optionally, a maximum & minimum length may be provided. Note that the initial length is
        always 0, and entries must be added manually after initialization to get to the minimum.
        """
        cls.checkDynamicClasses()
        rules = {}
        if maxLength is not None:
            rules["maxLength"] = maxLength
        if minLength is not None:
            rules["minLength"] = minLength
        return bpy.props.PointerProperty(type=cls._collPropertyType, description=json.dumps(rules))


class UsableCustomIDPropertyGroup(CustomIDPropertyGroup):
    """Custom ID property group with users. Groups without any users are deleted on startup."""

    fakeUser: bpy.props.BoolProperty(
        name="Fake User",
        description="Save this data-block even if it has no users",
        default=False
    )

    @abstractmethod
    def getUsers(self) -> int:
        """Get the number of users this ID has."""

    users: bpy.props.IntProperty(
        name="Users",
        description="Data with no users will be removed when the blend-file is reopened",
        get=lambda self: self.getUsers(),
        set=lambda self, v: None
    )

    @classmethod
    @abstractmethod
    def getMaybeUnused(cls) -> list[Self]:
        """The IDs in the returned list are checked on startup, and those w/o users are deleted."""

    def drawAccessorExtras(self, context: bpy.types.Context, layout: bpy.types.UILayout):
        super().drawAccessorExtras(context, layout)
        usersSublayout = layout.row(align=True)
        usersSublayout.enabled = False
        usersSublayout.scale_x = .5
        usersSublayout.prop(self, "users", text="")
        fakeUserIcon = "FAKE_USER_ON" if self.fakeUser else "FAKE_USER_OFF"
        layout.prop(self, "fakeUser", text="", icon=fakeUserIcon)

    def chooserOptionDisplayName(self, context: bpy.types.Context):
        prefix = "F " if self.fakeUser else "  " if self.getUsers() else "0 "
        return prefix + super().chooserOptionDisplayName(context)


def getUnusedPropertyGroupRemovalHandler(*types: type[UsableCustomIDPropertyGroup]):
    """Handler to delete the property groups w/o users for the given usable custom ID types."""
    @bpy.app.handlers.persistent
    def handler(_):
        for idType in types:
            for propGroup in idType.getMaybeUnused():
                if not propGroup.fakeUser and not propGroup.getUsers():
                    propGroup.delete()
    return handler


class CustomIDCollOp(bpy.types.Operator):
    """Manipulates a custom ID collection referenced through a property."""

    # note that path to collection wrapper must be stored as a string, since the real data can't
    # be stored as a property (or through any other means) on the operator
    collPath: bpy.props.StringProperty()

    @property
    def _coll(self) -> _CustomIDCollectionProperty:
        return parseDataPath(self.collPath)


class CustomIDCollRefOp(CustomIDCollOp):
    """Manipulates a custom ID collection and a UUID reference to an item in it."""

    refPath: bpy.props.StringProperty()
    refProp: bpy.props.StringProperty()

    @property
    def _hasRefID(self):
        return self.refPath != ""

    @property
    def _refIDHolder(self):
        return parseDataPath(self.refPath)

    @property
    def _refID(self) -> str:
        if self._hasRefID:
            return getattr(self._refIDHolder, self.refProp)
        return None

    @_refID.setter
    def _refID(self, v: str):
        if self._hasRefID:
            refIDHolder = self._refIDHolder
            setattr(refIDHolder, self.refProp, v)
            refIDHolder.id_data.update_tag()


class CustomIDCollOpAdd(CustomIDCollRefOp):
    bl_idname = "brres.custom_id_collection_add"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Add Item"
    bl_description = "Add a new item"

    def execute(self, context: bpy.types.Context):
        try:
            self._refID = self._coll.add().uuid
            return {'FINISHED'}
        except AttributeError:
            return {'CANCELLED'}


class CustomIDCollOpRemove(CustomIDCollRefOp):
    bl_idname = "brres.custom_id_collection_remove"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Delete Item"
    bl_description = "Delete the selected item from the scene"

    def execute(self, context: bpy.types.Context):
        if self._refID == "" or len(self._coll) == 0:
            return {'CANCELLED'}
        try:
            i = self._coll.index(self._refID)
        except ValueError:
            i = self._coll.activeIdx
        self._coll.remove(i)
        self._refID = ""
        return {'FINISHED'}


class CustomIDCollOpClone(CustomIDCollRefOp):
    bl_idname = "brres.custom_id_collection_clone"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Duplicate Item"
    bl_description = "Duplicate the selected item"

    def execute(self, context: bpy.types.Context):
        try:
            i = self._coll.index(self._refID)
        except ValueError:
            i = self._coll.activeIdx
        try:
            self._refID = self._coll.duplicate(i).uuid
            return {'FINISHED'}
        except AttributeError:
            return {'CANCELLED'}


class CustomIDCollOpClearSelection(CustomIDCollRefOp):
    bl_idname = "brres.custom_id_collection_clear_selection"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Clear Selection"
    bl_description = "Clear this selection without deleting the item"

    def execute(self, context: bpy.types.Context):
        if self._refID == "":
            return {'CANCELLED'}
        self._refID = ""
        return {'FINISHED'}


class CustomIDCollOpChoose(CustomIDCollRefOp):
    bl_idname = "brres.custom_id_collection_choose"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Choose Item"
    bl_description = "Choose an item from the collection"
    bl_property = "chosenID"

    def options(self, context: bpy.types.Context):
        # can't use the _coll property directly because of some blender weirdness, so do this
        # (specifically, this isn't actually called by an instance of the operator; rather,
        # something else that's given its attributes but not its methods)
        return tuple((
            item.uuid,
            item.chooserOptionDisplayName(context),
            ""
        ) for item in parseDataPath(self.collPath)) # pylint: disable=not-an-iterable

    chosenID: bpy.props.EnumProperty(items=options)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        context.window_manager.invoke_search_popup(self)
        return {'FINISHED'}

    def execute(self, context: bpy.types.Context):
        if self._refID == self.chosenID:
            return {'CANCELLED'}
        self._refID = self.chosenID
        context.area.tag_redraw()
        return {'FINISHED'}


class CustomIDCollOpMoveUp(CustomIDCollOp):
    bl_idname = "brres.custom_id_collection_move_up"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Move Up"
    bl_description = "Move the selected item up"

    def execute(self, context: bpy.types.Context):
        initialIdx = self._coll.activeIdx
        self._coll.move(initialIdx, initialIdx - 1)
        if self._coll.activeIdx == initialIdx:
            return {'CANCELLED'}
        return {'FINISHED'}


class CustomIDCollOpMoveDown(CustomIDCollOp):
    bl_idname = "brres.custom_id_collection_move_down"
    bl_options = {'UNDO', 'INTERNAL'}
    bl_label = "Move Down"
    bl_description = "Move the selected item down"

    def execute(self, context: bpy.types.Context):
        initialIdx = self._coll.activeIdx
        self._coll.move(initialIdx, initialIdx + 1)
        if self._coll.activeIdx == initialIdx:
            return {'CANCELLED'}
        return {'FINISHED'}
