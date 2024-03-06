from abc import abstractmethod
import bpy


class ObjectLimiter():

    def __init__(self, context: bpy.types.Context):
        self._context = context
        self._objects: dict[str, bool] = {}

    def includes(self, obj: bpy.types.Object):
        if obj.name in self._objects:
            return self._objects[obj.name]
        result: bool
        if obj.type == 'ARMATURE' and any(self.includes(c) for c in obj.children_recursive):
            result = True
        else:
            result = self._shouldInclude(obj)
        self._objects[obj.name] = result
        return result

    @abstractmethod
    def _shouldInclude(self, obj: bpy.types.Object):
        """Whether obj should be included by this limiter (after caching/parenting processing)"""


class AllObjectLimiter(ObjectLimiter):

    def _shouldInclude(self, obj: bpy.types.Object):
        return True


class SelectedObjectLimiter(ObjectLimiter):

    def _shouldInclude(self, obj: bpy.types.Object):
        return obj.select_get()


class VisibleObjectLimiter(ObjectLimiter):

    def __init__(self, context: bpy.types.Context):
        super().__init__(context)
        self._getObjStatuses(context.view_layer.layer_collection)

    def _getObjStatuses(self, layerCollection: bpy.types.LayerCollection):
        """Evaluate, for each object contained recursively within a layer collection,
        whether that object should be included by this limiter."""
        if layerCollection.visible_get():
            # layer collection is visible, so test hiding for direct children
            # (sub-collections, objects)
            for childLayerCollection in layerCollection.children:
                self._getObjStatuses(childLayerCollection)
            for obj in layerCollection.collection.objects:
                self._objects[obj.name] = not obj.hide_get()
        else:
            # layer collection is invisible, so contents are invisible
            for obj in layerCollection.collection.all_objects:
                # ensure object isn't already tracked, in case it's visible in another collection
                if obj.name not in self._objects:
                    self._objects[obj.name] = False

    def _shouldInclude(self, obj: bpy.types.Object):
        # if we get here, it means the object wasn't found in the initial view layer sweep
        # so just default to hide_get()
        return not obj.hide_get()


class ObjectLimiterFactory():

    _LIMITER_TYPES = {
        "ALL": AllObjectLimiter,
        "SELECTED": SelectedObjectLimiter,
        "VISIBLE": VisibleObjectLimiter
    }

    @classmethod
    def create(cls, context: bpy.types.Context, typeStr: str):
        return cls._LIMITER_TYPES[typeStr](context)
