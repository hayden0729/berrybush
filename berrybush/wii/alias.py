from collections.abc import Sequence


def alias(*attrs: str, forceList: bool = False):
    """Property that works identically to some attribute, just under a different name.

    You can provide multiple attribute names to make this property
    act like a list containing all of them.
    If you want the list behavior with only one attribute, you can use the
    "forceList" keyword argument.
    """
    # for multi attrs, use attr list & special methods to make stuff as easy & intuitive as possible
    if len(attrs) > 1 or forceList:
        return property(
            fget=lambda self: AttrList(self, *attrs),
            fset=lambda self, v: replaceVals(self, attrs, v),
            fdel=lambda self: deleteVals(self, attrs))
    # if there's only one attr, just make the property work exactly like the attr
    return property(
        fget=lambda self: getattr(self, *attrs),
        fset=lambda self, v: setattr(self, *attrs, v),
        fdel=lambda self: delattr(self, *attrs)
    )


def replaceVals(obj, attrs: tuple[str, ...], newVals):
    """Replace the values of some attributes on an object with a set of new ones."""
    for attr, val in zip(attrs, newVals):
        setattr(obj, attr, val)


def deleteVals(obj, attrs: tuple[str, ...]):
    """Delete a set of attributes on an object."""
    for attr in attrs:
        delattr(obj, attr)


class AttrList(Sequence):
    """List of some attributes for some parent object.

    Using list-like access, the object's values for these attributes can be both retrieved and set.

    The attributes are referenced by name in the constructor, and then afterwards, they can be
    found at indices based on the order in which they were passed.
    This is used for multi-attribute aliases.
    """

    def __init__(self, parent, *attrs: str):
        self._parent = parent
        self._attrs = list(attrs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.__class__(self._parent, *self._attrs[i])
        return getattr(self._parent, self._attrs[i])

    def __len__(self):
        return len(self._attrs)

    def __setitem__(self, key, val):
        if isinstance(key, slice):
            for attr, v in zip(self._attrs[key], val):
                setattr(self._parent, attr, v)
        else:
            setattr(self._parent, self._attrs[key], val)

    def __repr__(self):
        return "AttrList" + repr(tuple(self))
