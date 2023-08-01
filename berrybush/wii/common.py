# standard imports
from enum import Enum
from itertools import chain
from typing import Iterable, TypeVar, TYPE_CHECKING
# special typing imports
if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = object


T = TypeVar("T")
U = TypeVar("U")


class Tree():
    """Simple base class for data stored in a tree structure."""

    def __init__(self, parent: Self = None):
        self._parent: Self = None
        self._children: list[Self] = []
        self.parent = parent

    def addChild(self, child: Self):
        """Add a child to this tree item. If the child is already present, do nothing."""
        if child not in self._children:
            child.parent = None # remove child from current parent
            child._parent = self
            self._children.append(child) # set parent and add here

    def removeChild(self, child: Self):
        """Remove a child from this tree item."""
        if child._parent is self:
            child._parent = None
            self._children.remove(child)
        else:
            raise ValueError("Child does not belong to this parent")

    @property
    def parent(self):
        """Parent of this tree item."""
        return self._parent

    @parent.setter
    def parent(self, parent: Self):
        if parent is not self._parent:
            if self._parent is not None:
                self._parent.removeChild(self) # remove from current parent
            if parent is not None:
                parent.addChild(self) # add to new parent

    @property
    def children(self):
        """Children of this tree item."""
        return tuple(self._children)

    def deepChildren(self, includeSelf: bool = True) -> chain[Self]:
        """Generator for children, grandchildren, etc of this tree item retrieved recursively."""
        childrenGen = chain.from_iterable(c.deepChildren() for c in self._children)
        if not includeSelf:
            return childrenGen
        return chain((self, ), childrenGen)

    def ancestors(self, includeSelf: bool = True) -> tuple[Self, ...]:
        """Tuple containing the ancestors of this tree item retrieved recursively."""
        parent = self.parent
        return () + (parent.ancestors() if parent else ()) + ((self, ) if includeSelf else ())


class EnumWithAttr(Enum):
    """Use this for an enum with a custom attribute ("_attr_") in addition to its value.

    Based on stuff from https://stackoverflow.com/questions/12680080/python-enums-with-attributes
    """
    def __new__(cls, val, attr):
        entry = object.__new__(cls)
        entry._value_ = val
        entry._attr_ = attr
        return entry


def getKey(d: dict, val):
    """Get first key for some value in a dict.

    Raises ValueError if the value is not present.
    """
    try:
        return next(k for k, v in d.items() if v == val)
    except StopIteration as e:
        raise ValueError(f"'{val}' is not in dict") from e


def keyVals(d: dict[T, U], vals: tuple) -> list[U]:
    """Get a list containing the values for a sequence of keys in a dict.

    Keys not in the dict are ignored.
    """
    return [d[v] for v in vals if v in d]


def keyValsDef(d: dict[T, U], vals: tuple, default) -> list[U]:
    """Get a list containing the values for a sequence of keys in a dict.

    If any keys aren't in the dict, a default value is used.
    """
    return [d[v] if v in d else default for v in vals]


def unique(l: Iterable[T]) -> list[T]:
    """Return a list containing the unique values from an iterable with order preserved."""
    try: # make a dict from it, fast and clean
        return list(dict.fromkeys(l))
    except TypeError: # type's unhashable so we've gotta do some more work
        used = []
        return [v for v in l if v not in used and (used.append(v) or True)]


def fillList(l: list[T], n: int, v) -> list[T]:
    """Return a copy of a list filled until with some value until its length's a multiple of n.

    If the list is empty, fill it to the value. (0 doesn't count as a multiple)
    """
    length = len(l)
    padAmount = (n - (length % n)) % n if length > 0 else n
    return list(l) + [v] * padAmount
