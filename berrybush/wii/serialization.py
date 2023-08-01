# standard imports
from abc import abstractmethod, ABC
from typing import Generic, TypeVar


S_PARENT_T = TypeVar("S_PARENT_T", bound="Serializable")
S_PARENT_T = TypeVar("S_PARENT_T", bound="Readable")
S_PARENT_T = TypeVar("S_PARENT_T", bound="Writable")
S_DATA_T = TypeVar("S_DATA_T")


class Serializable(ABC, Generic[S_PARENT_T]):
    """Object that supports some degree of conversion to/from bytes.

    Has a serializable parent that makes it possible to deal with references pointing beyond this
    object's scope.
    """

    def __init__(self, parent: S_PARENT_T = None):
        self.parentSer = parent


class Readable(Serializable[S_PARENT_T]):
    """Object that can be unpacked from bytes."""

    @abstractmethod
    def unpack(self, data: bytes):
        """Unpack data from bytes and store it in this object, which is returned."""
        return self


class Writable(Serializable[S_PARENT_T]):
    """Object that can be packed to bytes."""

    @abstractmethod
    def size(self) -> int:
        """Size of this serializable object, in bytes."""

    @abstractmethod
    def pack(self) -> bytes:
        """Pack the data stored in this object to bytes."""


class AddressedSerializable(Serializable[S_PARENT_T]):
    """Serializable object with an offset indicating its location."""

    def __init__(self, parent: S_PARENT_T = None, offset = 0):
        super().__init__(parent)
        self._offset = offset

    @property
    def offset(self):
        """Absolute address of this serializer."""
        return self._offset

class Serializer(AddressedSerializable[S_PARENT_T], Generic[S_PARENT_T, S_DATA_T]):
    """Serializable helper to convert some BRRES type to/from bytes."""

    DATA_TYPE: type[S_DATA_T]

    def __init__(self, parent: S_PARENT_T = None, offset = 0):
        super().__init__(parent, offset)
        self._data: S_DATA_T = None

    def getInstance(self) -> S_DATA_T:
        """Get this serializer's instance of the data type it manages."""
        return self._data


# general guidelines for implementing the serializer methods in subclasses:
# (note that these are general, and exceptions can & should be made for middlemen like dicts)
# - unpack: keep absolute offsets except strings; avoid parent access & never call getInstance
# - _updateInstance: update instance based on stored offsets & parent data (not parent getInstance)
# - fromInstance: create info from instance w/ direct data references; don't access parent
# - pack: get abs offsets & pack based on parent (data or getInstance, but use data when possible)


class Reader(Readable[S_PARENT_T], Serializer[S_PARENT_T, S_DATA_T]):
    """Helper to read a BRRES type from bytes."""

    def __init__(self, parent: S_PARENT_T = None, offset = 0):
        super().__init__(parent, offset)
        self._dataCached = False

    @abstractmethod
    def unpack(self, data: bytes):
        """Unpack data from bytes and store it in this reader, which is returned."""
        super().unpack(data)
        self._data = None
        self._dataCached = False
        return self

    def _updateInstance(self):
        """Update this reader's data instance based on its current state."""

    def getInstance(self):
        """Get this reader's instance of the data type it manages.

        This instance is created on unpack, and gets updated based on the reader's state the first
        time getInstance() is called. All following getInstance() calls return the same object
        without any modifications, unless unpack() is called again (which creates a new instance
        and restarts the cycle).
        """
        if not self._dataCached:
            self._dataCached = True
            self._updateInstance()
        return super().getInstance()


class Writer(Writable[S_PARENT_T], Serializer[S_PARENT_T, S_DATA_T]):
    """Helper to write a BRRES type to bytes."""

    def __init__(self, parent: S_PARENT_T = None, offset = 0):
        super().__init__(parent, offset)
        self._size = 0

    def fromInstance(self, data: S_DATA_T):
        """Update this writer based on an instance of the data it serializes & return the writer.

        Note that after calling this, the provided instance is guaranteed to be the one returned by
        getInstance().
        """
        self._data = data
        self._size = self._calcSize()
        return self

    @abstractmethod
    def _calcSize(self) -> int:
        """Calculate the size, in bytes, of the data stored in this writer.

        This size is ultimately stored in self._size during fromInstance(), and the size() method
        simply returns self._size. You shouldn't ever have to call this method yourself.

        If you override fromInstance() and add your own size calculation there, you can just call
        super()._calcSize() here.
        """
        return 0

    def size(self):
        return self._size

    @abstractmethod
    def pack(self) -> bytes:
        if self._size <= 0:
            raise RuntimeError("Cannot pack writable object with size <= 0")
        return b""


class StrPoolReadMixin():
    """Mixin for a readable object that uses a BRRES string pool."""

    def readString(self, data: bytes, offset: int) -> str:
        """Read & return a string from the master BRRES string pool."""
        return self.parentSer.readString(data, offset)


class StrPoolWriteMixin():
    """Mixin for a writable object that uses a BRRES string pool."""

    def getStrings(self) -> set[str]:
        """Return a set of strings this data uses for the master BRRES string pool."""
        return set()

    def stringOffset(self, string: str):
        """Absolute offset of a string in the master BRRES string pool. 0 if not found."""
        return self.parentSer.stringOffset(string)
