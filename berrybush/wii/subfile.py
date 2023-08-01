# standard imports
from abc import abstractmethod
from struct import Struct
from typing import TypeVar, TYPE_CHECKING
# internal imports
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
# special typing imports
if TYPE_CHECKING:
    from brres import BRRESSerializer, BRRESReader, BRRESWriter
else:
    BRRESSerializer = BRRESReader = BRRESWriter = object


BRRES_SER_T = TypeVar("BRRES_SER_T", bound=BRRESSerializer)
FILE_T = TypeVar("FILE_T", bound="Subfile")


class SubfileVersionError(Exception):
    """Attempted to set an invalid subfile version"""

    def __init__(self, subfileType: type["Subfile"], ver: int):
        super().__init__(f"Version {ver} is invalid for {subfileType.__name__} subfiles")


class Subfile():
    """Component of a BRRES file for storing an individual resource (model, texture, etc)."""

    _VALID_VERSIONS: tuple[int, ...] # versions supported so far

    def __init__(self, name: str = None, version = -1):
        self.name = name
        self.version = version

    @property
    def version(self) -> int:
        """Subfile format version.

        Must be valid for this subfile type. Setting to -1 makes it the most recent option."""
        return self._version

    @version.setter
    def version(self, version):
        if version == -1: # -1 indicates to just use most recent version
            version = self._VALID_VERSIONS[-1]
        if version not in self._VALID_VERSIONS:
            raise SubfileVersionError(type(self), version)
        self._version = version


class SubfileSerializer(Serializer[BRRES_SER_T, FILE_T]):
    """Serializer for a BRRES subfile."""
    FOLDER_NAME: str # name of brres folder used for storing this subfile type
    MAGIC: bytes # 4 bytes used for file type identification
    _CMN_STRCT = Struct(">4s IIi") # common header at start of every subfile


class SubfileReader(SubfileSerializer[BRRESReader, FILE_T], Reader, StrPoolReadMixin):
    """Reader for a BRRES subfile."""

    @abstractmethod
    def unpack(self, data: bytes):
        super().unpack(data)
        unpackedHeader = self._CMN_STRCT.unpack_from(data, self._offset)
        self._data = self.DATA_TYPE(version=unpackedHeader[2])
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.name = self.parentSer.fileName(self)


class SubfileWriter(SubfileSerializer[BRRESWriter, FILE_T], Writer, StrPoolWriteMixin):
    """Writer for a BRRES subfile."""

    @abstractmethod
    def pack(self):
        head = self._CMN_STRCT.pack(self.MAGIC, self._size, self._data.version, -self._offset)
        return super().pack() + head
