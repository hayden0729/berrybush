# standard imports
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from .subfile import BRRES_SER_T, Subfile, SubfileSerializer, SubfileReader, SubfileWriter
from .tex0 import ListableImageFormat, IA8, RGB565, RGB5A3, TEX0


class PLT0(Subfile):
    """BRRES subfile for texture palettes."""

    _VALID_VERSIONS = (1, 3)

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.fmt: ListableImageFormat = None
        self.colors = np.ndarray((0, 0, 4))

    def isCompatible(self, tex: TEX0):
        """Whether this palette can be used with a palette image."""
        return tex.isPaletteIndices and np.max(tex.images) < len(self.colors)

    def __len__(self):
        return len(self.colors)


class PLT0Serializer(SubfileSerializer[BRRES_SER_T, PLT0]):

    DATA_TYPE = PLT0
    FOLDER_NAME = "Palettes(NW4R)"
    MAGIC = b"PLT0"

    _IMG_FORMATS: tuple[type[ListableImageFormat]] = (IA8, RGB565, RGB5A3)
    _HEAD_STRCT = Struct(">iiIHxxii 24x")


class PLT0Reader(PLT0Serializer, SubfileReader):

    def unpack(self, data: bytes):
        super().unpack(data)
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        plt = self._data
        plt.fmt: type[ListableImageFormat] = self._IMG_FORMATS[unpackedHeader[2]]
        if plt.fmt is None:
            raise NotImplementedError("Unsupported image format detected")
        numEntries = unpackedHeader[3]
        dataOffset = unpackedHeader[0]
        if dataOffset > 0:
            plt.colors = plt.fmt.importList(data[self.offset + dataOffset:], numEntries)
        return self


class PLT0Writer(PLT0Serializer, SubfileWriter):

    def _calcSize(self):
        listSize = self._data.fmt.listSize(len(self._data.colors))
        return self._CMN_STRCT.size + self._HEAD_STRCT.size + listSize

    def pack(self):
        plt = self._data
        packedHeader = self._HEAD_STRCT.pack(self._CMN_STRCT.size + self._HEAD_STRCT.size,
                                             self.stringOffset(self._data.name) - self.offset,
                                             self._IMG_FORMATS.index(plt.fmt), len(plt.colors),
                                             0, 0)
        return super().pack() + packedHeader + plt.fmt.exportList(plt.colors)
