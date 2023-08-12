# standard imports
from typing import TypeVar
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from .animation import AnimSubfile
from .binaryutils import pad
from .bitstruct import BitStruct, Bits
from .brresdict import DictReader, DictWriter
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
from .subfile import BRRES_SER_T, SubfileSerializer, SubfileReader, SubfileWriter


VIS0_SER_T = TypeVar("VIS0_SER_T", bound="VIS0Serializer")


class AnimCode(BitStruct):
    fixedVal = Bits(1, bool)
    fixed = Bits(1, bool)


class JointAnim():
    """Contains visibility animation data for a joint."""

    def __init__(self, jointName: str = None):
        self.jointName = jointName
        self.frames = np.ndarray((0), bool)


class JointAnimSerializer(Serializer[VIS0_SER_T, JointAnim]):

    _HEAD_STRCT = Struct(">iI")


class JointAnimReader(JointAnimSerializer["VIS0Reader"], Reader, StrPoolReadMixin):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = JointAnim()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        anim.jointName = self.readString(data, self.offset + unpackedHeader[0])
        animCode = AnimCode(unpackedHeader[1])
        if animCode.fixed:
            anim.frames = np.array([animCode.fixedVal])
        else:
            # data is stored as a simple sequence of bits, padded to a multiple of 4 bytes
            numFrames = self.parentSer.length
            dataOffset = self.offset + self._HEAD_STRCT.size
            dataSize = pad(numFrames, 32) // 8
            unpackedData = np.frombuffer(data[dataOffset : dataOffset + dataSize], dtype=np.uint8)
            anim.frames = np.unpackbits(unpackedData)[:numFrames].astype(bool)
        return self


class JointAnimWriter(JointAnimSerializer["VIS0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "VIS0Writer", offset = 0):
        super().__init__(parent, offset)
        self._animCode = AnimCode()

    def fromInstance(self, data: JointAnim):
        super().fromInstance(data)
        self._animCode = AnimCode()
        if np.all(data.frames == data.frames[0]):
            self._animCode.fixed = True
            self._animCode.fixedVal = data.frames[0]
            self._size = self._HEAD_STRCT.size
        else:
            # calculate & store data size (here instead of in _calcSize() for convenience;
            # only have to check if fixed once)
            numFrames = self.parentSer.getInstance().length
            dataSize = pad(numFrames, 32) // 8
            self._size = self._HEAD_STRCT.size + dataSize
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        packedData = b""
        if not self._animCode.fixed:
            numFrames = self.parentSer.getInstance().length
            dataSize = pad(numFrames, 32) // 8
            frames = self._data.frames[:numFrames]
            frames = np.pad(frames, (0, numFrames - len(frames)), "edge")
            packedData = pad(np.packbits(frames).tobytes(), dataSize)
        nameOffset = self.stringOffset(self._data.jointName) - self.offset
        return self._HEAD_STRCT.pack(nameOffset, int(self._animCode)) + packedData


class VIS0(AnimSubfile):
    """BRRES subfile for MDL0 joint visibility animations."""

    _VALID_VERSIONS = (4, )

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.jointAnims: list[JointAnim] = []


class VIS0Serializer(SubfileSerializer[BRRES_SER_T, VIS0]):

    DATA_TYPE = VIS0
    FOLDER_NAME = "AnmVis(NW4R)"
    MAGIC = b"VIS0"

    _HEAD_STRCT = Struct(">iiiiHH 3x ?")


class VIS0Reader(VIS0Serializer, SubfileReader):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        self.length: int = 0

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = VIS0()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        dataOffset = unpackedHeader[0]
        self._data.length = self.length = unpackedHeader[4]
        self._data.enableLoop = unpackedHeader[6]
        if dataOffset > 0:
            d = DictReader(self, self.offset + dataOffset).unpack(data)
            animData = d.readEntries(data, JointAnimReader)
            self._data.jointAnims = [jointData.getInstance() for jointData in animData.values()]
        return self


class VIS0Writer(VIS0Serializer, SubfileWriter):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        dictOffset = offset + self._CMN_STRCT.size + self._HEAD_STRCT.size
        self._jointAnims: DictWriter[JointAnimWriter] = DictWriter(self, dictOffset)

    def getStrings(self):
        return self._jointAnims.getStrings()

    def fromInstance(self, data: VIS0):
        super().fromInstance(data)
        animWriters: dict[str, JointAnimWriter] = {}
        dataOffset = self._jointAnims.offset + DictWriter.sizeFromLen(len(data.jointAnims))
        for a in data.jointAnims:
            animWriters[a.jointName] = writer = JointAnimWriter(self, dataOffset).fromInstance(a)
            dataOffset += writer.size()
        self._jointAnims.fromInstance(animWriters)
        self._size = dataOffset - self.offset
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        packedHeader = self._HEAD_STRCT.pack(
            self._CMN_STRCT.size + self._HEAD_STRCT.size, 0,
            self.stringOffset(self._data.name) - self.offset, 0,
            self._data.length, len(self._data.jointAnims), self._data.enableLoop,
        )
        jointAnimWriters: list[JointAnimWriter] = self._jointAnims.getInstance().values()
        packedData = b"".join(w.pack() for w in jointAnimWriters)
        return super().pack() + packedHeader + self._jointAnims.pack() + packedData
