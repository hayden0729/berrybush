# standard imports
from typing import TypeVar
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from .alias import alias
from .animation import AnimSubfile, groupAnimWriters
from .binaryutils import maxBitVal
from .bitstruct import BitStruct, Bits
from .brresdict import DictReader, DictWriter
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
from .subfile import BRRES_SER_T, SubfileSerializer, SubfileReader, SubfileWriter
from . import gx


CLR0_SER_T = TypeVar("CLR0_SER_T", bound="CLR0Serializer")


class AnimCode(BitStruct):
    _hasDif0 = Bits(1, bool)
    _fixDif0 = Bits(1, bool)
    _hasDif1 = Bits(1, bool)
    _fixDif1 = Bits(1, bool)
    _hasAmb0 = Bits(1, bool)
    _fixAmb0 = Bits(1, bool)
    _hasAmb1 = Bits(1, bool)
    _fixAmb1 = Bits(1, bool)
    _hasStand0 = Bits(1, bool)
    _fixStand0 = Bits(1, bool)
    _hasStand1 = Bits(1, bool)
    _fixStand1 = Bits(1, bool)
    _hasStand2 = Bits(1, bool)
    _fixStand2 = Bits(1, bool)
    _hasConst0 = Bits(1, bool)
    _fixConst0 = Bits(1, bool)
    _hasConst1 = Bits(1, bool)
    _fixConst1 = Bits(1, bool)
    _hasConst2 = Bits(1, bool)
    _fixConst2 = Bits(1, bool)
    _hasConst3 = Bits(1, bool)
    _fixConst3 = Bits(1, bool)
    _pad = Bits(10, int)

    hasClrs = alias(
        "_hasDif0", "_hasDif1", "_hasAmb0", "_hasAmb1", "_hasStand0", "_hasStand1", "_hasStand2",
        "_hasConst0", "_hasConst1", "_hasConst2", "_hasConst3"
    )
    fixClrs = alias(
        "_fixDif0", "_fixDif1", "_fixAmb0", "_fixAmb1", "_fixStand0", "_fixStand1", "_fixStand2",
        "_fixConst0", "_fixConst1", "_fixConst2", "_fixConst3"
    )


class RegAnim():
    """Contains animation data for a color register."""

    def __init__(self, colors: np.ndarray, mask: np.ndarray):
        self.colors = colors
        self.mask = mask
        """Mask that lets you ignore parts of the color animation data, which is stored as RGBA8."""

    def __len__(self):
        return len(self.colors)

    def __eq__(self, other):
        return (
            isinstance(other, RegAnim)
            and np.all(self.colors == other.colors)
            and np.all(self.mask == other.mask)
        )

    @property
    def normalized(self):
        """Colors for this animation, normalized from 0-1.

        Colors can be retrieved and set through this property, or directly through "colors".
        """
        return self.colors / maxBitVal(8)

    @normalized.setter
    def normalized(self, colors: np.ndarray):
        self.colors = (colors * maxBitVal(8) + .5).astype(np.uint8)


class RegAnimWriter(Writer["MatAnimWriter", RegAnim]):

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def _calcSize(self):
        return self._data.colors.size

    def pack(self):
        return self._data.colors.tobytes()


class MatAnim():
    """Contains animation data for a material's color registers."""

    def __init__(self, matName: str = None):
        self.matName = matName
        self.difRegs: dict[int, RegAnim] = {}
        self.ambRegs: dict[int, RegAnim] = {}
        self.standRegs: dict[int, RegAnim] = {}
        self.constRegs: dict[int, RegAnim] = {}

    @property
    def allRegs(self):
        """Animations for all registers that can be animated by this animation.

        They appear in the order (difRegs, ambRegs, standRegs, constRegs), in one flattened
        tuple. Non-animated registers are represented with None.
        """
        difRegs = tuple(self.difRegs.get(i) for i in range(gx.MAX_CLR_ATTRS))
        ambRegs = tuple(self.ambRegs.get(i) for i in range(gx.MAX_CLR_ATTRS))
        standRegs = tuple(self.standRegs.get(i) for i in range(gx.MAX_TEV_STAND_COLORS))
        constRegs = tuple(self.constRegs.get(i) for i in range(gx.MAX_TEV_CONST_COLORS))
        return difRegs + ambRegs + standRegs + constRegs

    def setRegAnim(self, i: int, anim: RegAnim | None):
        """Set a register animation via an index into allRegs."""
        regDicts = (self.difRegs, self.ambRegs, self.standRegs, self.constRegs)
        mxs = (gx.MAX_CLR_ATTRS, gx.MAX_CLR_ATTRS, gx.MAX_TEV_STAND_COLORS, gx.MAX_TEV_CONST_COLORS)
        for regDict, maxSlots in zip(regDicts, mxs):
            if i < maxSlots:
                regDict[i] = anim
                return
            else:
                i -= maxSlots
        raise IndexError("Register index out of range")


class MatAnimSerializer(Serializer[CLR0_SER_T, MatAnim]):

    _HEAD_STRCT = Struct(">i 4s")
    _REG_ENTRY_STRCT = Struct(">4s 4s")


class MatAnimReader(MatAnimSerializer["CLR0Reader"], Reader, StrPoolReadMixin):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = MatAnim()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        anim.matName = self.readString(data, self.offset + unpackedHeader[0])
        c = AnimCode.unpack(unpackedHeader[1])
        dataOffset = self.offset + self._HEAD_STRCT.size
        for i, (has, fixed) in enumerate(zip(c.hasClrs, c.fixClrs)):
            if has:
                entryData = self._REG_ENTRY_STRCT.unpack_from(data, dataOffset)
                mask = np.frombuffer(entryData[0], dtype=np.uint8)
                colorData = entryData[1] # either fixed value or anim offset
                if not fixed:
                    # note: offset is relative to exactly where the offset is stored
                    # (4 bytes after data offset)
                    animOffset = dataOffset + 4 + int.from_bytes(colorData, "big")
                    # another note: do length + 1 bc there's an entry for both start and end edges
                    colorData = data[animOffset : animOffset + (self.parentSer.length + 1) * 4]
                colors = np.frombuffer(colorData, dtype=np.uint8).reshape(-1, 4)
                anim.setRegAnim(i, RegAnim(colors, mask))
                dataOffset += self._REG_ENTRY_STRCT.size
        return self


class MatAnimWriter(MatAnimSerializer["CLR0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "CLR0Writer", offset = 0):
        super().__init__(parent, offset)
        self._regs: list[RegAnimWriter] = []
        self._animCode = AnimCode()

    @property
    def regs(self):
        """Color data for this animation's non-fixed registers."""
        return [r for r in self._regs if len(r.getInstance().colors) > 1]

    def fromInstance(self, data: MatAnim):
        super().fromInstance(data)
        c = self._animCode
        c.hasClrs = (a is not None for a in data.allRegs)
        c.fixClrs = (a is not None and len(a.colors) == 1 for a in data.allRegs)
        self._regs = [RegAnimWriter(self).fromInstance(a) for a in data.allRegs if a is not None]
        self._size = self._HEAD_STRCT.size + self._REG_ENTRY_STRCT.size * len(self._regs)
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        """Pack this writer's main data, describing its format w/ pointers to frame data."""
        nameOffset = self.stringOffset(self._data.matName) - self.offset
        packedHeader = self._HEAD_STRCT.pack(nameOffset, self._animCode.pack())
        packedData = b""
        offset = self.offset + self._HEAD_STRCT.size + 4
        for w in self._regs:
            anim = w.getInstance()
            data = int(w.offset - offset).to_bytes(4, "big") if len(anim) > 1 else w.pack()
            packedData += self._REG_ENTRY_STRCT.pack(anim.mask.tobytes(), data)
            offset += self._REG_ENTRY_STRCT.size
        return packedHeader + packedData


class CLR0(AnimSubfile):
    """BRRES subfile for MDL0 material color register animations."""

    _VALID_VERSIONS = (4, )

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.matAnims: list[MatAnim] = []


class CLR0Serializer(SubfileSerializer[BRRES_SER_T, CLR0]):

    DATA_TYPE = CLR0
    FOLDER_NAME = "AnmClr(NW4R)"
    MAGIC = b"CLR0"

    _HEAD_STRCT = Struct(">iiiiHH 3x ?")


class CLR0Reader(CLR0Serializer, SubfileReader):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        self.length: int = 0

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = CLR0()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        dataOffset = unpackedHeader[0]
        self._data.length = self.length = unpackedHeader[4]
        self._data.enableLoop = unpackedHeader[6]
        if dataOffset > 0:
            d = DictReader(self, self.offset + dataOffset).unpack(data)
            animData = d.readEntries(data, MatAnimReader)
            self._data.matAnims = [matData.getInstance() for matData in animData.values()]
        return self


class CLR0Writer(CLR0Serializer, SubfileWriter):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        dictOffset = offset + self._CMN_STRCT.size + self._HEAD_STRCT.size
        self._matAnims: DictWriter[MatAnimWriter] = DictWriter(self, dictOffset)
        self._clrAnims: list[list[RegAnimWriter]] = []

    def getStrings(self):
        return self._matAnims.getStrings()

    def fromInstance(self, data: CLR0):
        super().fromInstance(data)
        animWriters: dict[str, MatAnimWriter] = {}
        dataOffset = self._matAnims.offset + DictWriter.sizeFromLen(len(data.matAnims))
        for a in data.matAnims:
            animWriters[a.matName] = writer = MatAnimWriter(self, dataOffset).fromInstance(a)
            dataOffset += writer.size()
        self._clrAnims = groupAnimWriters([a.regs for a in animWriters.values()])
        for anims in self._clrAnims:
            for anim in anims:
                anim.offset = dataOffset
            dataOffset += anims[0].size()
        self._matAnims.fromInstance(animWriters)
        self._size = dataOffset - self.offset
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        packedHeader = self._HEAD_STRCT.pack(
            self._CMN_STRCT.size + self._HEAD_STRCT.size, 0,
            self.stringOffset(self._data.name) - self.offset, 0,
            self._data.length, len(self._data.matAnims), self._data.enableLoop,
        )
        matAnimWriters: list[MatAnimWriter] = self._matAnims.getInstance().values()
        packedData = b"".join(w.pack() for w in matAnimWriters)
        packedData += b"".join(w[0].pack() for w in self._clrAnims)
        return super().pack() + packedHeader + self._matAnims.pack() + packedData
