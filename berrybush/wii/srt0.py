# standard imports
from typing import TypeVar
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from .animation import (
    Animation, AnimSubfile, I12, readFrameRefs, packFrameRefs, groupAnimWriters
)
from .alias import alias
from .bitstruct import BitStruct, Bits
from .brresdict import DictReader, DictWriter
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
from .subfile import BRRES_SER_T, SubfileSerializer, SubfileReader, SubfileWriter
from . import gx, transform as tf


SRT0_SER_T = TypeVar("SRT0_SER_T", bound="SRT0Serializer")
MAT_ANIM_SER_T = TypeVar("MAT_ANIM_SER_T", bound="MatAnimSerializer")


class AnimCode(BitStruct):
    _pad0 = Bits(1, bool, True)
    identityS = Bits(1, bool)
    identityR = Bits(1, bool)
    identityT = Bits(1, bool)
    isoS = Bits(1, bool)
    _fixSX = Bits(1, bool)
    _fixSY = Bits(1, bool)
    _fixR = Bits(1, bool)
    _fixTX = Bits(1, bool)
    _fixTY = Bits(1, bool)
    _pad1 = Bits(22, int)

    fixS = alias("_fixSX", "_fixSY")
    fixR = alias("_fixR", forceList=True)
    fixT = alias("_fixTX", "_fixTY")


class TexAnim():
    """Contains animation data for a texture.

    This data is separated into 3 lists: one for scale, one for rotation, and one for translation.
    If any of these lists are empty, the model's values are used for that transformation.
    """

    def __init__(self):
        self.scale = [Animation(np.array(((0, tf.IDENTITY_S, 0), ))) for _ in range(2)]
        self.rot = [Animation(np.array(((0, tf.IDENTITY_R, 0), )))]
        self.trans = [Animation(np.array(((0, tf.IDENTITY_T, 0), ))) for _ in range(2)]


class TexAnimSerializer(Serializer[MAT_ANIM_SER_T, TexAnim]):

    _HEAD_STRCT = Struct(">4s")


class TexAnimReader(TexAnimSerializer["MatAnimReader"], Reader, StrPoolReadMixin):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = TexAnim()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        c = AnimCode.unpack(unpackedHeader[0])
        o = self.offset + self._HEAD_STRCT.size
        hasS, hasR, hasT = not c.identityS, not c.identityR, not c.identityT
        o += readFrameRefs(data, None, o, c.isoS, False, c.fixS, hasS, I12, anim.scale)
        o += readFrameRefs(data, None, o, False, False, c.fixR, hasR, I12, anim.rot)
        o += readFrameRefs(data, None, o, False, False, c.fixT, hasT, I12, anim.trans)
        return self


class TexAnimWriter(TexAnimSerializer["MatAnimWriter"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "MatAnimWriter", offset = 0):
        super().__init__(parent, offset)
        self._s: list[I12 | float] = []
        self._r: list[I12 | float] = []
        self._t: list[I12 | float] = []
        self._animCode = AnimCode()

    @property
    def animData(self):
        return self._s + self._r + self._t

    @property
    def numKeyframes(self):
        return max(len(d.getInstance().keyframes) for d in self.animData if isinstance(d, I12))

    def _packAnims(self, data: list[Animation], isScale = False):
        """Process animation data for one transformation.
        
        Return a tuple that contains a bunch of info about this data (e.g., whether it's isometric,
        whether it's fixed, etc).
        """
        fixed = [False] * len(data)
        animData: list[I12 | float] = []
        # iso can only actually be written for scale, but knowing if data is iso is still useful
        iso = all(c == data[0] for c in data)
        writeIso = iso and isScale
        for i, anim in enumerate(data if not writeIso else data[:1]):
            if len(anim.keyframes) == 1:
                fixed[i] = True
                animData.append(anim.keyframes[0, 1])
            else:
                animData.append(I12().fromInstance(anim))
        if iso:
            fixed[:] = fixed[:1] * len(fixed)
        identityVal = 1 if isScale else 0
        identity = iso and fixed[0] and data[0].keyframes[0, 1] == identityVal
        if identity:
            animData[:] = []
        return (identity, writeIso, fixed, animData)

    def fromInstance(self, data: TexAnim):
        super().fromInstance(data)
        c = self._animCode
        c.identityS, c.isoS, c.fixS, self._s = self._packAnims(data.scale, True)
        c.identityR, _, c.fixR, self._r = self._packAnims(data.rot)
        c.identityT, _, c.fixT, self._t = self._packAnims(data.trans)
        self._size = self._HEAD_STRCT.size + 4 * len(self.animData)
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        """Pack this writer's main data, describing its format w/ pointers to frame data."""
        packedHeader = self._animCode.pack()
        frameRefOffset = self.offset + self._HEAD_STRCT.size
        return packedHeader + packFrameRefs(self.animData, frameRefOffset, individualRelative=True)


class MatAnim():
    """Contains animation data for a material's texture transforms.

    This data is separated into 2 lists: one for the regular textures, and one for the indirect
    texture transforms. Each list has an entry for each texture/transform. If a texture/transform
    has no animation, the entry is None.
    """

    def __init__(self, matName: str = None):
        self.matName = matName
        self.texAnims: dict[int, TexAnim] = {}
        self.indAnims: dict[int, TexAnim] = {}


class MatAnimSerializer(Serializer[SRT0_SER_T, MatAnim]):

    _HEAD_STRCT = Struct(">iII")


class MatAnimReader(MatAnimSerializer["SRT0Reader"], Reader, StrPoolReadMixin):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = MatAnim()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        anim.matName = self.readString(data, self.offset + unpackedHeader[0])
        o = self.offset + self._HEAD_STRCT.size
        # read texture data & indirect transform data
        anims = (self._data.texAnims, self._data.indAnims)
        maxs = (gx.MAX_TEXTURES, gx.MAX_INDIRECT_MTCS)
        for animDict, maxSlots, usedFlag in zip(anims, maxs, unpackedHeader[1:3]):
            for i in range(maxSlots):
                if usedFlag & (1 << i):
                    animOffset = self.offset + int.from_bytes(data[o : o + 4], "big")
                    o += 4
                    animDict[i] = TexAnimReader(self, animOffset).unpack(data).getInstance()
        return self


class MatAnimWriter(MatAnimSerializer["SRT0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "SRT0Writer", offset = 0):
        super().__init__(parent, offset)
        self._anims: list[TexAnimWriter] = []

    @property
    def animData(self):
        """All animation writers used by the texture anim writers of this material anim writer."""
        return (d for w in self._anims for d in w.animData if isinstance(d, I12))

    def fromInstance(self, data: MatAnim):
        super().fromInstance(data)
        texAnims = sorted(data.texAnims.items(), key=lambda item: item[0])
        indAnims = sorted(data.indAnims.items(), key=lambda item: item[0])
        anims = texAnims + indAnims
        animWriterOffset = self.offset + self._HEAD_STRCT.size + 4 * len(anims)
        for idx, anim in anims:
            writer = TexAnimWriter(self, animWriterOffset).fromInstance(anim)
            self._anims.append(writer)
            animWriterOffset += writer.size()
        self._size = animWriterOffset - self.offset
        return self

    def _calcSize(self):
        return super()._calcSize()

    @classmethod
    def _getUsedFlags(cls, anims: dict[int, TexAnim], maxSlots: int):
        """Get flags for the present slots in a dict of texture animations."""
        flags = 0
        for i in range(maxSlots):
            if i in anims:
                flags |= (1 << i)
        return flags

    def pack(self):
        nameOffset = self.stringOffset(self._data.matName) - self.offset
        usedTexFlags = self._getUsedFlags(self._data.texAnims, gx.MAX_TEXTURES)
        usedIndFlags = self._getUsedFlags(self._data.indAnims, gx.MAX_INDIRECT_MTCS)
        packedHeader = self._HEAD_STRCT.pack(nameOffset, usedTexFlags, usedIndFlags)
        anmOffsets = b"".join(int(a.offset - self.offset).to_bytes(4, "big") for a in self._anims)
        return packedHeader + anmOffsets + b"".join(a.pack() for a in self._anims)


class SRT0(AnimSubfile):
    """BRRES subfile for MDL0 texture movement animations."""

    _VALID_VERSIONS = (5, )

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.matAnims: list[MatAnim] = []
        self.mtxGen: type[tf.MtxGenerator] = tf.MayaMtxGen2D


class SRT0Serializer(SubfileSerializer[BRRES_SER_T, SRT0]):

    DATA_TYPE = SRT0
    FOLDER_NAME = "AnmTexSrt(NW4R)"
    MAGIC = b"SRT0"

    _HEAD_STRCT = Struct(">iiiiHHi 3x ?")
    _MTX_GEN_TYPES = (tf.MayaMtxGen2D, tf.XSIMtxGen2D, tf.MaxMtxGen2D)


class SRT0Reader(SRT0Serializer, SubfileReader):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = SRT0()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        dataOffset = unpackedHeader[0]
        self._data.length = unpackedHeader[4]
        self._data.mtxGen = self._MTX_GEN_TYPES[unpackedHeader[6]]
        self._data.enableLoop = unpackedHeader[7]
        if dataOffset > 0:
            d = DictReader(self, self.offset + dataOffset).unpack(data)
            animData = d.readEntries(data, MatAnimReader)
            self._data.matAnims = [matData.getInstance() for matData in animData.values()]
        return self


class SRT0Writer(SRT0Serializer, SubfileWriter):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        dictOffset = offset + self._CMN_STRCT.size + self._HEAD_STRCT.size
        self._matAnims: DictWriter[MatAnimWriter] = DictWriter(self, dictOffset)
        self._animData: list[list[I12]] = []

    def getStrings(self):
        return self._matAnims.getStrings()

    def fromInstance(self, data: SRT0):
        super().fromInstance(data)
        animWriters: dict[str, MatAnimWriter] = {}
        dataOffset = self._matAnims.offset + DictWriter.sizeFromLen(len(data.matAnims))
        for a in data.matAnims:
            animWriters[a.matName] = writer = MatAnimWriter(self, dataOffset).fromInstance(a)
            dataOffset += writer.size()
        self._matAnims.fromInstance(animWriters)
        self._animData = groupAnimWriters([list(a.animData) for a in animWriters.values()])
        for anims in self._animData:
            for anim in anims:
                anim.offset = dataOffset
            dataOffset += anims[0].size()
        self._size = dataOffset - self.offset
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        packedHeader = self._HEAD_STRCT.pack(
            self._CMN_STRCT.size + self._HEAD_STRCT.size, 0,
            self.stringOffset(self._data.name) - self.offset, 0,
            self._data.length, len(self._data.matAnims),
            self._MTX_GEN_TYPES.index(self._data.mtxGen),  self._data.enableLoop
        )
        matAnimWriters: list[MatAnimWriter] = self._matAnims.getInstance().values()
        packedData = b"".join(w.pack() for w in matAnimWriters)
        packedData += b"".join(w[0].pack() for w in self._animData)
        return super().pack() + packedHeader + self._matAnims.pack() + packedData
