# standard imports
from typing import TypeVar
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from .animation import (
    Animation, AnimSerializer, AnimSubfile, I4, I6, I12, D1, D2, D4,
    readFrameRefs, packFrameRefs, groupAnimWriters, serializeAnims
)
from .alias import alias
from .bitstruct import BitStruct, Bits
from .brresdict import DictReader, DictWriter
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
from .subfile import BRRES_SER_T, SubfileSerializer, SubfileReader, SubfileWriter
from . import transform as tf


CHR0_SER_T = TypeVar("CHR0_SER_T", bound="CHR0Serializer")


class AnimCode(BitStruct):
    _pad = Bits(1, bool, True)
    identitySRT = Bits(1, bool)
    identityRT = Bits(1, bool)
    identityS = Bits(1, bool)
    isoS = Bits(1, bool) # isometric: same value for each component
    isoR = Bits(1, bool)
    isoT = Bits(1, bool)
    mdlS = Bits(1, bool)
    mdlR = Bits(1, bool)
    mdlT = Bits(1, bool)
    segScaleComp = Bits(1, bool)
    segScaleCompParent = Bits(1, bool)
    hierarchicalScale = Bits(1, bool)
    _fixSX = Bits(1, bool)
    _fixSY = Bits(1, bool)
    _fixSZ = Bits(1, bool)
    _fixRX = Bits(1, bool)
    _fixRY = Bits(1, bool)
    _fixRZ = Bits(1, bool)
    _fixTX = Bits(1, bool)
    _fixTY = Bits(1, bool)
    _fixTZ = Bits(1, bool)
    hasS = Bits(1, bool)
    hasR = Bits(1, bool)
    hasT = Bits(1, bool)
    fmtS = Bits(2, int)
    fmtR = Bits(3, int)
    fmtT = Bits(2, int)

    fixS = alias("_fixSX", "_fixSY", "_fixSZ")
    fixR = alias("_fixRX", "_fixRY", "_fixRZ")
    fixT = alias("_fixTX", "_fixTY", "_fixTZ")


class JointAnim():
    """Contains animation data for a joint.
    
    This data is separated into 3 lists: one for scale, one for rotation, and one for translation.
    If any of these lists are empty, the model's values are used for that transformation. Otherwise,
    the lists must be filled with lists of keyframes - one for each transformation component.
    """

    def __init__(self, jointName: str = None):
        self.jointName = jointName
        self.scale = [Animation(np.array(((0, tf.IDENTITY_S, 0), ))) for _ in range(3)]
        self.rot = [Animation(np.array(((0, tf.IDENTITY_R, 0), ))) for _ in range(3)]
        self.trans = [Animation(np.array(((0, tf.IDENTITY_T, 0), ))) for _ in range(3)]
        self.segScaleComp = False
        self.segScaleCompParent = False # at least one child of this joint has ssc enabled
        self.hierarchicalScale = False


class JointAnimSerializer(Serializer[CHR0_SER_T, JointAnim]):

    _HEAD_STRCT = Struct(">i 4s")
    _KEYFRAME_FMTS: tuple[type[AnimSerializer], ...] = (type(None), I4, I6, I12, D1, D2, D4)


class JointAnimReader(JointAnimSerializer["CHR0Reader"], Reader, StrPoolReadMixin):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = JointAnim()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        anim.jointName = self.readString(data, self.offset + unpackedHeader[0])
        c = AnimCode.unpack(unpackedHeader[1])
        anim.segScaleComp = c.segScaleComp
        anim.segScaleCompParent = c.segScaleCompParent
        anim.hierarchicalScale = c.hierarchicalScale
        o = self._HEAD_STRCT.size
        baseOffset = self.offset
        fmtS, fmtR, fmtT = (self._KEYFRAME_FMTS[fmt] for fmt in (c.fmtS, c.fmtR, c.fmtT))
        l = self.parentSer.length
        o += readFrameRefs(data, baseOffset, o, c.isoS, c.mdlS, c.fixS, c.hasS, fmtS, anim.scale, l)
        o += readFrameRefs(data, baseOffset, o, c.isoR, c.mdlR, c.fixR, c.hasR, fmtR, anim.rot, l)
        o += readFrameRefs(data, baseOffset, o, c.isoT, c.mdlT, c.fixT, c.hasT, fmtT, anim.trans, l)
        return self


class JointAnimWriter(JointAnimSerializer["CHR0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "CHR0Writer", offset = 0):
        super().__init__(parent, offset)
        self._s: list[AnimSerializer | float] = []
        self._r: list[AnimSerializer | float] = []
        self._t: list[AnimSerializer | float] = []
        self._animCode = AnimCode()

    @property
    def _animData(self):
        return self._s + self._r + self._t

    @property
    def animData(self):
        return (d for d in self._animData if isinstance(d, AnimSerializer))

    def _packAnims(self, data: list[Animation], identityVal: int):
        """Process animation data for one transformation.
        
        Return a tuple that contains a bunch of info about this data (e.g., whether it's isometric,
        whether it's fixed, etc).
        """
        iso = useModel = False
        fixed = [False] * len(data)
        animData: list[AnimSerializer | float] = []
        nonFixed: list[Animation] = []
        nonFixedSers: list[AnimSerializer] = []
        if len(data) == 0:
            useModel = True
        else: # there is custom animation data, don't just use model
            iso = all(c == data[0] for c in data[1:]) # iso: all components have the same data
            if iso:
                data = data[:1]
            # get fixed data & prepare non-fixed data for format filtering
            for i, anim in enumerate(data):
                frameVals = set(anim.keyframes[:, 1])
                if len(frameVals) == 1:
                    fixed[i] = True
                    animData.append(frameVals.pop())
                else:
                    animData.append(None)
                    nonFixed.append(anim)
            # until serializeAnims() is optimized, just use i12 for everything
            # nonFixedSers = serializeAnims(nonFixed, list(self._KEYFRAME_FMTS[1:]))
            nonFixedSers = [I12().fromInstance(anim) for anim in nonFixed]
            if iso:
                fixed[:] = fixed[:1] * len(fixed)
        identity = iso and fixed[0] and data[0].keyframes[0, 1] == identityVal
        if identity:
            animData[:] = []
        exists = not identity and not useModel
        fmtIdx = self._KEYFRAME_FMTS.index(type(nonFixedSers[0])) if nonFixed else 0
        # put non-fixed data back in the main list
        nonFixedIdx = 0
        for i, anim in enumerate(animData):
            if anim is None:
                animData[i] = nonFixedSers[nonFixedIdx]
                nonFixedIdx += 1
        return (identity, iso, useModel, fixed, exists, fmtIdx, animData)

    def fromInstance(self, data: JointAnim):
        super().fromInstance(data)
        c = self._animCode
        c.segScaleComp = self._data.segScaleComp
        c.segScaleCompParent = self._data.segScaleCompParent
        c.hierarchicalScale = self._data.hierarchicalScale
        idS, c.isoS, c.mdlS, c.fixS, c.hasS, c.fmtS, self._s = self._packAnims(data.scale, 1)
        idR, c.isoR, c.mdlR, c.fixR, c.hasR, c.fmtR, self._r = self._packAnims(data.rot, 0)
        idT, c.isoT, c.mdlT, c.fixT, c.hasT, c.fmtT, self._t = self._packAnims(data.trans, 0)
        c.identityS = idS
        c.identityRT = idR and idT
        c.identitySRT = idS and idR and idT
        self._size = self._HEAD_STRCT.size + 4 * len(self._animData)
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        """Pack this writer's main data, describing its format w/ pointers to frame data."""
        nameOffset = self.stringOffset(self._data.jointName) - self.offset
        packedHeader = self._HEAD_STRCT.pack(nameOffset, self._animCode.pack())
        return packedHeader + packFrameRefs(self._animData, self.offset)


class CHR0(AnimSubfile):
    """BRRES subfile for MDL0 joint movement animations."""

    _VALID_VERSIONS = (5, )

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.jointAnims: list[JointAnim] = []
        self.mtxGen: type[tf.MtxGenerator] = tf.StdMtxGen3D


class CHR0Serializer(SubfileSerializer[BRRES_SER_T, CHR0]):

    DATA_TYPE = CHR0
    FOLDER_NAME = "AnmChr(NW4R)"
    MAGIC = b"CHR0"

    _HEAD_STRCT = Struct(">iiiiHH 3x ? i")
    _MTX_GEN_TYPES = (tf.StdMtxGen3D, tf.XSIMtxGen3D, tf.MayaMtxGen3D)


class CHR0Reader(CHR0Serializer, SubfileReader):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = CHR0()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        dataOffset = unpackedHeader[0]
        self._data.length = unpackedHeader[4]
        self._data.enableLoop = unpackedHeader[6]
        self._data.mtxGen = self._MTX_GEN_TYPES[unpackedHeader[7]]
        if dataOffset > 0:
            d = DictReader(self, self.offset + dataOffset).unpack(data)
            animData = d.readEntries(data, JointAnimReader)
            self._data.jointAnims = [jointData.getInstance() for jointData in animData.values()]
        return self

    @property
    def length(self):
        return self._data.length


class CHR0Writer(CHR0Serializer, SubfileWriter):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        dictOffset = offset + self._CMN_STRCT.size + self._HEAD_STRCT.size
        self._jointAnims: DictWriter[JointAnimWriter] = DictWriter(self, dictOffset)
        self._animData: list[list[AnimSerializer]] = []

    def getStrings(self):
        return self._jointAnims.getStrings()

    def fromInstance(self, data: CHR0):
        super().fromInstance(data)
        animWriters: dict[str, JointAnimWriter] = {}
        dataOffset = self._jointAnims.offset + DictWriter.sizeFromLen(len(data.jointAnims))
        for a in data.jointAnims:
            animWriters[a.jointName] = writer = JointAnimWriter(self, dataOffset).fromInstance(a)
            dataOffset += writer.size()
        self._animData = groupAnimWriters([list(a.animData) for a in animWriters.values()], False)
        for anims in self._animData:
            for anim in anims:
                anim.offset = dataOffset
            dataOffset += anims[0].size()
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
            self._MTX_GEN_TYPES.index(self._data.mtxGen)
        )
        jointAnimWriters: list[JointAnimWriter] = self._jointAnims.getInstance().values()
        packedData = b"".join(w.pack() for w in jointAnimWriters)
        packedData += b"".join(w[0].pack() for w in self._animData)
        return super().pack() + packedHeader + self._jointAnims.pack() + packedData
