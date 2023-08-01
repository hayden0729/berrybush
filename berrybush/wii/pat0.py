# standard imports
from typing import TypeVar
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from .animation import AnimSubfile, groupAnimWriters, calcAnimLen, calcFrameScale
from .bitstruct import BitStruct, Bits
from .brresdict import DictReader, DictWriter
from .common import unique
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
from .subfile import BRRES_SER_T, SubfileSerializer, SubfileReader, SubfileWriter
from . import gx


PAT0_SER_T = TypeVar("PAT0_SER_T", bound="PAT0Serializer")
MAT_ANIM_SER_T = TypeVar("MAT_ANIM_SER_T", bound="MatAnimSerializer")


class AnimCode(BitStruct):
    enabled = Bits(1, bool)
    fixed = Bits(1, bool)
    hasTex = Bits(1, bool)
    hasPlt = Bits(1, bool)


class TexAnim():
    """Contains animation data for changing a texture.

    This is made up of three lists: a list of texture names, a list of palette names, and a
    3-column numpy array storing keyframe indices along with indices into those first two lists.
    """

    def __init__(self, keyframes: np.ndarray = None, length: float = 0,
                 texNames: list[str] = None, pltNames: list[str] = None):
        self.length: float = 0
        self.texNames: list[str] = [] if texNames is None else texNames
        self.pltNames: list[str] = [] if pltNames is None else pltNames
        self.keyframes = np.ndarray((0, 3), dtype=np.float32) if keyframes is None else keyframes

    def __len__(self):
        return len(self.keyframes)

    def __eq__(self, anim):
        # two anims should compare equal if they have the same lengths & keyframe indices,
        # and they have the same texture & palette names for each keyframe
        # (not necessarily the same local list indices - just the same evaluated names!)
        if not (isinstance(anim, TexAnim) and self.length == anim.length):
            return False
        if not np.array_equal(self.keyframes[:, 0], anim.keyframes[:, 0]):
            return False
        if bool(self.texNames) != bool(anim.texNames) or bool(self.pltNames) != bool(anim.pltNames):
            # either this anim has textures and the other doesn't,
            # or the other has palettes and this one doesn't
            return False
        # compare texure names for each frame
        if self.texNames:
            texNames = [self.texNames[t] for t in self.keyframes[:, 1].astype(np.uint16)]
            animTexNames = [anim.texNames[t] for t in anim.keyframes[:, 1].astype(np.uint16)]
            if texNames != animTexNames:
                return False
        # compare palette names for each frame
        if self.pltNames:
            pltNames = [self.pltNames[p] for p in self.keyframes[:, 2].astype(np.uint16)]
            animPltNames = [anim.pltNames[p] for p in anim.keyframes[:, 2].astype(np.uint16)]
            if pltNames != animPltNames:
                return False
        return True


class TexAnimSerializer(Serializer[MAT_ANIM_SER_T, TexAnim]):

    _HEAD_STRCT = Struct(">Hxxf")
    _KF_STRCT = Struct(">fHH")

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o


class TexAnimReader(TexAnimSerializer["MatAnimReader"], Reader, StrPoolReadMixin):

    def __init__(self, parent, offset = 0,
                 code: AnimCode = None, kfs: list[tuple[float, int, int]] = None):
        super().__init__(parent, offset)
        self._data = TexAnim()
        self._kfs: list[tuple[float, int, int]] = [] if kfs is None else kfs
        self._animCode = AnimCode() if code is None else code

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = TexAnim()
        numKfs, frameScale = self._HEAD_STRCT.unpack_from(data, self.offset)
        anim.length = calcAnimLen(frameScale)
        kfOffset = self.offset + self._HEAD_STRCT.size
        kfOffsets = range(kfOffset, kfOffset + numKfs * self._KF_STRCT.size, self._KF_STRCT.size)
        # read keyframes w/ indices into the pat0 master lists
        self._kfs = [self._KF_STRCT.unpack_from(data, o) for o in kfOffsets]
        return self

    def _updateInstance(self):
        super()._updateInstance()
        # get local tex/plt lists and convert keyframes from pat list indices into local indices
        patAnim = self.parentSer.parentSer
        kfs = np.array(self._kfs)
        animCode = self.parentSer.animCodes[self]
        allNamesPat = (patAnim.texNames, patAnim.pltNames)
        allNamesLocal = (self._data.texNames, self._data.pltNames)
        allHasData = (animCode.hasTex, animCode.hasPlt)
        for fIdcs, nPat, nLocal, hasData in zip(kfs.T[1:], allNamesPat, allNamesLocal, allHasData):
            if hasData:
                unqFileIdcs, fIdcs[:] = np.unique(fIdcs.astype(np.uint16), return_inverse=True)
                nLocal[:] = [nPat[t] for t in unqFileIdcs]
        self._data.keyframes = kfs


class TexAnimWriter(TexAnimSerializer["MatAnimWriter"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "PAT0Writer", offset = 0):
        super().__init__(parent, offset)
        self.animCode = AnimCode()

    def fromInstance(self, data: TexAnim):
        super().fromInstance(data)
        animCode = self.animCode
        animCode.hasTex = len(data.texNames) > 0
        animCode.hasPlt = len(data.pltNames) > 0
        animCode.fixed = len(data.keyframes) == 1 and not animCode.hasPlt
        animCode.enabled = animCode.hasTex or animCode.hasPlt
        return self

    def _calcSize(self):
        return self._HEAD_STRCT.size + self._KF_STRCT.size * len(self._data.keyframes)

    def pack(self):
        anim = self._data
        patWriter = self.parentSer.parentSer
        packedHeader = self._HEAD_STRCT.pack(len(anim.keyframes), calcFrameScale(anim.length))
        packedKfs = b""
        for f, (t, p) in zip(anim.keyframes[:, 0:1], anim.keyframes[:, 1:].astype(np.uint16)):
            try:
                texNameIdx = patWriter.texNames[anim.texNames[t]]
            except IndexError:
                texNameIdx = 0
            try:
                pltNameIdx = patWriter.pltNames[anim.pltNames[p]]
            except IndexError:
                pltNameIdx = 0
            packedKfs += self._KF_STRCT.pack(f, texNameIdx, pltNameIdx)
        return packedHeader + packedKfs


class MatAnim():
    """Contains animation data for changing a material's textures."""

    def __init__(self, matName: str = None):
        self.matName = matName
        self.texAnims: dict[int, TexAnim] = {}


class MatAnimSerializer(Serializer[PAT0_SER_T, MatAnim]):

    _HEAD_STRCT = Struct(">iI")
    _ANIM_REF_STRCT = Struct(">i")
    _ANIM_FIX_STRCT = Struct(">HH")


class MatAnimReader(MatAnimSerializer["PAT0Reader"], Reader, StrPoolReadMixin):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        self._texAnims: dict[int, TexAnimReader] = {}
        self.animCodes: dict[TexAnimReader, AnimCode] = {}

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = anim = MatAnim()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        anim.matName = self.readString(data, self.offset + unpackedHeader[0])
        animCode = unpackedHeader[1]
        # read tex anims
        # (offset to each anim is provided after mat anim header; # of anims is based on anim code)
        o = self.offset + self._HEAD_STRCT.size
        for texIdx in range(gx.MAX_TEXTURES):
            texAnimCode = AnimCode((animCode >> (texIdx * 4)) & 0b1111)
            if texAnimCode.enabled:
                texAnim: TexAnim
                if texAnimCode.fixed:
                    fixedTex, fixedPlt = self._ANIM_FIX_STRCT.unpack_from(data, o)
                    texAnim = TexAnimReader(self, kfs=[(0, fixedTex, fixedPlt)])
                    self._texAnims[texIdx] = texAnim
                else:
                    animOffset = self.offset + self._ANIM_REF_STRCT.unpack_from(data, o)[0]
                    texAnim = TexAnimReader(self, animOffset).unpack(data)
                    self._texAnims[texIdx] = texAnim
                self.animCodes[texAnim] = texAnimCode
                o += 4
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.texAnims = {i: a.getInstance() for i, a in self._texAnims.items()}


class MatAnimWriter(MatAnimSerializer["PAT0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "PAT0Writer", offset = 0):
        super().__init__(parent, offset)
        self._animCode = 0
        self.texAnims: dict[int, TexAnimWriter] = {}

    def fromInstance(self, data: MatAnim):
        super().fromInstance(data)
        size = self._HEAD_STRCT.size
        for texIdx, texAnim in self._data.texAnims.items():
            texAnimWriter = TexAnimWriter(self).fromInstance(texAnim)
            self.texAnims[texIdx] = texAnimWriter
            texAnimCode = texAnimWriter.animCode
            self._animCode |= int(texAnimCode) << (texIdx * 4)
            size += self._ANIM_FIX_STRCT.size if texAnimCode.fixed else self._ANIM_REF_STRCT.size
        self._size = size
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        """Pack this writer's main data, describing its format w/ pointers to texture animations."""
        nameOffset = self.stringOffset(self._data.matName) - self.offset
        packedHeader = self._HEAD_STRCT.pack(nameOffset, self._animCode)
        packedRefs = b""
        for texAnimWriter in self.texAnims.values():
            if texAnimWriter.animCode.fixed:
                texAnim = texAnimWriter.getInstance()
                t, p = texAnim.keyframes[0, 1:].astype(np.uint16)
                texIdx = self.parentSer.texNames[texAnim.texNames[t]] if texAnim.texNames else 0
                pltIdx = self.parentSer.pltNames[texAnim.pltNames[p]] if texAnim.pltNames else 0
                packedRefs += self._ANIM_FIX_STRCT.pack(texIdx, pltIdx)
            else:
                packedRefs += self._ANIM_REF_STRCT.pack(texAnimWriter.offset - self.offset)
        return packedHeader + packedRefs


class PAT0(AnimSubfile):
    """BRRES subfile for MDL0 texture swapping animations."""

    _VALID_VERSIONS = (4, )

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.matAnims: list[MatAnim] = []


class PAT0Serializer(SubfileSerializer[BRRES_SER_T, PAT0]):

    DATA_TYPE = PAT0
    FOLDER_NAME = "AnmTexPat(NW4R)"
    MAGIC = b"PAT0"

    _HEAD_STRCT = Struct(">iiiiiiiiHHHH 3x ?")


class PAT0Reader(PAT0Serializer, SubfileReader):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        self._matAnims: dict[str, MatAnimReader] = {}
        self.texNames: list[str] = []
        self.pltNames: list[str] = []

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = PAT0()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        self._data.length = unpackedHeader[8]
        self._data.enableLoop = unpackedHeader[12]
        # read mat anims
        matAnimsOffset = unpackedHeader[0]
        if matAnimsOffset > 0:
            d = DictReader(self, self.offset + matAnimsOffset).unpack(data)
            self._matAnims = d.readEntries(data, MatAnimReader)
        # read texture/palette names used by this pat0
        allNames = (self.texNames, self.pltNames)
        for offset, names, count in zip(unpackedHeader[1:3], allNames, unpackedHeader[10:12]):
            nameOffsets = Struct(">" + "i" * count).unpack_from(data, self.offset + offset)
            names[:] = [self.readString(data, self.offset + offset + o) for o in nameOffsets]
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.matAnims = [matData.getInstance() for matData in self._matAnims.values()]


class PAT0Writer(PAT0Serializer, SubfileWriter):

    def __init__(self, parent, offset = 0):
        super().__init__(parent, offset)
        dictOffset = offset + self._CMN_STRCT.size + self._HEAD_STRCT.size
        self._matAnims: DictWriter[MatAnimWriter] = DictWriter(self, dictOffset)
        self._texAnims: list[list[TexAnimWriter]] = []
        self.texNames: dict[str, int] = {}
        self.pltNames: dict[str, int] = {}

    def getStrings(self):
        return self._matAnims.getStrings() | set(self.texNames) | set(self.pltNames)

    def fromInstance(self, data: PAT0):
        super().fromInstance(data)
        animWriters: dict[str, MatAnimWriter] = {}
        dataOffset = self._matAnims.offset + DictWriter.sizeFromLen(len(data.matAnims))
        texAnims: list[list[TexAnimWriter]] = []
        for a in data.matAnims:
            animWriters[a.matName] = writer = MatAnimWriter(self, dataOffset).fromInstance(a)
            dataOffset += writer.size()
            texAnims.append(list(writer.texAnims.values()))
        self._matAnims.fromInstance(animWriters)
        self._texAnims = groupAnimWriters(texAnims, usePacked=False)
        # grab texture/palette names & associate w/ their indices
        texNames = unique(n for t in self._texAnims for n in t[0].getInstance().texNames)
        pltNames = unique(n for t in self._texAnims for n in t[0].getInstance().pltNames)
        self.texNames = {n: i for i, n in enumerate(texNames)}
        self.pltNames = {n: i for i, n in enumerate(pltNames)}
        # assign offsets for texture anims
        for anims in self._texAnims:
            if not anims[0].animCode.fixed:
                for anim in anims:
                    anim.offset = dataOffset
                dataOffset += anims[0].size()
        # name list size: one 32-bit offset for each entry of each list, doubled since each entry
        # has a name pointer and a subfile pointer (set on runtime, stored as null in file)
        nameListSize = 2 * 4 * (len(self.texNames) + len(self.pltNames))
        self._size = dataOffset + nameListSize - self.offset
        return self

    def _calcSize(self):
        return super()._calcSize()

    def pack(self):
        headSize = self._CMN_STRCT.size + self._HEAD_STRCT.size
        matAnims = self._matAnims.getInstance().values()
        packedData = self._matAnims.pack()
        packedData += b"".join(w.pack() for w in matAnims)
        packedData += b"".join(w[0].pack() for w in self._texAnims if not w[0].animCode.fixed)
        # pack name offsets
        packedNameLists: list[bytes] = []
        nameOffsets: list[int] = []
        curOffset = headSize + len(packedData)
        for names in (self.texNames, self.pltNames):
            st = Struct(">" + "i" * len(names))
            packedNames = st.pack(*(self.stringOffset(t) - curOffset - self.offset for t in names))
            nameOffsets.append(curOffset)
            packedNameLists.append(packedNames)
            curOffset += len(packedNames)
        # create space for texture pointers (not stored in file, but used in wii memory)
        for packedNames in packedNameLists.copy():
            nameOffsets.append(curOffset)
            packedNameLists.append(bytearray(len(packedNames)))
            curOffset += len(packedNames)
        packedHeader = self._HEAD_STRCT.pack(headSize, *nameOffsets, 0,
                                             self.stringOffset(self._data.name) - self.offset, 0,
                                             self._data.length, len(matAnims),
                                             len(self.texNames), len(self.pltNames),
                                             self._data.enableLoop)
        return super().pack() + packedHeader + packedData + b"".join(packedNameLists)
