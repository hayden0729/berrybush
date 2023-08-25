# standard imports
from abc import abstractmethod
from functools import cache
from struct import Struct
from typing import TypeVar
# 3rd party imports
import numpy as np
# internal imports
from .binaryutils import maxBitVal, pad
from .serialization import Reader, Writer
from .subfile import Subfile


WRITER_T = TypeVar("WRITER_T", bound=Writer)


# references to frame data are stored in animations through integer offsets. alternatively, if
# data is fixed, the data is stored directly as floats.
# throughout this file, i call this whole structure "frame references" - not the best name, since
# the fixed values are just values, not references, but it's a distinct name so i'm sticking w/ it
FRAME_PTR_STRCT = Struct(">I")
FRAME_FIXED_STRCT = Struct(">f")


ONE_MINUS_EPS = np.float32(1) - np.finfo(np.float32).eps


@cache
def calcAnimLen(frameScale: float):
    """Calculate an animation's length from its normalized frame scale."""
    return ONE_MINUS_EPS / np.float32(frameScale)


def calcFrameScale(animLen: float):
    """Calculate an animation's normalized frame scale from its length."""
    return calcAnimLen(animLen) # (...it's just the same calculation as calcAnimLen)


def hermite(x1, y1, t1, x2, y2, t2, newx):
    """Hermite interpolation for two points. Return the Y value for some X between them."""
    # https://www.cubic.org/docs/hermite.htm
    span = x2 - x1
    fac = (newx - x1) / span
    fac2 = fac ** 2
    fac3 = fac ** 3
    h1 = 2 * fac3 - 3 * fac2 + 1
    h2 = -2 * fac3 + 3 * fac2
    h3 = fac3 - 2 * fac2 + fac
    h4 = fac3 - fac2
    return h1 * y1 + h2 * y2 + (h3 * t1 + h4 * t2) * span



class Animation():
    """Animation with a set length, speed, and keyframe data."""

    def __init__(self, keyframes: np.ndarray = None, length: float = 0):
        super().__init__()
        self.keyframes = np.ndarray((0, 3), float) if keyframes is None else keyframes.astype(float)
        """Array containing the frame index, value, and tangent (in that order) for each keyframe.

        All of these (including frame index) can be non-int values. The first frame is always 0 in
        retail files, but other starting frames work fine in Dolphin (untested on console).
        (Up to the first defined frame, the animation just evaluates to that frame)
        """
        self.length = length

    def copy(self):
        """Create a copy of this animation with a new (but identical) keyframe array."""
        return Animation(self.keyframes.copy(), self.length)

    def __eq__(self, other):
        if not isinstance(other, Animation) or self.length != other.length:
            return False
        # short-circuit evaluation is faster than np.array_equal()
        selfKfs = self.keyframes
        otherKfs = other.keyframes
        if selfKfs.shape != otherKfs.shape:
            return False
        for selfSlice, otherSlice in zip(selfKfs, otherKfs):
            if not np.all(selfSlice == otherSlice):
                return False
        return True

    def __len__(self):
        return len(self.keyframes)

    def setSmooth(self):
        """Set this animation's tangents to be smooth.

        Note that if this animation has fewer than 2 keyframes, nothing is changed."""
        kfs = self.keyframes
        numKfs = len(kfs)
        if numKfs < 2:
            return
        prv = np.pad(kfs[:-1], ((1, 0), (0, 0)), "edge")
        nxt = np.pad(kfs[1:], ((0, 1), (0, 0)), "edge")
        kfs[:, 2] = (nxt[:, 1] - prv[:, 1]) / (nxt[:, 0] - prv[:, 0])

    def interpolate(self, frames: np.ndarray) -> np.ndarray:
        """Get interpolated values for this animation at the given positions."""
        kfs = self.keyframes
        interpolated = np.empty(frames.shape)
        nextKfIdcs = np.searchsorted(kfs[:, 0], frames) # next kf index for each frame
        outOfBoundsL = nextKfIdcs == 0
        outOfBoundsR = nextKfIdcs == len(kfs)
        inBounds = np.logical_not(np.logical_or(outOfBoundsL, outOfBoundsR))
        interpolated[outOfBoundsL] = kfs[0, 1]
        interpolated[outOfBoundsR] = kfs[-1, 1]
        idcs = nextKfIdcs[inBounds]
        interpolated[inBounds] = hermite(*kfs[idcs].T, *kfs[idcs - 1].T, frames[inBounds])
        return interpolated


class AnimSerializer(Reader[None, Animation], Writer[None, Animation]):
    """Format for packing/unpacking animation frame data to/from bytes."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def size(self) -> int:
        pass

    def _calcSize(self):
        return super()._calcSize()

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def copyFrom(self, o: "AnimSerializer"):
        """Copy another animation serializer's data into this one."""
        self._data = o._data.copy()

    @classmethod
    @abstractmethod
    def framesStorable(cls, length: int) -> bool:
        """Return whether an animation can be stored using this format based on total frames."""

    @classmethod
    @abstractmethod
    def tangentsStorable(cls, tans: np.ndarray) -> bool:
        """Return whether a tangent array can be stored using this format."""


class InterpolatedAnimSerializer(AnimSerializer):
    """Format for storing animation data based on keyframe interpolation."""

    _HEAD_STRCT: Struct
    _FRAME_STRCT: Struct

    def size(self):
        return self._HEAD_STRCT.size + pad(self._FRAME_STRCT.size * len(self._data.keyframes), 4)


class CompressedAnimSerializer(AnimSerializer):
    """Compressed frame format.

    A set of keyframes for this format has a "step" & "base" value (both floats), and rather than
    storing a float value for each keyframe, an int scalar is stored such that the keyframe value
    is equal to base + step * scalar. The step and scalars are always positive.
    """

    # arbitrary tolerance values for compression, taken from brawlbox
    SCALE_ERROR_TOL = .0005
    TAN_ERROR_TOL = .001

    def __init__(self, step: float = None):
        super().__init__()
        self.step = step
        """Step value used by this anim. If None, a value is temporarily calculated on pack."""

    def copyFrom(self, o: AnimSerializer):
        """Copy another animation serializer's data into this one."""
        super().copyFrom(o)
        if isinstance(o, CompressedAnimSerializer):
            self.step = o.step

    @classmethod
    @abstractmethod
    def findStep(cls, frameVals: np.ndarray, base: int, valRange: int):
        """Find a step value for some frame values with an error within the acceptable tolerance.

        valRange is the range of values (maximum - minimum). base is the minimum.

        Note that for the sake of speed, this may not be the best step value possible - it's just
        the first one found that has a low enough error.

        If no acceptable value is found, raise a ValueError.
        """

    @classmethod
    def _findStep(cls, frameVals: np.ndarray, base: int, valRange: int, maxStep: int):
        """Find a step value for some frame values with an error within the acceptable tolerance.

        valRange is the range of values (maximum - minimum). base is the minimum. maxStep isn't an
        actual step value, but a rather, the maximum number of step increments within the range.
        (For instance, with a range of 10, maxStep of 5, and minStep of 3, the possible steps are
        10/5 and 10/4, which are 2 and 2.5 (the minimum is not inclusive).

        Note that for the sake of speed, this may not be the best step value possible - it's just
        the first one found (iterating from max # of steps to 0) that has a low enough error.

        If no acceptable value is found, raise a ValueError.
        """
        for numSteps in range(maxStep, 0, -1):
            step = valRange / numSteps
            compressed = ((frameVals - base) / step + .5).astype(np.uint16)
            decompressed = base + step * compressed
            error = np.abs(frameVals - decompressed)
            if np.max(error) < cls.SCALE_ERROR_TOL:
                return step
        raise ValueError("No step value found with tolerable error")


class CompressedInterpAnimSerializer(InterpolatedAnimSerializer, CompressedAnimSerializer):
    """Compressed frame format based on keyframe interpolation."""

    _HEAD_STRCT = Struct(">Hxxfff")

    @classmethod
    def _tangentsStorable(cls, tans: np.ndarray, tanScale: int, tanBits: int):
        """Return whether a tangent array can be stored properly based on tangent storage settings.

        These settings are a scale by which tangents should be multiplied before they're stored, and
        the number of bits that the stored tangents take up. Returns False if the maximum tangent
        is too large for this bit size or the error when converting the tangents to their stored
        format is unacceptable.
        """
        intTans = np.round(tans * tanScale)
        maxAllowed = maxBitVal(tanBits - 1) # tangents are signed
        if np.max(intTans) > maxAllowed or np.min(intTans) < -maxAllowed - 1:
            return False
        cmpr = intTans / tanScale # tangent values post-compression
        return np.allclose(tans, cmpr, atol=cls.TAN_ERROR_TOL)

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = Animation()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data)
        numKeyframes = unpackedHeader[0]
        self._data.length = calcAnimLen(unpackedHeader[1])
        self.step, base = unpackedHeader[2:4]
        keyframes = []
        for kfIdx in range(numKeyframes):
            kfOffset = self._HEAD_STRCT.size + kfIdx * self._FRAME_STRCT.size
            idx, scalar, tan = self._unpackFrame(data, kfOffset)
            keyframes.append((idx, base + self.step * scalar, tan))
        self._data.keyframes = np.array(keyframes, dtype=float)
        return self

    @abstractmethod
    def _unpackFrame(self, data: bytes, offset: int) -> tuple[int, int, float]:
        """Unpack the frame index, step scalar, and tangent for a frame of animation."""

    def pack(self):
        vals = self._data.keyframes[:, 1]
        base = vals.min()
        step = self.step if self.step is not None else self.findStep(vals, base, vals.max() - base)
        vals = ((vals - base) / step + .5).astype(np.uint32)
        idcs = self._data.keyframes[:, 0]
        tans = self._data.keyframes[:, 2]
        frameScale = calcFrameScale(self._data.length)
        packedHeader = self._HEAD_STRCT.pack(len(vals), frameScale, step, base)
        return packedHeader + pad(self._packFrames(idcs, vals, tans), 4)

    @abstractmethod
    def _packFrames(self, frameIdcs: np.ndarray, vals: np.ndarray, tans: np.ndarray) -> bool:
        """Pack animation frame data to bytes."""


class I4(CompressedInterpAnimSerializer):
    """4-byte interpolated frame format."""

    _FRAME_STRCT = Struct(">I")

    @classmethod
    def findStep(cls, frameVals: np.ndarray, base: int, valRange: int):
        return cls._findStep(frameVals, base, valRange, maxBitVal(12))

    @classmethod
    def framesStorable(cls, length: int):
        return length <= maxBitVal(8)

    @classmethod
    def tangentsStorable(cls, tans: np.ndarray):
        return cls._tangentsStorable(tans, 32, 12)

    def _unpackFrame(self, data: bytes, offset: int) -> tuple[int, int, float]:
        frameInfo = self._FRAME_STRCT.unpack_from(data, offset)[0]
        idx = frameInfo >> 24 & maxBitVal(8)
        scalar = frameInfo >> 12 & maxBitVal(12)
        tan = frameInfo >> 0 & maxBitVal(12)
        if tan & (1 << 11): # tan is signed
            tan = -(maxBitVal(12) & ~tan + 1)
        tan /= 32
        return (idx, scalar, tan)

    def _packFrames(self, frameIdcs: np.ndarray, vals: np.ndarray, tans: np.ndarray):
        frameIdcs = np.round(frameIdcs).astype(np.uint32)
        tans = np.round(tans * 32).astype(np.int16).view(np.uint16) & maxBitVal(12)
        return ((frameIdcs << 24) | (vals << 12) | (tans << 0)).astype(">u4").tobytes()


class I6(CompressedInterpAnimSerializer):
    """6-byte interpolated frame format."""

    _FRAME_STRCT = Struct(">HHh")

    @classmethod
    def findStep(cls, frameVals: np.ndarray, base: int, valRange: int):
        return cls._findStep(frameVals, base, valRange, maxBitVal(16))

    @classmethod
    def framesStorable(cls, length: int):
        return length <= maxBitVal(16) / 32

    @classmethod
    def tangentsStorable(cls, tans: np.ndarray):
        return cls._tangentsStorable(tans, 256, 16)

    def _unpackFrame(self, data: bytes, offset: int) -> tuple[int, int, float]:
        frameInfo = self._FRAME_STRCT.unpack_from(data, offset)
        return (frameInfo[0] / 32, frameInfo[1], frameInfo[2] / 256)

    def _packFrames(self, frameIdcs: np.ndarray, vals: np.ndarray, tans: np.ndarray):
        frameIdcs = np.round(frameIdcs * 32).astype(np.uint16)
        tans =  np.round(tans * 256).astype(np.int16)
        return b"".join(self._FRAME_STRCT.pack(i, v, t) for i, v, t in zip(frameIdcs, vals, tans))


class I12(InterpolatedAnimSerializer):
    """12-byte interpolated frame format."""

    _HEAD_STRCT = Struct(">Hxxf")
    _FRAME_STRCT = Struct(">fff")

    @classmethod
    def framesStorable(cls, length: int):
        return True

    @classmethod
    def tangentsStorable(cls, tans: np.ndarray):
        return True

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = Animation()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data)
        numKeyframes = unpackedHeader[0]
        self._data.length = calcAnimLen(unpackedHeader[1])
        keyframes = []
        for kfIdx in range(numKeyframes):
            kfOffset = self._HEAD_STRCT.size + kfIdx * self._FRAME_STRCT.size
            kfInfo = self._FRAME_STRCT.unpack_from(data, kfOffset)
            keyframes.append(kfInfo)
        self._data.keyframes = np.array(keyframes, dtype=float)
        return self

    def pack(self):
        frameScale = calcFrameScale(self._data.length)
        packedHeader = self._HEAD_STRCT.pack(len(self._data.keyframes), frameScale)
        return packedHeader + self._data.keyframes.astype(">f4").tobytes()


class DiscreteAnimSerializer(AnimSerializer):
    """Format for storing animation data where a value is stored for every frame.
    
    Note: BrawlBox & other sources call this "linear", but that's confusing imo bc it sounds like
    linear interpolation, so I call it "discrete" instead.
    """

    _HEAD_STRCT: Struct
    _FRAME_TYPE: np.dtype

    def __init__(self, length = None):
        super().__init__()
        self.length = length
        """Total length of this animation. If None, temporarily calculated automatically on pack.

        (Required to be provided for unpack, as it determines how many frames are read)
        """

    def _interpolated(self):
        """All interpolated frame values for this animation."""
        a = self._data
        l = self.length if self.length is not None else a.keyframes[-1, 0] - a.keyframes[0, 0]
        return a.interpolate(np.arange(l + 1))

    def size(self):
        a = self._data
        l = self.length if self.length is not None else a.keyframes[-1, 0] - a.keyframes[0, 0]
        return self._HEAD_STRCT.size + pad(self._FRAME_TYPE.itemsize * int(round(l + 1)), 4)

    def fromInstance(self, data: Animation):
        super().fromInstance(data)
        self.length = data.length
        return self

    @classmethod
    def framesStorable(cls, length: int):
        return True

    @classmethod
    def tangentsStorable(cls, tans: np.ndarray):
        return True

    def copyFrom(self, o: AnimSerializer):
        """Copy another animation serializer's data into this one."""
        super().copyFrom(o)
        if isinstance(o, DiscreteAnimSerializer):
            self.length = o.length


class CompressedDiscAnimSerializer(DiscreteAnimSerializer, CompressedAnimSerializer):
    """Compressed frame format with a value for every frame."""

    _HEAD_STRCT = Struct(">ff")

    def __init__(self, length = None, step: float = None):
        DiscreteAnimSerializer.__init__(self, length)
        CompressedAnimSerializer.__init__(self, step)

    @classmethod
    def findStep(cls, frameVals: np.ndarray, base: int, valRange: int):
        return cls._findStep(frameVals, base, valRange, maxBitVal(cls._FRAME_TYPE.itemsize * 8))

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = Animation()
        self._data.length = length = self.length if self.length is not None else 0
        self.step, base = self._HEAD_STRCT.unpack_from(data)
        offset = self.offset + self._HEAD_STRCT.size
        vals = base + self.step * np.frombuffer(data[offset:], self._FRAME_TYPE, length + 1)
        kfs = (np.arange(len(vals)), vals, np.zeros(len(vals)))
        self._data.keyframes = np.stack(kfs, axis=1).astype(float)
        self._data.setSmooth()
        return self

    def pack(self):
        vals = self._interpolated()
        base = vals.min()
        step = self.step if self.step is not None else self.findStep(vals, base, vals.max() - base)
        vals = ((vals - base) / step + .5).astype(self._FRAME_TYPE)
        return self._HEAD_STRCT.pack(step, base) + pad(vals.tobytes(), 4)


class D1(CompressedDiscAnimSerializer):
    """1-byte discrete frame format."""

    _FRAME_TYPE = np.dtype(">u1")


class D2(CompressedDiscAnimSerializer):
    """2-byte discrete frame format."""

    _FRAME_TYPE = np.dtype(">u2")


class D4(DiscreteAnimSerializer):
    """4-byte discrete frame format."""

    _HEAD_STRCT = Struct(">")
    _FRAME_TYPE = np.dtype(">f4")

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = Animation()
        self._data.length = self.length if self.length is not None else 0
        vals = np.frombuffer(data[self.offset:], self._FRAME_TYPE, self._data.length + 1)
        kfs = (np.arange(len(vals)), vals, np.zeros(len(vals)))
        self._data.keyframes = np.stack(kfs, axis=1).astype(float)
        self._data.setSmooth()
        return self

    def pack(self):
        return self._interpolated().astype(self._FRAME_TYPE).tobytes()


class AnimSubfile(Subfile):
    """BRRES subfile for animations."""

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.length: int = 0
        """Number of frames in this animation, including both endpoints (frame span + 1)."""
        self.enableLoop: bool = False


def readFrameRefs(data: bytes, baseOffset: int, offset: int,
                  iso: bool, useModel: bool, fixed: list[bool], exists: bool,
                  fmt: type[AnimSerializer], anims: list[Animation], length: int = None):
    """Read animations for one transformation property, starting at a fixed value or data pointer.

    Two offsets into the data are required - one "base offset" to which the pointers are relative,
    and another one relative to the base offset that points to the spot in the refs to read from.

    The base offset can be None - in that case, the pointers will be relative to their own
    individual offsets, and the full offset should be provided for "offset".

    The contents of "frames" are replaced with a list of keyframes for each component of this
    property, unless "exists" and "useModel" are both False (in that case, nothing is changed).
    Return the number of bytes read for fixed values or data pointers (not the data itself).

    Note that the "length" parameter is only used for discrete animation formats, and gets
    completely ignored otherwise. (If a discrete format is read, it's necessary to tell how much
    frame data there is)
    """
    numComponents = len(fixed)
    if useModel:
        anims[:] = []
        return 0
    elif not exists:
        return 0
    else:
        anims[:] = []
        initialOffset = (baseOffset + offset) if baseOffset is not None else offset
        curOffset = initialOffset
        for compFixed in (fixed if not iso else fixed[:1]):
            strct = FRAME_FIXED_STRCT if compFixed else FRAME_PTR_STRCT
            frameData = strct.unpack_from(data, curOffset)[0]
            if compFixed: # frame data is a fixed val
                anims.append(Animation(np.array(((0, frameData, 0), ))))
            else: # frame data is an offset to keyframes
                dataOffset = (baseOffset if baseOffset is not None else curOffset) + frameData
                anim = fmt(length) if issubclass(fmt, DiscreteAnimSerializer) else fmt()
                anims.append(anim.unpack(data[dataOffset:]).getInstance())
            curOffset += strct.size
        if iso:
            anims[1:] = [anims[0].copy() for _ in range(numComponents - 1)]
        return curOffset - initialOffset


def packFrameRefs(data: list[AnimSerializer | float], baseOffset: int, individualRelative = False):
    """Pack keyframe references (data pointers and fixed values).

    A base offset upon which the data pointers will be based must be provided.

    "individualRelative" determines how the pointers' offsets are stored. If False, they're relative
    to the base offset. If True, they're relative to the offsets of the pointers themselves (in
    that case, the first pointer is assumed to be located at baseOffset)
    """
    packed = b""
    for d in data:
        if isinstance(d, float):
            packed += FRAME_FIXED_STRCT.pack(d)
        else:
            packed += FRAME_PTR_STRCT.pack(d.offset - baseOffset)
        if individualRelative:
            baseOffset += 4 # 4 = size of both fixed struct and ptr struct
    return packed


def serializeAnims(anims: list[Animation], fmts: list[type[AnimSerializer]]):
    """Get serializers for a list of animations.

    The format of these serializers is homogenous, and is chosen from the provided list (which may
    be modified) to maximize compression & accuracy.
    """
    # filter format based on maximum frame index (whether it's too big to be stored)
    for anim in anims:
        maxIdx = anim.keyframes[:, 0].max()
        while not fmts[0].framesStorable(maxIdx):
            fmts.pop(0)
    # filter format based on tangents (whether they're too big to be stored)
    for anim in anims:
        while not fmts[0].tangentsStorable(anim.keyframes[:, 2]):
            fmts.pop(0)
    # filter based on finding acceptable step values if compression is used
    sers: list[AnimSerializer] = []
    for anim in anims:
        vals = np.unique(anim.keyframes[:, 1])
        kfMin = vals.min()
        kfMax = vals.max()
        kfRange = kfMax - kfMin
        while issubclass(fmts[0], CompressedAnimSerializer):
            try:
                step = fmts[0].findStep(vals, kfMin, kfRange)
                sers.append(fmts[0](step).fromInstance(anim))
                break
            except ValueError:
                fmts.pop(0)
        else:
            sers.append(fmts[0]().fromInstance(anim))
    # finally, ensure all serializer entries have the type we've decided on
    # (most compressed that works for all entries)
    fmt = fmts[0]
    for i, animSer in enumerate(sers):
        if not isinstance(animSer, fmt):
            sers[i] = fmt().copyFrom(animSer)
    return sers


def groupAnimWriters(writers: list[list[WRITER_T]], sort = True, usePacked = True):
    """Sort a list of lists of writers & group entries w/ identical data.

    The writers can be of any type, but their data - retrieved through getInstance() - must work
    with len() if sorting is enabled.

    "Identical data" means the entries are equivalent when packed. Alternatively, you can set
    "usePacked" to False to compare their data (retrieved through getInstance()).

    For each list in the input, the maximum writer length within that list is retrieved.
    Writers are then sorted based on these maximum values for their parent lists.
    TODO: This bizarre method gives close results to how these are sorted in retail files, but it's
    not exact. I have no idea WTF the actual method is and need to revisit this.
    """
    packed = {anim: anim.pack() for anims in writers for anim in anims} if usePacked else {}
    inst = {anim: anim.getInstance() for anims in writers for anim in anims}
    animData: list[list[WRITER_T]] = []
    if sort:
        writers = sorted(writers, key=lambda l: -max(len(inst[s]) for s in l) if l else 0)
    for anims in writers:
        for anim in anims:
            found = False
            for l in animData:
                areEq = False
                if (packed[anim] == packed[l[0]]) if usePacked else (inst[anim] == inst[l[0]]):
                    l.append(anim)
                    found = True
                    break
            if not found:
                animData.append([anim])
    return animData
