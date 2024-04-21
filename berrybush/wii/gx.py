# standard imports
from abc import abstractmethod
from enum import Enum
from functools import cached_property, cache
from io import BytesIO
from typing import Iterable, TypeVar, Generic, TYPE_CHECKING
# 3rd party imports
import numpy as np
# internal imports
from .alias import alias
from .bitstruct import BitStruct, Bits, NormalizedBits, NormalizedSignedBits
from .binaryutils import maxBitVal
from .common import EnumWithAttr
# special typing imports
if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = object


# brief note:
# https://stackoverflow.com/questions/74103528/type-hinting-an-instance-of-a-nested-class
# the kind of structure described in this question is used a lot here
# (specifically in register types and vertex attribute types)
# unfortunately, the accepted (best) solution doesn't work in vscode, so i had to combine
# that with the other answer (type hint on every subclass) for full functionality
# if that vscode bug ever gets fixed, i can get rid of all those type hints, which would be great


# another note:
# all register names, enum names, etc are made up based on my understanding of what they do.
# my sources are dolphin, libogc, and yagcd, and for any info about specifics, it's probably best
# to consult those sources rather than whatever bs i have in here.


MAX_TEXTURE_SIZE = 1024

MAX_TEV_STAGES = 16
MAX_TEXTURES = 8
MAX_INDIRECT_MTCS = 3
MAX_INDIRECTS = 4
MAX_COLOR_SWAPS = 4
MAX_TEV_STAND_COLORS = 3
MAX_TEV_CONST_COLORS = 4

MAX_PSN_MTX_ATTRS = 1
MAX_TEX_MTX_ATTRS = 8
MAX_PSN_ATTRS = 1
MAX_NRM_ATTRS = 1
MAX_CLR_ATTRS = 2
MAX_UV_ATTRS = 8

PSN_MTX_ATTR_IDX = 0
TEX_MTX_ATTR_IDX = PSN_MTX_ATTR_IDX + MAX_PSN_MTX_ATTRS
PSN_ATTR_IDX = TEX_MTX_ATTR_IDX + MAX_TEX_MTX_ATTRS
NRM_ATTR_IDX = PSN_ATTR_IDX + MAX_PSN_ATTRS
CLR_ATTR_IDX = NRM_ATTR_IDX + MAX_NRM_ATTRS
UV_ATTR_IDX = CLR_ATTR_IDX + MAX_CLR_ATTRS
MAX_ATTRS = UV_ATTR_IDX + MAX_UV_ATTRS

MAX_ATTR_MTCS = 10 # in xf memory, there are 10 psn matrices, 10 nrm matrices, and 10 tex matrices


_REG_T = TypeVar("_REG_T", bound="Register")
_REG_STRUCT_T = TypeVar("_REG_STRUCT_T", bound="Register.ValStruct")
_CP_REG_STRUCT_T = TypeVar("_CP_REG_STRUCT_T", bound="CPReg.ValStruct")
_DEC_T = TypeVar("_DEC_T", bound="AttrDec")
_ATTR_T = TypeVar("_ATTR_T", bound="VertexAttr")
_COMP_T = TypeVar("_COMP_T", bound="VertexAttr.CompType")
_DATA_T = TypeVar("_DATA_T", bound="VertexAttr.DataType")


class BitEnum(Enum):
    """Enum that can be casted to an int. Intended for use with BitStruct."""

    def __int__(self):
        return self.value


class AlphaLogicOp(BitEnum):
    """Logical operator for use in the alpha test."""
    AND = 0
    OR = 1
    XOR = 2
    XNOR = 3


class BlendDstFactor(BitEnum):
    """Destination factor for use in the blending equation."""
    ZERO = 0
    ONE = 1
    SRC_COLOR = 2
    INV_SRC_COLOR = 3
    SRC_ALPHA = 4
    INV_SRC_ALPHA = 5
    DST_ALPHA = 6
    INV_DST_ALPHA = 7


class BlendLogicOp(BitEnum):
    """Logical operator to be used in blending when in logic mode."""
    CLEAR = 0
    AND = 1
    REVAND = 2
    COPY = 3
    INVAND = 4
    NOOP = 5
    XOR = 6
    OR = 7
    NOR = 8
    EQUIV = 9
    INV = 10
    REVOR = 11
    INVCOPY = 12
    INVOR = 13
    NAND = 14
    SET = 15


class BlendSrcFactor(BitEnum):
    """Source factor for use in the blending equation."""
    ZERO = 0
    ONE = 1
    DST_COLOR = 2
    INV_DST_COLOR = 3
    SRC_ALPHA = 4
    INV_SRC_ALPHA = 5
    DST_ALPHA = 6
    INV_DST_ALPHA = 7


class ColorBitSel(BitEnum):
    """Setting for how many bits are used for each channel of an indirect texture's colors."""
    ALL_8 = 0
    LOWER_5 = 1
    LOWER_4 = 2
    LOWER_3 = 3


class ColorChannel(BitEnum):
    R = 0
    G = 1
    B = 2
    A = 3


class CompareOp(BitEnum):
    """Function for comparing two values."""
    NEVER = 0
    LESS = 1
    EQUAL = 2
    LEQUAL = 3
    GREATER = 4
    NEQUAL = 5
    GEQUAL = 6
    ALWAYS = 7


class CullMode(BitEnum):
    """Determines which sides of an object should be culled."""
    NONE = 0
    FRONT = 1
    BACK = 2
    BOTH = 3


class IndCoordScalar(BitEnum):
    """Scale value by which indirect texture coordinates are divided after lookup."""
    DIV_1 = 0
    DIV_2 = 1
    DIV_4 = 2
    DIV_8 = 3
    DIV_16 = 4
    DIV_32 = 5
    DIV_64 = 6
    DIV_128 = 7
    DIV_256 = 8


class IndMtxIdx(BitEnum):
    """Indirect matrix index."""
    NONE = 0
    IDX_0 = 1
    IDX_1 = 2
    IDX_2 = 3


class IndMtxType(BitEnum):
    """Indirect matrix type (either static/stored in memory or dynamic/based on ST scale)."""
    STATIC = 0
    S = 1
    T = 2


class IndWrapSet(BitEnum):
    """Indirect setting that determines how far a given coordinate should go before wrapping."""
    OFF = 0
    ON_256 = 1
    ON_128 = 2
    ON_64 = 3
    ON_32 = 4
    ON_16 = 5
    ON_0 = 6 # every coordinate just becomes 0


class TEVRasterSel(BitEnum):
    """Selection for a TEV stage's raster color."""
    COLOR_0 = 0
    COLOR_1 = 1
    ALPHA_BUMP = 5 # texture bump alpha, as set by TEVIndSettings.bumpAlphaComp
    NORMALIZED_ALPHA_BUMP = 6
    ZERO = 7


class TEVAlphaArg(BitEnum):
    """Argument for use in the TEV alpha calculation."""
    STANDARD_0_ALPHA = 0
    STANDARD_1_ALPHA = 1
    STANDARD_2_ALPHA = 2
    STANDARD_3_ALPHA = 3
    TEX_ALPHA = 4
    RASTER_ALPHA = 5
    CONSTANT = 6
    ZERO = 7


class TEVColorArg(BitEnum):
    """Argument for use in the TEV color calculation."""
    STANDARD_0_COLOR = 0
    STANDARD_0_ALPHA = 1
    STANDARD_1_COLOR = 2
    STANDARD_1_ALPHA = 3
    STANDARD_2_COLOR = 4
    STANDARD_2_ALPHA = 5
    STANDARD_3_COLOR = 6
    STANDARD_3_ALPHA = 7
    TEX_COLOR = 8
    TEX_ALPHA = 9
    RASTER_COLOR = 10
    RASTER_ALPHA = 11
    ONE = 12
    HALF = 13
    CONSTANT = 14
    ZERO = 15


class TEVBias(BitEnum):
    """Bias added to TEV calculation results. Can also be used to indicate comparison mode."""
    ZERO = 0
    HALF = 1
    NEGATIVE_HALF = 2
    COMPARISON_MODE = 3


class TEVConstSel(BitEnum):
    """Constant color selection for a TEV stage."""
    VAL_8_8 = 0 # these first values are fractions, 8/8, 7/8, etc
    VAL_7_8 = 1
    VAL_6_8 = 2
    VAL_5_8 = 3
    VAL_4_8 = 4
    VAL_3_8 = 5
    VAL_2_8 = 6
    VAL_1_8 = 7
    VAL_0_8 = 8
    RGB_0 = 12 # from here on, values refer to components from the stored constant colors
    RGB_1 = 13
    RGB_2 = 14
    RGB_3 = 15
    R_0 = 16
    R_1 = 17
    R_2 = 18
    R_3 = 19
    G_0 = 20
    G_1 = 21
    G_2 = 22
    G_3 = 23
    B_0 = 24
    B_1 = 25
    B_2 = 26
    B_3 = 27
    A_0 = 28
    A_1 = 29
    A_2 = 30
    A_3 = 31


class TEVOp(BitEnum):
    """Operation for use in TEV calculations.

    Either an arithmetic operator (+ or -) or comparison operator (> or =) based on whether the
    TEV stage is in "compare mode" (set by the stage's "bias" parameter).
    """
    ADD_OR_GREATER = 0
    SUBTRACT_OR_EQUALS = 1


class TEVScaleChan(BitEnum):
    """Scale or setting for what to compare in comparison mode."""
    ONE_OR_R = 0
    TWO_OR_RG = 1
    FOUR_OR_RGB = 2
    HALF_OR_BY_CHANNEL_OR_A = 3


class TexCoordSel(BitEnum):
    """Selection for a texture coordinate."""
    NONE = 0
    S = 1
    T = 2
    U = 3


class TexCoordSource(BitEnum):
    """Attribute from which texture coordinates can come on vertices."""
    POSITION = 0
    NORMAL = 1
    COLOR = 2 # use tex gen type to specify which color is used
    BINORMAL_T = 3
    BINORMAL_B = 4
    UV_0 = 5
    UV_1 = 6
    UV_2 = 7
    UV_3 = 8
    UV_4 = 9
    UV_5 = 10
    UV_6 = 11
    UV_7 = 12


class TexGen(BitEnum):
    """Determines how texture coordinates are generated from their source."""
    REGULAR = 0
    EMBOSS_MAP = 1
    COLOR_0 = 2 # r and g of vertex colors (use w/ "color" tex coord source)
    COLOR_1 = 3


class TexInputForm(BitEnum):
    """Dictates how many coordinates are passed in for a texture."""
    AB11 = 0
    ABC1 = 1


class TexProjectionType(BitEnum):
    """Type of coordinate projection used for a texture."""
    ST = 0
    STQ = 1


class ColorFormat():

    def __init__(self, *sizes):
        self._sizes = np.array(sizes)
        stride = self._sizes.sum() / 8
        if 0 >= stride or stride > 4 or stride % 1:
            raise TypeError(f"Color format stride must be an integer from 1-4. (Found {stride})")
        self.stride = int(stride)
        self.nchans = len(self._sizes)
        self._maxs = (1 << self._sizes) - 1 # max for each channel
        self._dtype = np.dtype(f">u{self.stride if self.stride != 3 else 4}") # numpy dtype
        self._shifts = np.append(np.cumsum(self._sizes[::-1])[::-1][1:], 0).astype(self._dtype)

    def normalize(self, data: np.ndarray):
        """Return a normalized copy of a color list based on the max value for each channel."""
        return data / self._maxs

    def denormalize(self, data: np.ndarray):
        """Return a denormalized copy of a color list based on the max value for each channel."""
        return data * self._maxs

    def quantize(self, data: np.ndarray):
        """Return a copy of a normalized color list with this format's precision limits applied."""
        return self.normalize(self.denormalize(data).round())

    def unpack(self, b: bytes, count = -1):
        """Unpack a list of colors from bytes."""
        data: np.ndarray = None
        if self.stride == 3: # if stride is 3, we need to read individual bytes & combine
            data = np.frombuffer(b, np.uint8, count * 3 if count != -1 else -1).reshape(-1, 3)
            data = np.bitwise_or.reduce(data << np.arange(16, -1, -8), 1).astype(self._dtype)
        else:
            data = np.frombuffer(b, self._dtype, count)
        return data.reshape(-1, 1) >> self._shifts & self._maxs

    def pack(self, data: np.ndarray):
        """Pack a list of colors to bytes."""
        nchans = data.shape[1]
        if nchans != self.nchans:
            raise TypeError(f"Color format {self._sizes} cannot pack data with {nchans} channels")
        data = data.round().astype(self._dtype) << self._shifts
        data = np.bitwise_or.reduce(data, 1).astype(self._dtype)
        if self.stride == 3: # if stride is 3, we need to pack individual bytes
            data = (data.reshape(-1, 1) >> np.arange(16, -1, -8)).astype(np.uint8)
        return data.tobytes()


class AttrDec(BitEnum, EnumWithAttr):
    """Declares the format in which some vertex attribute is stored in draw commands."""
    @cached_property
    def stride(self) -> int:
        """Size of this attribute, as stored in draw commands, in bytes. -1 for direct data."""
        return self._attr_


class StdAttrDec(AttrDec):
    """Declares the format in which a standard vertex attribute is stored in draw commands."""
    NOT_PRESENT = 0, 0 # attribute isn't included
    DIRECT = 1, -1 # data is provided directly
    IDX8 = 2, 1 # 8 bit index
    IDX16 = 3, 2 # 16 bit index


class MtxAttrDec(AttrDec):
    """Declares the format in which a matrix vertex attribute is stored in draw commands."""
    NOT_PRESENT = 0, 0 # attribute isn't included
    IDX8 = 1, 1 # 8 bit index


class VertexAttr(Generic[_COMP_T, _DATA_T]):
    """Describes the format for some data that can be associated with a vertex.

    Each entry of this data (i.e., each point) is called an item, and this can be used
    to pack these items to & from bytes.

    Has a data type that determines how the data's stored, and a component type that indicates the
    number of components per item.
    """

    MAX_ATTRS: int
    PADDED_COUNT: int
    PAD_VAL: int

    class CompType(BitEnum, EnumWithAttr):
        """Describes a vertex attribute component type, with a name and number of dimensions."""
        @cached_property
        def count(self) -> int:
            return self._attr_
        @classmethod
        @cache
        def maxDims(cls):
            """Maximum allowed dimensions for this attribute type."""
            return max(e.count for e in cls)

    class DataType(BitEnum, EnumWithAttr):
        """Describes a vertex attribute data type (e.g., 16-bit int, 32-bit float, etc.)."""
        @cached_property
        def fmt(self) -> np.dtype:
            return self._attr_

    def __init__(self, dtype: DataType = 0):
        self._dtype = self.DataType(dtype) # false positive - pylint: disable=no-value-for-parameter

    @property
    @abstractmethod
    def ctype(self) -> CompType:
        """Component type for this attribute, indicating the number of components per item."""

    @property
    @abstractmethod
    def dtype(self) -> DataType:
        """Data type for this attribute, indicating the format of each item's data."""

    @property
    @abstractmethod
    def stride(self) -> int:
        """Size of one item for this attribute, in bytes."""

    @abstractmethod
    def unpackBuffer(self, b: bytes, count: int) -> np.ndarray:
        """Unpack an array of data from bytes following this format."""

    def calcBufferSize(self, count: int) -> int:
        """Get the size in bytes of data following this format, given the number of entries."""
        return self.stride * count

    @abstractmethod
    def packBuffer(self, arr: np.ndarray) -> bytes:
        """Pack an array of data to bytes following this format."""

    def copy(self):
        """Return a copy of this vertex attribute format."""
        return type(self)(self._dtype)

    def copyFrom(self, other: Self):
        """Copy this format's settings from another."""
        self._dtype = other._dtype

    @classmethod
    def pad(cls, data: np.ndarray) -> np.ndarray:
        """Pad a numpy array's last axis to the standard length for this attribute type."""
        padWidths = [(0, 0)] * data.ndim
        padWidths[-1] = (0, cls.PADDED_COUNT - data.shape[-1])
        return np.pad(data, padWidths, constant_values=cls.PAD_VAL)

    def __eq__(self, other: Self):
        return self.dtype == other.dtype and self.ctype == other.ctype


class StdVertexAttr(VertexAttr[_COMP_T, "StdVertexAttr.DataType"]):
    """Standard vertex attribute template followed by most attribute types.

    Data type indicates the format of each component, so data and component type are independent
    from one another and can be set independently. There's also a scale, used to scale down integer
    data from packed values. Scale is disregarded for float data.
    """

    PAD_VAL = 0

    dtype: "DataType"
    class DataType(VertexAttr.DataType):
        UINT8 = 0, np.dtype(">u1")
        INT8 = 1, np.dtype(">i1")
        UINT16 = 2, np.dtype(">u2")
        INT16 = 3, np.dtype(">i2")
        FLOAT32 = 4, np.dtype(">f4")

    def __init__(self, dtype=DataType.FLOAT32, ctype: VertexAttr.CompType = 0, scale = 0):
        super().__init__(dtype)
        self._ctype = self.CompType(ctype) # false positive - pylint: disable=no-value-for-parameter
        self.scale = scale

    @property
    def ctype(self):
        return self._ctype

    @ctype.setter
    def ctype(self, ctype: VertexAttr.CompType):
        self._ctype = ctype

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DataType):
        self._dtype = dtype

    @property
    def scale(self):
        """Scale for this attribute. If type isn't float, data is stored scaled by (1 << scale)."""
        return self._scale

    @scale.setter
    def scale(self, scale: int):
        maxScale = maxBitVal(8)
        if scale not in range(maxScale + 1):
            raise ValueError(f"Attribute scale must be in range 0-{maxScale} (inclusive).")
        self._scale = scale

    @property
    def stride(self):
        return self.dtype.fmt.itemsize * self.ctype.count

    def unpackBuffer(self, b: bytes, count: int):
        compCount = self.ctype.count
        unpacked = np.frombuffer(b, self.dtype.fmt, count * compCount).reshape(count, compCount)
        if self.dtype is not self.DataType.FLOAT32:
            unpacked = unpacked / (1 << self.scale)
        return self.pad(unpacked)

    def packBuffer(self, arr: np.ndarray):
        # arrays may have more than 2 dimensions (in case of nbt normals),
        # but shape[0] is always # of entries and shape[1:] always sums to # of components
        # so, use reshaping & then trim based on desired component count
        compCount = self.ctype.count
        packed = np.ascontiguousarray(arr.reshape(len(arr), -1)[:, :compCount])
        if self.dtype is not self.DataType.FLOAT32:
            packed = np.round(packed * (1 << self.scale))
        return packed.astype(self.dtype.fmt).tobytes()

    def copy(self):
        return type(self)(self._dtype, self._ctype, self.scale)

    def copyFrom(self, other: Self):
        super().copyFrom(other)
        self.ctype = other.ctype
        self.scale = other.scale

    def __eq__(self, other: Self):
        return super().__eq__(other) and self.scale == other.scale


class PsnAttr(StdVertexAttr["PsnAttr.CompType"]):
    MAX_ATTRS = MAX_PSN_ATTRS
    PADDED_COUNT = 3
    ctype: "CompType"
    class CompType(VertexAttr.CompType):
        XY = 0, 2
        XYZ = 1, 3
    def __init__(self, dtype = StdVertexAttr.DataType.FLOAT32, ctype = CompType.XYZ, scale = 0):
        super().__init__(dtype, ctype, scale)


class NrmAttr(StdVertexAttr["NrmAttr.CompType"]):
    MAX_ATTRS = MAX_NRM_ATTRS
    PADDED_COUNT = 3
    ctype: "CompType"
    class CompType(VertexAttr.CompType):
        N = 0, 3 # normal vector, xyz
        NBT = 1, 9 # normal, binormal, and tangent have one index, all xyz
        NBT_SPLIT = 2, 3 # normal, binormal, and tangent have 3 indices, all xyz
        @property
        def isNBT(self):
            return self in {self.NBT, self.NBT_SPLIT}
        # note: anything involving binormals/tangents (i.e., anything involving nbt or nbt_split)
        # is completely untested, as it's unused in nsmbw (and maybe nw4r in general, idk)


class ClrAttr(VertexAttr["ClrAttr.CompType", "ClrAttr.DataType"]):

    MAX_ATTRS = MAX_CLR_ATTRS
    PADDED_COUNT = 4
    PAD_VAL = 1

    ctype: "CompType"
    class CompType(VertexAttr.CompType):
        RGB = 0, 3
        RGBA = 1, 4

    dtype: "DataType"
    class DataType(VertexAttr.DataType):
        RGB565 = 0, ColorFormat(5, 6, 5)
        RGB8 = 1, ColorFormat(8, 8, 8)
        RGBX8 = 2, ColorFormat(8, 8, 8, 8)
        RGBA4 = 3, ColorFormat(4, 4, 4, 4)
        RGBA6 = 4, ColorFormat(6, 6, 6, 6)
        RGBA8 = 5, ColorFormat(8, 8, 8, 8)
        @property
        def fmt(self):
            return np.float32
        @cached_property
        def colorFmt(self) -> ColorFormat:
            return self._attr_

    def __init__(self, dtype = DataType.RGBA8):
        super().__init__(dtype)

    @property
    def ctype(self):
        if self._dtype in (self.DataType.RGB565, self.DataType.RGB8, self.DataType.RGBX8):
            return self.CompType.RGB
        return self.CompType.RGBA

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DataType):
        self._dtype = dtype

    @property
    def stride(self):
        return self._dtype.colorFmt.stride

    def unpackBuffer(self, b: bytes, count: int):
        colorFmt = self._dtype.colorFmt
        return self.pad(colorFmt.normalize(colorFmt.unpack(b, count)))

    def packBuffer(self, arr: np.ndarray):
        colorFmt = self._dtype.colorFmt
        return colorFmt.pack(colorFmt.denormalize(arr[:, :colorFmt.nchans]))


class UVAttr(StdVertexAttr["UVAttr.CompType"]):
    MAX_ATTRS = MAX_UV_ATTRS
    PADDED_COUNT = 2
    ctype: "CompType"
    class CompType(VertexAttr.CompType):
        U = 0, 1
        UV = 1, 2
    def __init__(self, dtype = StdVertexAttr.DataType.FLOAT32, ctype = CompType.UV, scale = 0):
        super().__init__(dtype, ctype, scale)


class AttrDef(Generic[_DEC_T, _ATTR_T]):
    """Combines a vertex attr declaration and format to fully define how it's used in vertices."""

    def __init__(self, dec: _DEC_T = StdAttrDec.NOT_PRESENT, fmt: _ATTR_T = None):
        self.dec = dec
        self.fmt = fmt

    @property
    def stride(self):
        """Size of this attribute, as stored in draw commands, in bytes."""
        decStride = self.dec.stride
        return decStride if decStride != -1 else self.fmt.stride


class Register(Generic[_REG_STRUCT_T]):
    """Register in a Wii GPU component (CP, XF, BP).

    Has a range of possible addresses, as well as a BitStruct that dictates the components which
    make up this register's value. Each instance has an instance of this BitStruct, as well as an
    index into the address range (used for things like TEV stages, where different stage indices
    have different addresses, but the format's the same).
    """

    VALID_ADDRESSES: tuple[bytes, ...] # all possible addresses for this register

    class ValStruct(BitStruct):
        """Structure for the data stored in this register."""

    def __init__(self, idx = 0, addr: bytes = None, b: bytes = None, v: _REG_STRUCT_T = None):
        """Create a register instance.

        Its address index can be set directly or from the address itself (as long as it's valid).
        You can also pass in a packed value (b) to be unpacked as this register's value,
        or an unpacked value (v) to be copied and used.
        """
        self.idx = idx if addr is None else self.VALID_ADDRESSES.index(addr)
        self._bits: _REG_STRUCT_T
        if v is not None:
            self._bits = v.copy()
        elif b is not None:
            self._bits = self.ValStruct.unpack(b)
        else:
            self._bits = self.ValStruct()

    @property
    def bits(self) -> ValStruct:
        return self._bits

    @property
    def addr(self):
        return self.VALID_ADDRESSES[self.idx]


class CPReg(Register[_CP_REG_STRUCT_T]):
    """Register in the command processor."""

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        def applyTo(self, d: "VertexDef"):
            """Modify a vertex def from this register's settings."""
        def applyFrom(self, d: "VertexDef"):
            """Modify this register's settings from a vertex def."""


class VertexDec0(CPReg["VertexDec0.ValStruct"]):
    """First of 2 vertex attribute declaration registers."""

    VALID_ADDRESSES = (b"\x50", )

    bits: "ValStruct"
    class ValStruct(CPReg.ValStruct):
        psnMtx = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx0 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx1 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx2 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx3 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx4 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx5 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx6 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        _texMtx7 = Bits(1, MtxAttrDec, MtxAttrDec.NOT_PRESENT)
        psn = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        nrm = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _clr0 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _clr1 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _pad = Bits(15, int)
        texMtcs = alias(*(f"_texMtx{i}" for i in range(8)))
        clrs = alias("_clr0", "_clr1")

        def applyTo(self, d: "VertexDef"):
            decs = (self.psnMtx, *self.texMtcs, self.psn, self.nrm, *self.clrs)
            for aDef, dec in zip(d.attrs, decs):
                aDef.dec = dec

        def applyFrom(self, d: "VertexDef"):
            self.psnMtx = d.psnMtcs[0].dec
            self.texMtcs = [aDef.dec for aDef in d.texMtcs]
            self.psn = d.psns[0].dec
            self.nrm = d.nrms[0].dec
            self.clrs = [aDef.dec for aDef in d.clrs]


class VertexDec1(CPReg["VertexDec1.ValStruct"]):
    """Second of 2 vertex attribute declaration registers."""

    VALID_ADDRESSES = (b"\x60", )

    bits: "ValStruct"
    class ValStruct(CPReg.ValStruct):
        _uv0 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv1 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv2 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv3 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv4 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv5 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv6 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _uv7 = Bits(2, StdAttrDec, StdAttrDec.NOT_PRESENT)
        _pad = Bits(16, int)
        uvs = alias(*(f"_uv{i}" for i in range(8)))

        def applyTo(self, d: "VertexDef"):
            for aDef, dec in zip(d.uvs, self.uvs):
                aDef.dec = dec

        def applyFrom(self, d: "VertexDef"):
            self.uvs = [aDef.dec for aDef in d.uvs]


class VertexFmt0(CPReg["VertexFmt0.ValStruct"]):
    """First of 3 vertex attribute format registers."""

    VALID_ADDRESSES = (b"\x70", )

    bits: "ValStruct"
    class ValStruct(CPReg.ValStruct):
        psnCType = Bits(1, PsnAttr.CompType, PsnAttr.CompType.XYZ)
        psnDType = Bits(3, PsnAttr.DataType, PsnAttr.DataType.UINT8)
        psnScale = Bits(5, int)
        _nrmCType = Bits(1, NrmAttr.CompType, NrmAttr.CompType.N)
        nrmDType = Bits(3, NrmAttr.DataType, NrmAttr.DataType.UINT8)
        _clrCType0 = Bits(1, ClrAttr.CompType, ClrAttr.CompType.RGB)
        _clrDType0 = Bits(3, ClrAttr.DataType, ClrAttr.DataType.RGB565)
        _clrCType1 = Bits(1, ClrAttr.CompType, ClrAttr.CompType.RGB)
        _clrDType1 = Bits(3, ClrAttr.DataType, ClrAttr.DataType.RGB565)
        _uvCType0 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType0 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale0 = Bits(5, int)
        # if false, divisor is ignored for int8/uint8
        dequant = Bits(1, bool, True)
        # indicates the NBT_SPLIT normal ctype, since there's not enough room at nrmCType
        # (only 1 bit there for 3 values)
        _nbtSplit = Bits(1, bool, False)
        clrCTypes = alias("_clrCType0", "_clrCType1")
        clrDTypes = alias("_clrDType0", "_clrDType1")
        uvCTypes = alias("_uvCType0", forceList=True)
        uvDTypes = alias("_uvDType0", forceList=True)
        uvScales = alias("_uvScale0", forceList=True)

        @property
        def nrmCType(self):
            # false positive - pylint: disable=no-value-for-parameter
            return NrmAttr.CompType(self._nrmCType.value + int(self._nbtSplit))

        @nrmCType.setter
        def nrmCType(self, v):
            if v is NrmAttr.CompType.NBT_SPLIT:
                self._nrmCType = NrmAttr.CompType.NBT
                self._nbtSplit = True
            else:
                self._nrmCType = v
                self._nbtSplit = False

        def applyTo(self, d: "VertexDef"):
            d.psns[0].fmt.ctype = self.psnCType
            d.psns[0].fmt.dtype = self.psnDType
            d.psns[0].fmt.scale = self.psnScale
            d.nrms[0].fmt.ctype = self.nrmCType
            d.nrms[0].fmt.dtype = self.nrmDType
            for aDef, dt in zip(d.clrs, self.clrDTypes):
                aDef.fmt.dtype = dt
            d.uvs[0].fmt.ctype = self.uvCTypes[0]
            d.uvs[0].fmt.dtype = self.uvDTypes[0]
            d.uvs[0].fmt.scale = self.uvScales[0]

        def applyFrom(self, d: "VertexDef"):
            self.psnCType = d.psns[0].fmt.ctype
            self.psnDType = d.psns[0].fmt.dtype
            self.psnScale = d.psns[0].fmt.scale
            self.nrmCType = d.nrms[0].fmt.ctype
            self.nrmDType = d.nrms[0].fmt.dtype
            self.clrDTypes = (aDef.fmt.dtype for aDef in d.clrs)
            self.clrCTypes = (aDef.fmt.ctype for aDef in d.clrs)
            self.uvCTypes[0] = d.uvs[0].fmt.ctype
            self.uvDTypes[0] = d.uvs[0].fmt.dtype
            self.uvScales[0] = d.uvs[0].fmt.scale


class VertexFmt1(CPReg["VertexFmt1.ValStruct"]):
    """Second of 3 vertex attribute format registers."""

    VALID_ADDRESSES = (b"\x80", )

    bits: "ValStruct"
    class ValStruct(CPReg.ValStruct):
        _uvCType1 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType1 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale1 = Bits(5, int)
        _uvCType2 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType2 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale2 = Bits(5, int)
        _uvCType3 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType3 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale3 = Bits(5, int)
        _uvCType4 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType4 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _pad = Bits(1, bool, 1)
        uvCTypes = alias("_uvCType1", "_uvCType2", "_uvCType3", "_uvCType4")
        uvDTypes = alias("_uvDType1", "_uvDType2", "_uvDType3", "_uvDType4")
        uvScales = alias("_uvScale1", "_uvScale2", "_uvScale3")

        def applyTo(self, d: "VertexDef"):
            fmts = tuple(aDef.fmt for aDef in d.uvs)
            for fmt, ct, dt, s in zip(fmts[1:4], self.uvCTypes, self.uvDTypes, self.uvScales):
                fmt.ctype = ct
                fmt.dtype = dt
                fmt.scale = s
            fmts[4].ctype = self._uvCType4
            fmts[4].dtype = self._uvDType4

        def applyFrom(self, d: "VertexDef"):
            self.uvCTypes = (aDef.fmt.ctype for aDef in d.uvs[1:5])
            self.uvDTypes = (aDef.fmt.dtype for aDef in d.uvs[1:5])
            self.uvScales = (aDef.fmt.scale for aDef in d.uvs[1:4])


class VertexFmt2(CPReg["VertexFmt2.ValStruct"]):
    """Third of 3 vertex attribute format registers."""

    VALID_ADDRESSES = (b"\x90", )

    bits: "ValStruct"
    class ValStruct(CPReg.ValStruct):
        _uvScale4 = Bits(5, int)
        _uvCType5 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType5 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale5 = Bits(5, int)
        _uvCType6 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType6 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale6 = Bits(5, int)
        _uvCType7 = Bits(1, UVAttr.CompType, UVAttr.CompType.UV)
        _uvDType7 = Bits(3, UVAttr.DataType, UVAttr.DataType.UINT8)
        _uvScale7 = Bits(5, int)
        uvCTypes = alias("_uvCType5", "_uvCType6", "_uvCType7")
        uvDTypes = alias("_uvDType5", "_uvDType6", "_uvDType7")
        uvScales = alias("_uvScale4", "_uvScale5", "_uvScale6", "_uvScale7")

        def applyTo(self, d: "VertexDef"):
            fmts = tuple(aDef.fmt for aDef in d.uvs)
            fmts[4].scale = self._uvScale4
            for fmt, ct, dt, s in zip(fmts[5:8], self.uvCTypes, self.uvDTypes, self.uvScales[1:]):
                fmt.ctype = ct
                fmt.dtype = dt
                fmt.scale = s

        def applyFrom(self, d: "VertexDef"):
            self.uvCTypes = (aDef.fmt.ctype for aDef in d.uvs[5:8])
            self.uvDTypes = (aDef.fmt.dtype for aDef in d.uvs[5:8])
            self.uvScales = (aDef.fmt.scale for aDef in d.uvs[4:8])


class VertexDef():
    """Container for the definitions of every possible vertex attribute."""

    def __init__(self):
        self.psnMtcs: tuple[AttrDef[MtxAttrDec, None]]
        self.texMtcs: tuple[AttrDef[MtxAttrDec, None], ...]
        self.psnMtcs = tuple(AttrDef(dec=MtxAttrDec.NOT_PRESENT) for _ in range(MAX_PSN_MTX_ATTRS))
        self.texMtcs = tuple(AttrDef(dec=MtxAttrDec.NOT_PRESENT) for _ in range(MAX_TEX_MTX_ATTRS))
        self.psns = tuple(AttrDef(fmt=PsnAttr()) for _ in range(MAX_PSN_ATTRS))
        self.nrms = tuple(AttrDef(fmt=NrmAttr()) for _ in range(MAX_NRM_ATTRS))
        self.clrs = tuple(AttrDef(fmt=ClrAttr()) for _ in range(MAX_CLR_ATTRS))
        self.uvs = tuple(AttrDef(fmt=UVAttr()) for _ in range(MAX_UV_ATTRS))

    @property
    def attrs(self) -> list[AttrDef[AttrDec, VertexAttr]]:
        """List of all the attributes in this vertex definition."""
        return self.psnMtcs + self.texMtcs + self.psns + self.nrms + self.clrs + self.uvs

    @property
    def stride(self) -> int:
        """Size of one vertex following this definition, in bytes."""
        return sum(a.stride for a in self.attrs)

    def getDecs(self) -> list["LoadCP"]:
        """Get a list of vertex declaration commands for this definition."""
        cmds = []
        for reg in (VertexDec0(), VertexDec1()):
            reg.bits.applyFrom(self)
            cmds.append(LoadCP(reg))
        return cmds

    def getFmts(self) -> list["LoadCP"]:
        """Get a list of vertex format commands for this definition."""
        cmds = []
        for reg in (VertexFmt0(), VertexFmt1(), VertexFmt2()):
            reg.bits.applyFrom(self)
            cmds.append(LoadCP(reg))
        return cmds

    def getCounts(self):
        """Get a command with the attribute counts for this vertex definition."""
        counts = AttrCounts()
        n = self.nrms[0]
        counts.bits.nrms = 0 if n.dec is StdAttrDec.NOT_PRESENT else (2 if n.fmt.ctype.isNBT else 1)
        counts.bits.clrs = sum(d.dec is not StdAttrDec.NOT_PRESENT for d in self.clrs)
        counts.bits.uvs = sum(d.dec is not StdAttrDec.NOT_PRESENT for d in self.uvs)
        return LoadXF(counts)

    def getFlags(self):
        """Get an int made up of flags for which attributes of this vertex are enabled."""
        flags = 0
        for i, d in enumerate(self.attrs):
            flags |= ((d.dec not in (StdAttrDec.NOT_PRESENT, MtxAttrDec.NOT_PRESENT)) << i)
        return flags

    def read(self, data):
        """Process a sequence of GX commands using this vertex definition.

        This can process regular commands, but unlike gx.read(), it can also read draw commands.
        """
        if not isinstance(data, BytesIO):
            data = BytesIO(data) # io objects can be passed in directly, or you can use bytes
        vertStride = self.stride
        cmds = []
        drawSizes: dict[DrawPrimitives, int] = {}
        drawData = b""
        # read commands
        while True:
            opcode = data.read(1)
            if not opcode:
                break
            try:
                cmdtype = COMMANDS[opcode]
                if issubclass(cmdtype, DrawPrimitives):
                    numVerts = int.from_bytes(data.read(2), "big")
                    drawData += data.read(numVerts * vertStride)
                    cmd = cmdtype()
                    drawSizes[cmd] = numVerts
                else:
                    cmd = cmdtype.unpack(data)
            except KeyError as e: # invalid opcode
                raise ValueError(f"Invalid GX command opcode: {opcode}") from e
            cmds.append(cmd)
        # unpack vertex data as array w/ a row for every vertex & a column for every attribute byte
        drawData = np.frombuffer(drawData, dtype=np.uint8).reshape(-1, vertStride)
        # construct vertex data from the unpacked columns
        vertData = np.zeros((len(drawData), MAX_ATTRS), dtype=np.uint16)
        byteIdx = 0
        for attrIdx, attr in enumerate(self.attrs):
            aStride = attr.dec.stride
            if aStride == 0:
                continue
            # grab byte data & combine
            if aStride == 1: # optimization!
                vertData[:, attrIdx] = drawData[:, byteIdx]
            elif aStride > 1:
                dataSlice = np.ascontiguousarray(drawData[:, byteIdx : byteIdx + aStride])
                vertData[:, attrIdx] = dataSlice.view(f">u{aStride}")[:, 0]
            elif attr.dec is StdAttrDec.DIRECT:
                raise ValueError("Direct vertex data is unsupported, as it's not used by NW4R.")
            byteIdx += aStride
        vertIdx = 0
        # apply vertex data to cmds
        for cmd, numVerts in drawSizes.items():
            cmd.vertData = vertData[vertIdx : vertIdx + numVerts]
            vertIdx += numVerts
        return cmds

    def pack(self, cmds: list["Command"]) -> bytes:
        """Pack a sequence of GX commands using this vertex definition.

        This can pack regular commands, but unlike gx.read(), it can also pack draw commands.
        """
        packed: dict[Command, bytes] = {}
        drawSizes = {cmd: len(cmd) for cmd in cmds if isinstance(cmd, DrawPrimitives)}
        vertData = np.zeros((sum(drawSizes.values()), MAX_ATTRS), dtype=">u2")
        vertIdx = 0
        # pack commands
        for cmd in cmds:
            if isinstance(cmd, DrawPrimitives):
                numVerts = drawSizes[cmd]
                vertData[vertIdx : vertIdx + numVerts] = cmd.vertData
                vertIdx += numVerts
                packed[cmd] = None
            else:
                packed[cmd] = cmd.pack()
        # pack vertex data to array w/ row for each vertex & column for each attribute data byte
        totalVerts = vertIdx
        vertStride = self.stride
        data = np.zeros((totalVerts, vertStride), dtype=np.uint8)
        byteIdx = 0
        for i, attr in enumerate(self.attrs):
            if attr.dec is StdAttrDec.DIRECT:
                raise ValueError("Direct vertex data is unsupported, as it's not used by NW4R.")
            aStride = attr.dec.stride
            if aStride == 0:
                continue
            # grab byte data & break it down
            dataSlice = data[:, byteIdx : byteIdx + aStride]
            dataSlice[:] = np.ascontiguousarray(vertData[:, i : i + 1]).view(np.uint8)[:, -aStride:]
            byteIdx += aStride
        vertIdx = 0
        for cmd, numVerts in drawSizes.items():
            cmdData = data[vertIdx : vertIdx + numVerts].tobytes()
            packed[cmd] = cmd.OPCODE + numVerts.to_bytes(2, "big") + cmdData
            vertIdx += numVerts
        return b"".join(packed.values())


class XFReg(Register[_REG_STRUCT_T]):
    """Register in the transform unit."""


class AttrCounts(XFReg["AttrCounts.ValStruct"]):
    """Defines how many items are present for each vertex attribute type on each vertex."""

    VALID_ADDRESSES = (b"\x10\x08", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        clrs = Bits(2, int)
        nrms = Bits(2, int)
        uvs = Bits(4, int)
        _pad = Bits(24, int)


class TexCoordSettings(XFReg["TexCoordSettings.ValStruct"]):
    """Settings for one texture's coordinate generation."""

    VALID_ADDRESSES = (b"\x10\x40", b"\x10\x41", b"\x10\x42", b"\x10\x43",
                       b"\x10\x44", b"\x10\x45", b"\x10\x46", b"\x10\x47")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _pad0 = Bits(1, bool)
        projectionType = Bits(1, TexProjectionType, TexProjectionType.ST)
        inputForm = Bits(1, TexInputForm, TexInputForm.AB11)
        _pad1 = Bits(1, bool)
        texGenType = Bits(3, TexGen, TexGen.REGULAR)
        texCoordSrc = Bits(5, TexCoordSource, TexCoordSource.POSITION)
        embossSrc = Bits(3, int) # bump mapping texture reference index
        embossLight = Bits(3, int) # bump mapping light reference index
        _pad2 = Bits(14, int)


class PostEffectSettings(XFReg["PostEffectSettings.ValStruct"]):
    """Settings for an optional second texture coordinate transformation."""

    VALID_ADDRESSES = (b"\x10\x50", b"\x10\x51", b"\x10\x52", b"\x10\x53",
                       b"\x10\x54", b"\x10\x55", b"\x10\x56", b"\x10\x57")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        mtxIdx = Bits(6, int) # matrix to use (row index; should be multiple of 3 for matrix start)
        _pad0 = Bits(2, int)
        normalize = Bits(1, bool) # whether to renormalize coords before the second transformation
        _pad = Bits(23, int)


class BPReg(Register[_REG_STRUCT_T]):
    """Register in the blitting processor (blend processor? bypass raster state?) idk bruh"""


class IndMtxSettings(BPReg["IndMtxSettings.ValStruct"]):
    """Holds part of one of three 3x3 indirect texture matrices in BP memory.

    Check the implementation for a ton of comments that explain what this register's settings mean.
    """

    # matrix index = self.idx // 3
    # explanation of items & scale:

    # if the matrix layout is something like:
    # A C E
    # B D F
    # 0 0 1 (last row not set by commands, always 0 0 1)
    # if idx % 3 is 0, items are A and B
    # if idx % 3 is 1, items are C and D
    # if idx % 3 is 2, items are E and F
    # items are stored as signed 10-bit numbers from 0-1 (11 bits including sign),
    # then multiplied by 2^[full matrix scale - 17]

    # note that the 0-1 range isn't inclusive on top; technically, the max is 1-epsilon
    # because of this, there's some math in the props to make things work w/ bitstruct normalization

    # full matrix scale:
    # each command stores a small component of it
    # if full matrix scale in binary is laid out like ABCDEF:
    # if idx % 3 is 0, scale is EF
    # if idx % 3 is 1, scale is CD
    # if idx % 3 is 2, scale is AB

    VALID_ADDRESSES = (b"\x06", b"\x07", b"\x08",
                       b"\x09", b"\x0a", b"\x0b",
                       b"\x0c", b"\x0d", b"\x0e")

    SCALE_MIN = -17
    EPSILON = 1 / maxBitVal(10)
    MAX_VAL = 1 - EPSILON

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _val0 = NormalizedSignedBits(11, float)
        _val1 = NormalizedSignedBits(11, float)
        scale = Bits(2, int)

        @property
        def _item0(self):
            return self._val0 * IndMtxSettings.MAX_VAL

        @property
        def _item1(self):
            return self._val1 * IndMtxSettings.MAX_VAL

        @_item0.setter
        def _item0(self, value):
            self._val0 = value / IndMtxSettings.MAX_VAL

        @_item1.setter
        def _item1(self, value):
            self._val1 = value / IndMtxSettings.MAX_VAL

        items = alias("_item0", "_item1")


class TEVIndSettings(BPReg["TEVIndSettings.ValStruct"]):
    """Indirect texture settings for one TEV stage."""

    VALID_ADDRESSES = (b"\x10", b"\x11", b"\x12", b"\x13", b"\x14", b"\x15", b"\x16", b"\x17",
                       b"\x18", b"\x19", b"\x1a", b"\x1b", b"\x1c", b"\x1d", b"\x1e", b"\x1f")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        indirectID = Bits(2, int) # indirect texture slot (0-3)
        format = Bits(2, ColorBitSel, ColorBitSel.ALL_8) # which bits of each color channel are used
        biasS = Bits(1, bool) # which texture coords should be affected by bias
        biasT = Bits(1, bool)
        biasU = Bits(1, bool)
        # on selected coord component (aka color channel; remember these are indirect textures),
        # bits excluded by format are used for bump alpha; if format is all 8, upper 5 are used
        bumpAlphaComp = Bits(2, TexCoordSel, TexCoordSel.NONE)
        mtxIdx = Bits(2, IndMtxIdx, IndMtxIdx.NONE) # used indirect matrix index
        mtxType = Bits(2, IndMtxType, IndMtxType.STATIC) # used indirect matrix type
        wrapS = Bits(3, IndWrapSet, IndWrapSet.OFF) # wrap for each texture coord
        wrapT = Bits(3, IndWrapSet, IndWrapSet.OFF)
        utcLOD = Bits(1, bool) # if true, unmodified texture coords are used for lod calculations
        addPrev = Bits(1, bool) # if true, output from previous stage is added to texture coords
        _pad = Bits(3, int)


class IndCoordScale(BPReg["IndCoordScale.ValStruct"]):
    """Scale for each dimension of each indirect texture (2 textures per command).

    Texture coordinates are scaled by 2^(value stored here * -1).
    """

    VALID_ADDRESSES = (b"\x25", b"\x26")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _s0 = Bits(4, IndCoordScalar)
        _t0 = Bits(4, IndCoordScalar)
        _s1 = Bits(4, IndCoordScalar)
        _t1 = Bits(4, IndCoordScalar)
        _pad = Bits(8, int)

        @property
        def scales(self):
            return ((self._s0, self._t0), (self._s1, self._t1))

        @scales.setter
        def scales(self, v: list[tuple[IndCoordScalar, IndCoordScalar]]):
            self._s0, self._t0, self._s1, self._t1 = (c for coords in v for c in coords)


class IndSources(BPReg["IndSources.ValStruct"]):
    """Texture indices & coord indices for each of the 4 indirect textures."""

    VALID_ADDRESSES = (b"\x27", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _texIdx0 = Bits(3, int)
        _texCoordIdx0 = Bits(3, int)
        _texIdx1 = Bits(3, int)
        _texCoordIdx1 = Bits(3, int)
        _texIdx2 = Bits(3, int)
        _texCoordIdx2 = Bits(3, int)
        _texIdx3 = Bits(3, int)
        _texCoordIdx3 = Bits(3, int)
        texIdcs = alias("_texIdx0", "_texIdx1", "_texIdx2", "_texIdx3")
        texCoordIdcs = alias("_texCoordIdx0", "_texCoordIdx1", "_texCoordIdx2", "_texCoordIdx3")


class TEVSources(BPReg["TEVSources.ValStruct"]):
    """Texture, color, and coord sources for 2 TEV stages."""

    VALID_ADDRESSES = (b"\x28", b"\x29", b"\x2a", b"\x2b", b"\x2c", b"\x2d", b"\x2e", b"\x2f")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _texIdx0 = Bits(3, int)
        _texCoordIdx0 = Bits(3, int)
        _texEnable0 = Bits(1, bool)
        _rasterSel0 = Bits(3, TEVRasterSel, TEVRasterSel.COLOR_0)
        _pad0 = Bits(2, int)
        _texIdx1 = Bits(3, int)
        _texCoordIdx1 = Bits(3, int)
        _texEnable1 = Bits(1, bool)
        _rasterSel1 = Bits(3, TEVRasterSel, TEVRasterSel.COLOR_0)
        _pad1 = Bits(2, int)
        texIdcs = alias("_texIdx0", "_texIdx1")
        texEnables = alias("_texEnable0", "_texEnable1")
        texCoordIdcs = alias("_texCoordIdx0", "_texCoordIdx1")
        rasterSels = alias("_rasterSel0", "_rasterSel1")


class DepthSettings(BPReg["DepthSettings.ValStruct"]):
    """Settings regarding the Z buffer."""

    VALID_ADDRESSES = (b"\x40", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        enable = Bits(1, bool, True)
        depthOp = Bits(3, CompareOp, CompareOp.LEQUAL)
        updateDepth = Bits(1, bool, True) # whether to write new values when test enabled
        _pad = Bits(19, int)


class BlendSettings(BPReg["BlendSettings.ValStruct"]):
    """Settings regarding alpha blending."""

    VALID_ADDRESSES = (b"\x41", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        enableBlend = Bits(1, bool) # priority 2, src * srcfac + dst * dstfac
        enableLogic = Bits(1, bool) # priority 3, logic operation between src and dst
        enableDither = Bits(1, bool)
        updateColor = Bits(1, bool)
        updateAlpha = Bits(1, bool)
        dstFactor = Bits(3, BlendDstFactor, BlendDstFactor.INV_SRC_ALPHA)
        srcFactor = Bits(3, BlendSrcFactor, BlendSrcFactor.SRC_ALPHA)
        subtract = Bits(1, bool) # priority 1, dst - source (factors ignored in this case)
        logic = Bits(4, BlendLogicOp, BlendLogicOp.CLEAR)
        _pad = Bits(8, int)


class ConstAlphaSettings(BPReg["ConstAlphaSettings.ValStruct"]):
    """Constant alpha value that overrides alpha output from blending if enabled."""

    VALID_ADDRESSES = (b"\x42", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        value = NormalizedBits(8, int)
        enable = Bits(1, bool)
        _pad = Bits(15, int)


class TEVColorParams(BPReg["TEVColorParams.ValStruct"]):
    """Parameters for one TEV stage's color calculation."""

    VALID_ADDRESSES = (b"\xc0", b"\xc2", b"\xc4", b"\xc6", b"\xc8", b"\xca", b"\xcc", b"\xce",
                       b"\xd0", b"\xd2", b"\xd4", b"\xd6", b"\xd8", b"\xda", b"\xdc", b"\xde")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        argD = Bits(4, TEVColorArg, TEVColorArg.ZERO)
        argC = Bits(4, TEVColorArg, TEVColorArg.ZERO)
        argB = Bits(4, TEVColorArg, TEVColorArg.ZERO)
        argA = Bits(4, TEVColorArg, TEVColorArg.ZERO)
        bias = Bits(2, TEVBias, TEVBias.ZERO)
        op = Bits(1, TEVOp, TEVOp.ADD_OR_GREATER)
        clamp = Bits(1, bool)
        _sc = Bits(2, TEVScaleChan, TEVScaleChan.ONE_OR_R) # channel in comp mode, scale otherwise
        output = Bits(2, int) # color index for stage output (0 is final, 1-3 are standard regs)
        scale = compareMode = alias("_sc")
        args = alias("argA", "argB", "argC", "argD")


class TEVAlphaParams(BPReg["TEVAlphaParams.ValStruct"]):
    """Parameters for one TEV stage's alpha calculation."""

    VALID_ADDRESSES = (b"\xc1", b"\xc3", b"\xc5", b"\xc7", b"\xc9", b"\xcb", b"\xcd", b"\xcf",
                       b"\xd1", b"\xd3", b"\xd5", b"\xd7", b"\xd9", b"\xdb", b"\xdd", b"\xdf")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        rasterSwapIdx = Bits(2, int) # color swap setting to use for raster color
        textureSwapIdx = Bits(2, int) # color swap setting for texture color
        argD = Bits(3, TEVAlphaArg, TEVAlphaArg.ZERO)
        argC = Bits(3, TEVAlphaArg, TEVAlphaArg.ZERO)
        argB = Bits(3, TEVAlphaArg, TEVAlphaArg.ZERO)
        argA = Bits(3, TEVAlphaArg, TEVAlphaArg.ZERO)
        bias = Bits(2, TEVBias, TEVBias.ZERO)
        op = Bits(1, TEVOp, TEVOp.ADD_OR_GREATER)
        clamp = Bits(1, bool)
        _sc = Bits(2, TEVScaleChan, TEVScaleChan.ONE_OR_R) # channel in comp mode, scale otherwise
        output = Bits(2, int) # color index for stage output (0 is final, 1-3 are standard regs)
        scale = compareMode = alias("_sc")
        args = alias("argA", "argB", "argC", "argD")


class AlphaTestSettings(BPReg["AlphaTestSettings.ValStruct"]):
    """Settings for a test used to discard pixels based on alpha value.

    Test: if not (alpha comp1 value1) logic (alpha comp2 value2) then discard
    """

    VALID_ADDRESSES = (b"\xf3", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _value0 = NormalizedBits(8, float)
        _value1 = NormalizedBits(8, float)
        _comp0 = Bits(3, CompareOp, CompareOp.ALWAYS)
        _comp1 = Bits(3, CompareOp, CompareOp.ALWAYS)
        logic = Bits(2, AlphaLogicOp, AlphaLogicOp.AND)
        values = alias("_value0", "_value1")
        comps = alias("_comp0", "_comp1")


class TEVColorSettings(BPReg["TEVColorSettings.ValStruct"]):
    """Constant color & alpha indices used for 2 TEV stages, and color swap table data.

    The color swap table is a set of four color swap settings.
    A color swap defines mappings for a color's R, G, B, and A values.
    Each TEV stage uses a color swap (set in alpha params) on its texture & raster color inputs.
    A standard one just goes RGBA, but if you switch it to BGBA for example, then both the
    red and blue channels will use the blue channel's value.
    Each ColorSettings command is responsible for two table entries.
    So, the first one (address f6) controls the first R and G. The second controls the first BA.
    Third does the second RG, fourth the second BA, etc.
    """

    VALID_ADDRESSES = (b"\xf6", b"\xf7", b"\xf8", b"\xf9", b"\xfa", b"\xfb", b"\xfc", b"\xfd")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        # swaps: odd cmds refer define swaps for r and g, even are for b and a
        _swapRB = Bits(2, ColorChannel, ColorChannel.R)
        _swapGA = Bits(2, ColorChannel, ColorChannel.R)
        swapR = swapB = alias("_swapRB")
        swapG = swapA = alias("_swapGA")
        # used constant color & alpha indices for 2 tev stages
        _constColorSel0 = Bits(5, TEVConstSel, TEVConstSel.VAL_8_8)
        _constAlphaSel0 = Bits(5, TEVConstSel, TEVConstSel.VAL_8_8)
        _constColorSel1 = Bits(5, TEVConstSel, TEVConstSel.VAL_8_8)
        _constAlphaSel1 = Bits(5, TEVConstSel, TEVConstSel.VAL_8_8)
        constColorSels = alias("_constColorSel0", "_constColorSel1")
        constAlphaSels = alias("_constAlphaSel0", "_constAlphaSel1")


class TEVColorReg(BPReg["TEVColorReg.ValStruct"]):
    """Values for 2 channels of a TEV color register entry."""

    VALID_ADDRESSES = (b"\xe0", b"\xe1", b"\xe2", b"\xe3", b"\xe4", b"\xe5", b"\xe6", b"\xe7")

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        _rb = NormalizedBits(8, float) # r for even address index, b for odd
        _pad0 = Bits(4, int)
        _ag = NormalizedBits(8, float) # a for even address index, g for odd
        _pad1 = Bits(3, int)
        isConstant = Bits(1, bool) # if true, this is for a const color; otherwise, it's standard
        r = b = alias("_rb")
        a = g = alias("_ag")


class BPMask(BPReg["BPMask.ValStruct"]):
    """Bit mask to be applied to the value for the next BP command in a command sequence."""

    VALID_ADDRESSES = (b"\xfe", )

    bits: "ValStruct"
    class ValStruct(Register.ValStruct):
        mask = Bits(24, int)


class Command():
    """GX command identified by an opcode."""
    OPCODE: bytes


class ContainedCommand(Command):
    """GX command that can be packed & unpacked to/from bytes without any extra data needed."""

    @classmethod
    def unpack(cls, b: BytesIO):
        """Unpack command from a BytesIO object (starting after the opcode)."""
        return cls()

    def pack(self):
        """Pack command to bytes."""
        return self.OPCODE


class NOP(ContainedCommand):
    """Blank command for padding; doesn't do anything, but supports any number of padding bytes"""

    OPCODE = b"\x00"

    def __init__(self, size=1):
        self.size = size

    @classmethod
    def unpack(cls, b: BytesIO):
        return cls(1)

    def pack(self):
        return super().pack() * self.size


class LoadReg(ContainedCommand, Generic[_REG_T]):
    """Load data into some register in one of the Wii GPU's components (CP, XF, BP)."""

    REGISTERS: tuple[type[_REG_T], ...] # all register types in cmd's corresponding gx component

    def __init__(self, reg: _REG_T = None):
        self.reg = reg

    @property
    def reg(self) -> _REG_T:
        return self._reg

    @reg.setter
    def reg(self, v):
        if v is not None and type(v) not in self.REGISTERS:
            raise TypeError(f"Failed to set {type(self).__name__} command register, as register "
                            f"type '{type(v).__name__}' is invalid for this command")
        self._reg = v

    @classmethod
    @cache
    def _regAddrs(cls):
        return {addr: reg for reg in cls.REGISTERS for addr in reg.VALID_ADDRESSES}

    @classmethod
    @cache
    def regFromAddr(cls, addr: bytes):
        """Get a register type for this command type from its address.

        Raises a ValueError if address is invalid for this command type.
        """
        try:
            return cls._regAddrs()[addr]
        except KeyError as e:
            raise ValueError(f"Invalid {cls.__name__} command register address: {addr}") from e

    @staticmethod
    def filterRegs(cmds: Iterable["LoadReg"], regType: type[_REG_T]) -> tuple[_REG_T, ...]:
        """Take an iterable of cmds & return a tuple w/ the regs of all cmds using some reg type."""
        return tuple(cmd.reg for cmd in cmds if isinstance(cmd.reg, regType))


class LoadCP(LoadReg[CPReg]):
    """Load data into a CP register."""

    OPCODE = b"\x08"

    REGISTERS = (VertexDec0, VertexDec1, VertexFmt0, VertexFmt1, VertexFmt2)

    @classmethod
    def unpack(cls, b: BytesIO):
        newCmd: cls = super().unpack(b)
        addr = b.read(1)
        data = b.read(4)
        regType = cls.regFromAddr(addr)
        newCmd.reg = regType(addr=addr, b=data)
        return newCmd

    def pack(self):
        return super().pack() + self.reg.addr + self.reg.bits.pack()


class LoadXF(LoadReg[XFReg]):
    """Load data into an XF register."""

    OPCODE = b"\x10"

    REGISTERS = (AttrCounts, TexCoordSettings, PostEffectSettings)

    @classmethod
    def unpack(cls, b: BytesIO):
        newCmd: cls = super().unpack(b)
        dataSize = (int.from_bytes(b.read(2), "big") + 1) * 4 # stored as (size in 4-byte units) - 1
        addr = b.read(2)
        data = b.read(dataSize)
        regType = cls.regFromAddr(addr)
        newCmd.reg = regType(addr=addr, b=data)
        return newCmd

    def pack(self):
        dataSize = self.reg.bits.size // 4 - 1 # stored as (size in 4-byte units) - 1
        return super().pack() + dataSize.to_bytes(2, "big") + self.reg.addr + self.reg.bits.pack()


class LoadIndexedXF(ContainedCommand):
    """Load an indexed XF object."""

    _MEM_START = 0

    class _Struct(BitStruct):
        addr = Bits(12, int)
        length = Bits(4, int)
        idx = Bits(16, int)

    def __init__(self, mtxIdx = 0, length = 11, idx = 0):
        self.mtxIdx = mtxIdx # address in xf memory to load to
        self.length = length # length of data (bytes) - 1
        self.idx = idx # index in some external structure to load from

    @classmethod
    def unpack(cls, b: BytesIO):
        newCmd: cls = super().unpack(b)
        data = cls._Struct.unpack(b.read(4))
        newCmd.addr = data.addr
        newCmd.length = data.length
        newCmd.idx = data.idx
        return newCmd

    def pack(self):
        s = self._Struct()
        s.addr = self.addr
        s.length = self.length
        s.idx = self.idx
        return super().pack() + s.pack()

    @property
    def mtxIdx(self):
        """Index of the matrix in XF memory that this command's address points to."""
        return self.addrToIdx(self.addr)

    @mtxIdx.setter
    def mtxIdx(self, i):
        self.addr = self.idxToAddr(i)

    @classmethod
    @abstractmethod
    def addrToIdx(cls, addr: np.ndarray) -> np.ndarray:
        """Get the index of a matrix for this type given its address in XF memory."""

    @classmethod
    @abstractmethod
    def idxToAddr(cls, idx: np.ndarray) -> np.ndarray:
        """Get the XF address for the matrix at some index for this type."""


class LoadPsnMtx(LoadIndexedXF):

    OPCODE = b"\x20"

    @classmethod
    def addrToIdx(cls, addr: np.ndarray):
        return addr // 12

    @classmethod
    def idxToAddr(cls, idx: np.ndarray):
        return idx * 12


class LoadNrmMtx(LoadIndexedXF):

    OPCODE = b"\x28"

    def __init__(self, mtxIdx = 0, length = 8, idx = 0):
        super().__init__(mtxIdx, length, idx)

    @classmethod
    def addrToIdx(cls, addr: np.ndarray):
        return (addr - 1024) // 9

    @classmethod
    def idxToAddr(cls, idx: np.ndarray):
        return idx * 9 + 1024


class LoadTexMtx(LoadIndexedXF):

    OPCODE = b"\x30"

    @classmethod
    def addrToIdx(cls, addr: np.ndarray):
        return (addr - 120) // 12

    @classmethod
    def idxToAddr(cls, idx: np.ndarray):
        return idx * 12 + 120


class LoadLightObj(LoadIndexedXF): # pylint: disable=abstract-method

    OPCODE = b"\x38"


class LoadBP(LoadReg[BPReg]):
    """Load data into a BP register."""

    OPCODE = b"\x61"

    REGISTERS = (IndMtxSettings, TEVIndSettings, IndCoordScale, IndSources,
                 TEVSources, DepthSettings, BlendSettings, ConstAlphaSettings,
                 TEVColorParams, TEVAlphaParams, AlphaTestSettings, TEVColorSettings,
                 TEVColorReg, BPMask)

    @classmethod
    def unpack(cls, b: BytesIO):
        newCmd: cls = super().unpack(b)
        addr = b.read(1)
        regType = cls.regFromAddr(addr)
        data = b.read(3)
        newCmd.reg = regType(addr=addr, b=data)
        return newCmd

    def pack(self):
        return super().pack() + self.reg.addr + self.reg.bits.pack()


class DrawPrimitives(Command):
    """Draw primitives to the screen through a list of vertices."""

    OPCODE: bytes
    _faceCache = np.ndarray((0, 0))

    def __init__(self, numVerts = 0, vertData: np.ndarray = None):
        if vertData is None:
            self._vertData = np.zeros((numVerts, MAX_ATTRS), dtype=np.uint16)
        else:
            self.vertData = vertData

    def _getAttr(self, attrIdx: int, numAttrs: int):
        """Get vertex data from the attribute index and # per vertex."""
        return self._vertData[:, attrIdx : attrIdx + numAttrs].T

    def _setAttr(self, attrIdx: int, numAttrs: int, data: np.ndarray):
        """Set vertex data from the attribute index, # per vertex, & new data."""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.uint16)
        dataLen = data.shape[1]
        selfLen = len(self)
        if dataLen != selfLen:
            raise ValueError(f"Incorrect vertex data length (expected {selfLen}, got {dataLen})")
        self._vertData[:, attrIdx : attrIdx + numAttrs] = data.astype(np.uint16).T

    def _attrProp(attrIdx: int, numAttrs: int): # pylint: disable=no-self-argument
        """Property for an attribute in this command's vertex data."""
        return property(
            fget=lambda self: self._getAttr(attrIdx, numAttrs),
            fset=lambda self, v: self._setAttr(attrIdx, numAttrs, v)
        )

    psnMtcs: np.ndarray = _attrProp(PSN_MTX_ATTR_IDX, MAX_PSN_MTX_ATTRS)
    texMtcs: np.ndarray = _attrProp(TEX_MTX_ATTR_IDX, MAX_TEX_MTX_ATTRS)
    psns: np.ndarray = _attrProp(PSN_ATTR_IDX, MAX_PSN_ATTRS)
    nrms: np.ndarray = _attrProp(NRM_ATTR_IDX, MAX_NRM_ATTRS)
    clrs: np.ndarray = _attrProp(CLR_ATTR_IDX, MAX_CLR_ATTRS)
    uvs: np.ndarray = _attrProp(UV_ATTR_IDX, MAX_UV_ATTRS)

    @property
    def vertData(self):
        """Directly access this command's vertex data.

        This data is a 2D array with a row for each vertex and a column for each possible attribute.
        (The attribute properties like psns and clrs are views into this data)
        """
        return self._vertData

    @vertData.setter
    def vertData(self, data: np.ndarray):
        if data.shape[1] != MAX_ATTRS:
            raise TypeError("Vertex data must have a column for every possible attribute")
        if data.dtype != np.uint16:
            raise TypeError("Vertex data must be an array of unsigned 16-bit integer indices")
        self._vertData = data

    def __len__(self):
        return len(self._vertData)

    @classmethod
    def maxLen(cls) -> int:
        """Maximum length in vertices allowed for this command type."""
        return maxBitVal(16)

    @classmethod
    @abstractmethod
    def _genFaces(cls, numFaces: int) -> np.ndarray:
        """Generate faces for this command type, made up of vertex indices."""

    def faces(self) -> np.ndarray:
        """Faces for this command, made up of vertex indices."""
        cls = type(self)
        numFaces = self.numFaces()
        if numFaces > len(cls._faceCache):
            cls._faceCache = cls._genFaces(numFaces)
            cls._faceCache.flags.writeable = False
        return cls._faceCache[:numFaces]

    @abstractmethod
    def numFaces(self) -> int:
        """Number of faces in this draw command."""


class DrawQuads(DrawPrimitives):
    OPCODE = b"\x80"
    @classmethod
    def maxLen(cls):
        maxLen = super().maxLen()
        return maxLen - (maxLen % 4)
    @classmethod
    def _genFaces(cls, numFaces: int):
        return np.arange(numFaces * 4).reshape(-1, 4)
    def numFaces(self):
        return len(self) // 4


class DrawTriangles(DrawPrimitives):
    OPCODE = b"\x90"
    @classmethod
    def maxLen(cls):
        maxLen = super().maxLen()
        return maxLen - (maxLen % 3)
    @classmethod
    def _genFaces(cls, numFaces: int):
        return np.arange(numFaces * 3).reshape(-1, 3)
    def numFaces(self):
        return len(self) // 3

    def strip(self):
        """Return a list of draw commands equivalent to this one w/ tristrips for compression."""
        stripped: list[DrawPrimitives] = []
        verts, vertIdcs = np.unique(self.vertData, return_inverse=True, axis=0)
        strips = tristrip(vertIdcs.reshape(-1, 3).tolist(), DrawTriangleStrip.maxLen())
        soloTris = []
        for strip in strips:
            if len(strip) == 3:
                soloTris += strip
            else:
                stripCmd = DrawTriangleStrip(vertData=verts[strip])
                stripped.append(stripCmd)
        # compile isolated triangles into their own command
        # (multiple commands if too many to fit into one)
        numSoloVerts = len(soloTris)
        soloVertData = verts[soloTris]
        maxTriLen = DrawTriangles.maxLen()
        for vertStart in range(0, numSoloVerts, maxTriLen):
            vertEnd = min(vertStart + maxTriLen, numSoloVerts)
            soloCmd = DrawTriangles(vertData=soloVertData[vertStart:vertEnd])
            stripped.append(soloCmd)
        return stripped


class DrawTriangleStrip(DrawPrimitives):
    OPCODE = b"\x98"
    @classmethod
    def _genFaces(cls, numFaces: int):
        faces = np.arange(numFaces).reshape(-1, 1) # start w/ column of increasing indices
        faces = faces + np.arange(3).reshape(1, 3) # add [0 1 2] to each one - [[0 1 2] [1 2 3]] etc
        faces[1::2, :] = faces[1::2, ::-1] # flip every other - [[0 1 2] [3 2 1] [2 3 4]] etc
        return faces
    def numFaces(self):
        return len(self) - 2


class DrawTriangleFan(DrawPrimitives):
    OPCODE = b"\xa0"
    @classmethod
    def _genFaces(cls, numFaces: int):
        faces = np.arange(numFaces).reshape(-1, 1) # start w/ column of increasing indices
        faces = faces + np.arange(3).reshape(1, 3) # add [0 1 2] to each one - [[0 1 2] [1 2 3]] etc
        faces[:, 0] = 0 # first index is always 0 - [[0 1 2] [0 2 3] [0 3 4]] etc
        return faces
    def numFaces(self):
        return len(self) - 2


class DrawLines(DrawPrimitives):
    OPCODE = b"\xa8"
    @classmethod
    def maxLen(cls):
        maxLen = super().maxLen()
        return maxLen - (maxLen % 2)
    @classmethod
    def _genFaces(cls, numFaces: int):
        return np.ndarray((0, 0))
    def numFaces(self):
        return 0


class DrawLineStrip(DrawPrimitives):
    OPCODE = b"\xb0"
    @classmethod
    def _genFaces(cls, numFaces: int):
        return np.ndarray((0, 0))
    def numFaces(self):
        return 0


class DrawPoints(DrawPrimitives):
    OPCODE = b"\xb8"
    @classmethod
    def _genFaces(cls, numFaces: int):
        return np.ndarray((0, 0))
    def numFaces(self):
        return 0


def tristrip(tris: list[tuple[int, int, int]], maxLen: int = None):
    """Convert triangles (tuples w/ 3 vertex indices) to strips (lists of vertex indices)."""
    # this is a basic implementation that pretty much makes random lines until it can't anymore
    # the result's not too shabby though!
    strips: list[list[int]] = []
    edgeAdjacentVerts: dict[tuple[int, int], list[int]] = {}
    # create map from edges to the all their adjacent vertices
    for tri in tris:
        edgeAdjacentVerts.setdefault((tri[0], tri[1]), []).append(tri[2])
        edgeAdjacentVerts.setdefault((tri[1], tri[2]), []).append(tri[0])
        edgeAdjacentVerts.setdefault((tri[2], tri[0]), []).append(tri[1])
    # create strips by picking arbitrary starting points and then going down arbitrary paths
    while edgeAdjacentVerts:
        firstEdge, firstEdgeAdjacentVerts = edgeAdjacentVerts.popitem() # pop to get edge fast
        edgeAdjacentVerts[firstEdge] = firstEdgeAdjacentVerts # add back so item isn't removed
        strip = list(firstEdge)
        strips.append(strip)
        expandTristrip(strip, edgeAdjacentVerts, maxLen)
        # after initial strip expansion, expand it in the opposite direction as well
        # to do this, reverse the strip, then expand it
        # then, if reversing flipped the faces (which happens if the strip has an odd length),
        # we have to flip them back by adding an extra vert to the beginning
        # isFlipped = len(strip) % 2
        # strip.reverse()
        # expandTristrip(strip, edgeAdjacentVerts, maxLen, isReversed=True)
        # if isFlipped:
        #     if len(strip) % 2:
        #         strip.reverse()
        #     else:
        #         strip.insert(0, strip[0]) # reverse doesn't flip faces; only way is extra vert
        # COMMENTED OUT FOR NOW BECAUSE THE INSERTION IN THE LINE ABOVE MAKES REVERSING NOT
        # WORTH IT (compression gains are balanced out by the extra vertices)
    return strips


def expandTristrip(strip: list[int], edgeAdjacentVerts: dict[tuple[int, int], list[int]],
                   maxLen: int = None, isReversed = False):
    """Expand a triangle strip forwards until it can't be expanded anymore."""
    # order alternates with every entry (clockwise vs counter)
    # isReversed determines which order to start with
    doReverse = isReversed
    latestEdge = tuple(strip[-2:])
    noMaxLen = maxLen is None
    while latestEdge in edgeAdjacentVerts and (noMaxLen or len(strip) < maxLen):
        adjacentVerts = edgeAdjacentVerts[latestEdge]
        newVert = adjacentVerts.pop() # pop one vert adjacent to this edge
        strip.append(newVert)
        tri = (*latestEdge, newVert) * 2
        for edgeIdx in range(1, 3):
            # in addition to deleting the data for this edge, delete the data for
            # equivalent edges/adjacent verts
            # for instance, the tri (1, 2, 3) will have an entry for (1, 2) to 3,
            # (2, 3) to 1, and (3, 1) to 2; if we're looking at the (1, 2) edge, the
            # other entries will still need to be popped as well since they represent
            # the same tri
            offsetEdge = tri[edgeIdx : edgeIdx + 2]
            offsetAdjacentVerts = edgeAdjacentVerts[offsetEdge]
            offsetAdjacentVerts.remove(tri[edgeIdx + 2])
            if not offsetAdjacentVerts:
                del edgeAdjacentVerts[offsetEdge]
        if not adjacentVerts:
            try:
                del edgeAdjacentVerts[latestEdge]
            except KeyError:
                # we can only get here if the entry for this edge was deleted in the above loop,
                # i.e., two of this triangle's edges are the same
                # (i.e., the triangle is actually just a line)
                pass
        doReverse = not doReverse
        latestEdge = tuple(strip[-2:][::-1]) if doReverse else tuple(strip[-2:])


COMMANDS = {cmd.OPCODE: cmd for cmd in (
    NOP,
    LoadCP,
    LoadXF,
    LoadBP,
    LoadPsnMtx,
    LoadNrmMtx,
    LoadTexMtx,
    LoadLightObj,
    DrawQuads,
    DrawTriangles,
    DrawTriangleStrip,
    DrawTriangleFan,
    DrawLines,
    DrawLineStrip,
    DrawPoints
)}


def read(data) -> list[ContainedCommand]:
    """Process a sequence of GX command bytes into a list of commands.

    This does not work with vertex data, as that requires a vertex definition. For vertex data, use
    a vertex definition's read() method (which will also process any extra commands).
    """
    if not isinstance(data, BytesIO):
        data = BytesIO(data) # io objects can be passed in directly, or you can use bytes
    cmds = []
    nop = NOP.OPCODE
    while True:
        opcode = data.read(1)
        if not opcode:
            break
        if opcode == nop: # just skip nops for optimization
            continue
        try:
            cmdtype = COMMANDS[opcode]
            cmd = cmdtype.unpack(data)
        except TypeError as e: # draw command
            raise ValueError(f"Cannot process command with opcode {opcode}, as it contains vertex "
                             f"data. To read this command, use a VertexDef's read() method.") from e
        except KeyError as e: # invalid opcode
            raise ValueError(f"Invalid GX command opcode: {opcode}") from e
        cmds.append(cmd)
    return cmds

def pack(cmds: list[ContainedCommand]) -> bytes:
    """Pack a sequence of GX commands to bytes.

    This does not work with vertex data, as that requires a vertex definition. For vertex data, use
    a vertex definition's pack() method (which will also process any extra commands).
    """
    return b"".join(c.pack() for c in cmds)
