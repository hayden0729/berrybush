# standard imports
from typing import TypeVar
# internal imports
from .binaryutils import maxBitVal, bitsToBytes


T = TypeVar("T")


class _Bits:
    """Defines the format of one BitStruct field. See BitStruct for details."""

    def __init__(self, size: int, dtype: type, default = 0):
        self.size = size
        self.dtype = dtype
        self.mask = maxBitVal(self.size)
        self.maxVal = self.mask
        self.minVal = 0
        self.default = default

    def outOfRange(self, v: int):
        """Raise an error for an out-of-range value."""
        raise ValueError(f"Invalid value for BitStruct field (value is {v}, "
                         f"but format only allows {self.minVal} to {self.maxVal})")

    def checkRange(self, v: int):
        """Verify that this format can represent some value."""
        return self.minVal <= v and v <= self.maxVal

    def unpack(self, v: int):
        """Unpack bits for this format."""
        return self.dtype(v)

    def pack(self, v):
        """Pack a value for this format to bits."""
        v = int(v)
        if not self.checkRange(v):
            self.outOfRange(v)
        return v


class SignedBitsMixin():

    def __init__(self, size: int, dtype: type, default = 0):
        super().__init__(size, dtype, default)
        self.maxVal = maxBitVal(self.size - 1)
        self.minVal = -self.maxVal - 1

    def unpack(self, v: int):
        """Unpack bits for this format."""
        if v & (1 << (self.size - 1)):
            # read negative number from binary
            v = -(self.mask & ~v + 1)
        return super().unpack(v)

    def pack(self, v):
        """Pack a value for this format to bits."""
        v = super().pack(v)
        if v < 0:
            # get two's complement & add sign bit
            # (while keeping value positive in python so that it can be packed properly)
            v = self.mask & ~abs(v) + 1
        return v


class NormalizedBitsMixin():

    def __init__(self, size: int, dtype: type, default = 0):
        super().__init__(size, dtype, default)

    def outOfRange(self, v: int):
        raise ValueError(f"Invalid value for BitStruct field (value is {v / self.maxVal}, "
                         f"but format only allows {self.minVal / self.maxVal} to {1})")

    def unpack(self, v: int):
        """Unpack bits for this format."""
        return super().unpack(v) / self.maxVal

    def pack(self, v):
        """Pack a value for this format to bits."""
        return super().pack(v * self.maxVal)


class _SignedBits(SignedBitsMixin, _Bits):
    """Bits that support signed values (which lowers the maximum possible value!)."""

class _NormalizedBits(NormalizedBitsMixin, _Bits):
    """Bits that should be interpreted as normalized to the range 0 to 1."""

class _NormalizedSignedBits(NormalizedBitsMixin, SignedBitsMixin, _Bits):
    """Bits that should be interpreted as normalized to the range -1 to 1."""


# fake constructors for Bits classes so we can have type hinting for dtype

def Bits(size: int, dtype: type[T], default = 0) -> T:
    return _Bits(size, dtype, default)

def SignedBits(size: int, dtype: type[T], default = 0) -> T:
    return _SignedBits(size, dtype, default)

def NormalizedBits(size: int, dtype: type[T], default = 0) -> T:
    return _NormalizedBits(size, dtype, default)

def NormalizedSignedBits(size: int, dtype: type[T], default = 0) -> T:
    return _NormalizedSignedBits(size, dtype, default)


def _bitProperty(fmt: _Bits, bitIdx: int):
    return property(
        fget=lambda self: self._getField(fmt, bitIdx),
        fset=lambda self, v: self._setField(fmt, bitIdx, v)
    )


class BitStructMeta(type):

    def __new__(mcs, clsname, bases, attrs):
        bitIdx = 0
        attrs["_bitFmts"] = bitFmts = {}
        for attrName, attr in attrs.items():
            if isinstance(attr, _Bits):
                attrs[attrName] = _bitProperty(attr, bitIdx)
                bitFmts[attr] = bitIdx
                bitIdx += attr.size
        attrs["size"] = bitsToBytes(bitIdx)
        return super().__new__(mcs, clsname, bases, attrs)


class BitStruct(object, metaclass=BitStructMeta):
    """Used for creating binary data structures that hold values of potentially varying sizes.

    Subclass and add fields through the Bits class. Any non-Bits members are ignored on pack/unpack.

    The bit format includes 3 components: size, type, and default.
    Size is the only required one, and defines its size in bits.

    "type" is the type of data. It should be convertible to & from an int.

    "default" is the field's default value for every class instance.

    Simple example::

        class TestStruct(BitStruct):
            a = Bits(1, bool)
            b = int = 3
            c = Bits(3, int)

    In this case, a has a size of 1 bit and c has 3. b gets ignored.
    Note that from top to bottom, members are sorted from least significant bits to most.
    The bits here would be arranged like "ccca" (0101) when packed, since b would get ignored.
    """

    size: int
    """Size of this BitStruct in bytes."""

    def __init__(self, val: int = None):
        if val is None:
            self._val = 0
            for fmt, bitIdx in self._bitFmts.items():
                self._setField(fmt, bitIdx, fmt.default)
        else:
            self._val = val

    def __eq__(self, other: "BitStruct"):
        return isinstance(other, BitStruct) and self._val == other._val

    def __int__(self):
        return self._val

    def copy(self):
        """Quickly return a copy of this BitStruct with the same values."""
        return type(self)(self._val)

    @classmethod
    def unpack(cls, b: bytes):
        """Unpack a BitStruct from a sequence of bytes."""
        return cls(int.from_bytes(b, "big"))

    def pack(self):
        """Pack this BitStruct to bytes (big endian, as few as possible based on format)."""
        return self._val.to_bytes(self.size, "big")

    def _getField(self, fmt: _Bits, bitIdx: int):
        return fmt.unpack(self._val >> bitIdx & fmt.mask)

    def _setField(self, fmt: _Bits, bitIdx: int, v: int):
        self._val = (self._val & ~(fmt.mask << bitIdx)) | (fmt.pack(v) << bitIdx)
