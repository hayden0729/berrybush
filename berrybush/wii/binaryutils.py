# standard imports
from functools import cache
from numbers import Number


@cache
def maxDifBit(i1: int, i2: int):
    """Return the index (from the right) of the most significant bit that's different for two ints.

    If they are equal, return -1.
    If they have different lengths, return the length of the longest one.
    """
    idx = -1
    while i1 != i2:
        i1 >>= 1
        i2 >>= 1
        idx += 1
    return idx


@cache
def strToInt(string: str):
    """Convert an ASCII string to an int representation."""
    return int.from_bytes(string.encode("ascii"), "big")


def maxBitVal(numBits: int):
    """Return the maximum value possible for a value stored in some number of bits."""
    return (1 << numBits) - 1


@cache
def normBitVal(val: float, numBits: int):
    """Normalize some value from 0-1 based on the max allowed by the # bits in which it's stored."""
    return val / maxBitVal(numBits)


@cache
def denormBitVal(val: float, numBits: int):
    """Inverse of normBitVal; multiply val by max allowed by # bits in which it's stored.

    Note that the output is rounded to the nearest integer, so some decimal precision may be lost.
    """
    return round(val * maxBitVal(numBits))


@cache
def pad(obj, n: int, startOffset = 0, extra: bool = False):
    """Pad an object (number or sequence of bytes) until it or its length is a multiple of n.

    Optionally, you can pass in a start offset that's temporarily added for the calculation.
    You can also set whether n should be added an additional time if it's exactly a multiple of n.
    """
    objLen = len(obj) if not isinstance(obj, Number) else obj
    padLen = n - ((startOffset + objLen) % n)
    if not extra:
        padLen %= n
    padding = bytearray(padLen) if not isinstance(obj, Number) else padLen
    return obj + padding


def bitsToBytes(b: int):
    """Get the number of bytes that a number of bits represents."""
    return -(b // -8) # ceiling division


def calcOffset(a: int, b: int):
    """Calculate an offset from address A to B. If either is 0, return 0."""
    if (a == 0 or b == 0):
        return 0
    return b - a
