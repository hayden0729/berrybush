# standard imports
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
import struct
from typing import Generic, Iterator, TypeVar, TYPE_CHECKING
# internal imports
from ..wii.binaryutils import pad
# special typing imports
if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = object


T = TypeVar("T", bound="GLSLType")
B = TypeVar("B", bound="GLSLBasicType")


class GLSLType(Generic[T]):
    """GLSL type definition.

    Use in custom structs by type-hinting members with type instances that define their parameters
    (e.g., array length).
    """

    @classmethod
    @abstractmethod
    def getDefault(cls):
        """Get the default value for this type."""

    @abstractmethod
    def getName(self) -> str:
        """Get the name of this type as it would appear in GLSL code."""

    @abstractmethod
    def getSize(self) -> int:
        """Return the size of this type, in bytes."""

    @abstractmethod
    def getAlignment(self) -> int:
        """Return the alignment of this type, in bytes."""

    @abstractmethod
    def packVal(self, val) -> bytes:
        """Pack a value for this type to bytes."""


class GLSLSimpleType(GLSLType):
    """GLSL type definition that doesn't require instances for hinting, just the type itself."""
    # pylint:disable=arguments-differ

    @classmethod
    @abstractmethod
    def getName(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def getSize(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def getAlignment(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def packVal(cls, val) -> bytes:
        pass


class GLSLBasicType(GLSLSimpleType):
    """Fundamental GLSL data type (int, float, etc)"""

    @classmethod
    def getDefault(cls):
        return 0

    @classmethod
    def getSize(cls):
        return 4

    @classmethod
    def getAlignment(cls):
        return 4

    @classmethod
    @abstractmethod
    def getPrefix(cls) -> str:
        """Get the prefix for another type (vector or matrix) containing this type."""


class GLSLBool(GLSLBasicType):

    @classmethod
    def getDefault(cls):
        return False

    @classmethod
    def getName(cls):
        return "bool"

    @classmethod
    def packVal(cls, val):
        return struct.pack("I", val)

    @classmethod
    def getPrefix(cls):
        return "b"


class GLSLInt(GLSLBasicType):

    @classmethod
    def getName(cls):
        return "int"

    @classmethod
    def packVal(cls, val):
        return struct.pack("i", val)

    @classmethod
    def getPrefix(cls):
        return "i"


class GLSLUInt(GLSLBasicType):

    @classmethod
    def getName(cls):
        return "uint"

    @classmethod
    def packVal(cls, val):
        return struct.pack("I", val)

    @classmethod
    def getPrefix(cls):
        return "u"


class GLSLFloatType(GLSLBasicType): # pylint: disable=abstract-method
    pass


class GLSLFloat(GLSLFloatType):

    @classmethod
    def getName(cls):
        return "float"

    @classmethod
    def packVal(cls, val):
        return struct.pack("f", val)

    @classmethod
    def getPrefix(cls):
        return ""


class GLSLDouble(GLSLFloatType):

    @classmethod
    def getName(cls):
        return "double"

    @classmethod
    def getSize(cls):
        return 8

    @classmethod
    def getAlignment(cls):
        return 8

    @classmethod
    def packVal(cls, val):
        return struct.pack("d", val)

    @classmethod
    def getPrefix(cls):
        return "d"


@dataclass(frozen=True)
class GLSLVec(GLSLType[B]):

    dtype: type[B]
    length: int

    @classmethod
    def getDefault(cls):
        return ()

    @cache
    def getName(self):
        return f"{self.dtype.getPrefix()}vec{self.length}"

    @cache
    def getSize(self):
        return self.dtype.getSize() * self.length

    @cache
    def getAlignment(self):
        return self.dtype.getSize() * (self.length + self.length % 2)

    @cache
    def packVal(self, val):
        packed = b"".join(self.dtype.packVal(v) for v in val)
        return packed + b"\x00" * (self.getSize() - len(packed))


@dataclass(frozen=True)
class GLSLArr(GLSLType[T]):

    dtype: type[T] | T
    length: int

    def __iter__(self) -> Iterator[T]:
        pass

    @classmethod
    def getDefault(cls):
        return ()

    @cache
    def _isScalarOrVectorArr(self):
        return (
            (isinstance(self.dtype, GLSLVec)) or
            (isinstance(self.dtype, type) and issubclass(self.dtype, GLSLBasicType))
        )

    @cache
    def getName(self):
        return f"{self.dtype.getName()}[{self.length}]"

    @cache
    def getSize(self):
        if self._isScalarOrVectorArr():
            return self.getAlignment() * self.length
        return self.dtype.getSize() * self.length

    @cache
    def getAlignment(self):
        if self._isScalarOrVectorArr():
            return pad(self.dtype.getSize(), 16)
        return self.dtype.getAlignment()

    def packVal(self, val):
        align = self.getAlignment()
        padding = b"\x00" * (align - self.dtype.getSize())
        packed = b"".join(self.dtype.packVal(v) + padding for v in val)
        return packed + b"\x00" * (self.getSize() - len(packed))


class GLSLMat(GLSLType):

    def __init__(self, dtype: type[GLSLFloatType], cols: int, rows: int = None):
        self.dtype = dtype
        self.cols = cols
        self.rows = rows if rows else cols

    @classmethod
    def getDefault(cls):
        return ()

    @cache
    def getName(self):
        dims = f"{self.cols}x{self.rows}" if self.cols != self.rows else self.cols
        return f"{self.dtype.getPrefix()}mat{dims}"

    @cache
    def getAlignment(self):
        return GLSLArr(GLSLVec(self.dtype, self.rows), self.cols).getAlignment()

    @cache
    def getSize(self):
        return GLSLArr(GLSLVec(self.dtype, self.rows), self.cols).getSize()

    @cache
    def packVal(self, val):
        return GLSLArr(GLSLVec(self.dtype, self.rows), self.cols).packVal(val)


def _glslField(name: str):
    return property(
        fget=lambda self: self._getField(name),
        fset=lambda self, v: self._setField(name, v)
    )


class GLSLStructMeta(type):

    def __new__(mcs, clsname, bases, attrs):
        hasComplexFields = False
        attrs["_glslFieldTypes"] = fields = {}
        for fieldName, f in attrs.items():
            if isinstance(f, GLSLType) or (isinstance(f, type) and issubclass(f, GLSLType)):
                attrs[fieldName] = _glslField(fieldName)
                fields[fieldName] = f
                if isinstance(f, type) and issubclass(f, GLSLStruct):
                    hasComplexFields = True
                elif isinstance(f, GLSLArr):
                    if isinstance(f.dtype, type) and issubclass(f.dtype, GLSLStruct):
                        hasComplexFields = True
        attrs["_hasComplexFields"] = hasComplexFields # true for nested structs (see pack())
        return super().__new__(mcs, clsname, bases, attrs)


class GLSLStruct(GLSLSimpleType, metaclass=GLSLStructMeta):
    """Custom struct type for GLSL. Subclass and add fields through GLSL type instances.

    Load this struct into a Blender shader info object using the info's typedef_source() method with
    this class's getSource() result. Then, you can create uniform variables of this type using the
    info's uniform_buf() method and load instances of it by creating UBOs with the instances' pack()
    results as data.
    """

    _glslFieldTypes: dict[str, GLSLType]
    _hasComplexFields: bool

    def __init__(self):
        self._glslFieldVals = {n: t.getDefault() for n, t in self._glslFieldTypes.items()}
        self._packed: bytes | None = None

    def _getField(self, name: str):
        return self._glslFieldVals[name]

    def _setField(self, name: str, v):
        if self._glslFieldVals[name] != v:
            self._glslFieldVals[name] = v
            # invalidate packed cache (literally the entire point of this function & _getField())
            self._packed = None

    @classmethod
    @cache
    def getSource(cls):
        """Return the GLSL definition for this struct."""
        fields = "".join(f"{t.getName()} {name};" for name, t in cls._glslFieldTypes.items())
        return f"struct {cls.getName()} {{{fields}}};"

    @classmethod
    def getDefault(cls):
        return cls()

    @classmethod
    @cache
    def getName(cls):
        return cls.__name__

    @classmethod
    @cache
    def getSize(cls) -> int:
        size = 0
        for name, t in cls._glslFieldTypes.items():
            size = pad(size, t.getAlignment()) + t.getSize()
        return pad(size, cls.getAlignment())

    @classmethod
    @cache
    def getAlignment(cls):
        return pad(max(t.getAlignment() for name, t in cls._glslFieldTypes.items()), 16)

    @classmethod
    def packVal(cls, val: Self) -> bytes:
        if val is None:
            return b"\x00" * cls.getSize()
        if val._packed is not None:
            return val._packed
        packed = b""
        for name, t in cls._glslFieldTypes.items():
            packed = pad(packed, t.getAlignment()) + t.packVal(getattr(val, name))
        packed = pad(packed, cls.getAlignment())
        # for efficiency, packed values are usually cached & cache is invalidated when fields change
        # however, if this struct contains other structs, we can't rely on that, since these other
        # structs are themselves mutable! so if that's the case, just don't cache ever
        if not val._hasComplexFields:
            val._packed = packed
        return packed

    def pack(self):
        """Pack this struct to bytes."""
        return self.packVal(self)
