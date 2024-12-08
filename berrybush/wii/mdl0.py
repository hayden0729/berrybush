# standard imports
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, Flag, auto
from statistics import multimode
from struct import Struct
from typing import TypeVar, TYPE_CHECKING
# 3rd party imports
import numpy as np
# internal imports
from .binaryutils import pad, maxBitVal, calcOffset, normBitVal, denormBitVal
from .brresdict import DictReader, DictWriter
from .common import fillList, getKey, keyVals, keyValsDef, unique, Tree
from . import gx
from .serialization import (
    AddressedSerializable, Readable, Writable, Reader, Writer, Serializer,
    StrPoolReadMixin, StrPoolWriteMixin
)
from .subfile import BRRES_SER_T, Subfile, SubfileSerializer, SubfileReader, SubfileWriter
from . import transform as tf
# special typing imports
if TYPE_CHECKING:
    from brres import BRRESReader, BRRESWriter
    from typing_extensions import Self
else:
    Self = BRRESReader = BRRESWriter = object


_MDL0_SER_T = TypeVar("_MDL0_SER_T", bound="MDL0Serializer")
_MAT_SER_T = TypeVar("_MAT_SER_T", bound="MaterialSerializer")
_RBL_T = TypeVar("_RBL_T", bound="Readable")
_WBL_T = TypeVar("_WBL_T", bound="Writable")
_LINK_T = TypeVar("_LINK_T", bound="ResourceLinkWriter")


class BillboardMode(Enum):
    """Setting for how a joint's billboard display works."""
    OFF = 0
    STANDARD = 1
    STANDARD_PERSP = 2
    ROTATION = 3
    ROTATION_PERSP = 4
    Y_ROTATION = 5
    Y_ROTATION_PERSP = 6


class IndTexMode(Enum):
    """Determines how an indirect texture is applied."""
    WARP = 0
    NORMAL_MAP = 1
    NORMAL_MAP_SPEC = 2
    FUR = 3
    USER_0 = 6
    USER_1 = 7


class MagFilter(Enum):
    """Magnification filter, applied when a texture is zoomed in."""
    NEAREST = 0
    LINEAR = 1


class MinFilter(Enum):
    """Minification filter, applied when a texture is zoomed out."""
    NEAREST = 0
    LINEAR = 1
    NEAREST_MIPMAP_NEAREST = 2
    LINEAR_MIPMAP_NEAREST = 3
    NEAREST_MIPMAP_LINEAR = 4
    LINEAR_MIPMAP_LINEAR = 5


class MaxAnisotropy(Enum):
    """Determines the maximum degree for anisotropic filtering."""
    ONE = 0
    TWO = 1
    FOUR = 2


class RenderGroup(Enum):
    """Material render group, used in object drawing order."""
    OPA = 0 # opaque, rendered first
    XLU = 128 # translucent, rendered last


class TexMapMode(Enum):
    """Method for obtaining a texture's coordinates."""
    UV = 0
    ENV_CAM = 1
    PROJECTION = 2
    ENV_LIGHT = 3
    ENV_SPEC = 4


class WrapMode(Enum):
    """Determines how texture wrapping is handled."""
    CLAMP = 0
    REPEAT = 1
    MIRROR = 2


class DefinitionCommand():
    """Command identified by a bytecode that makes up a definition."""

    BYTECODE: bytes

    @classmethod
    def unpack(cls, data: bytes, offset: int):
        """Unpack command from bytes (starting after the bytecode)."""
        return cls()

    def size(self):
        return 1 # bytecode is 1 byte

    def pack(self):
        """Pack command to bytes (including the bytecode)."""
        return int.to_bytes(self.BYTECODE, 1, "big")


class DefTerminate(DefinitionCommand):
    """Signals the end of definition command data."""

    BYTECODE = 1


class DefJointParent(DefinitionCommand):
    """Pairs a joint with its parent."""

    BYTECODE = 2

    _STRCT = Struct(">HH")

    def __init__(self, jointIdx = 0, parentIdx = 0):
        self.jointIdx = jointIdx
        self.parentIdx = parentIdx

    @classmethod
    def unpack(cls, data: bytes, offset: int):
        cmd = super().unpack(data, offset)
        unpackedData = cls._STRCT.unpack_from(data, offset)
        cmd.jointIdx, cmd.parentIdx = unpackedData
        return cmd

    def size(self):
        return super().size() + self._STRCT.size

    def pack(self):
        return super().pack() + self._STRCT.pack(self.jointIdx, self.parentIdx)


class DefDeformer(DefinitionCommand):
    """Defines a deformer's joint weights."""

    BYTECODE = 3

    _HEAD_STRCT = Struct(">HB")
    _WEIGHT_STRCT = Struct(">Hf")

    def __init__(self, dIdx = 0, weights: dict[int, float] = None):
        self.dIdx = dIdx
        self.weights = weights if weights is not None else {} # pairs joint indices w/ their weights

    @classmethod
    def unpack(cls, data: bytes, offset: int):
        cmd = super().unpack(data, offset)
        unpackedHeader = cls._HEAD_STRCT.unpack_from(data, offset)
        cmd.dIdx, numWeights = unpackedHeader
        for weightIdx in range(numWeights):
            weightOffset = offset + cls._HEAD_STRCT.size + weightIdx * cls._WEIGHT_STRCT.size
            unpackedWeight = cls._WEIGHT_STRCT.unpack_from(data, weightOffset)
            cmd.weights[unpackedWeight[0]] = unpackedWeight[1]
        return cmd

    def size(self):
        headSize = self._HEAD_STRCT.size
        weightSize = self._WEIGHT_STRCT.size * len(self.weights)
        return super().size() + headSize + weightSize

    def pack(self):
        packedHeader = self._HEAD_STRCT.pack(self.dIdx, len(self.weights))
        packedWeights = b"".join(self._WEIGHT_STRCT.pack(i, w) for i, w in self.weights.items())
        return super().pack() + packedHeader + packedWeights


class DefMeshRendering(DefinitionCommand):
    """Defines a mesh's rendering through its material, render priority, and visibility joint."""

    BYTECODE = 4

    _STRCT = Struct(">HHHB")

    def __init__(self, matIdx = 0, meshIdx = 0, visJointIdx = 0, drawPrio = 0):
        self.matIdx = matIdx
        self.meshIdx = meshIdx
        self.visJointIdx = visJointIdx
        self.drawPrio = drawPrio

    @classmethod
    def unpack(cls, data: bytes, offset: int):
        cmd = super().unpack(data, offset)
        unpackedData = cls._STRCT.unpack_from(data, offset)
        cmd.matIdx, cmd.meshIdx, cmd.visJointIdx, cmd.drawPrio = unpackedData
        return cmd

    def size(self):
        return super().size() + self._STRCT.size

    def pack(self):
        return super().pack() + self._STRCT.pack(
            self.matIdx, self.meshIdx, self.visJointIdx, self.drawPrio
        )


class DefDeformerMember(DefinitionCommand):
    """Associates a joint with a deformer in which it's used."""

    BYTECODE = 5

    _STRCT = Struct(">HH")

    def __init__(self, dIdx = 0, jointIdx = 0):
        self.dIdx = dIdx
        self.jointIdx = jointIdx

    @classmethod
    def unpack(cls, data: bytes, offset: int):
        cmd = super().unpack(data, offset)
        unpackedData = cls._STRCT.unpack_from(data, offset)
        cmd.dIdx, cmd.jointIdx = unpackedData
        return cmd

    def size(self):
        return super().size() + self._STRCT.size

    def pack(self):
        return super().pack() + self._STRCT.pack(self.dIdx, self.jointIdx)


class Definition(AddressedSerializable[_MDL0_SER_T], Readable[_MDL0_SER_T], Writable[_MDL0_SER_T]):
    """Structure for setting a few MDL0 properties."""

    _COMMAND_TYPES: dict[bytes, type[DefinitionCommand]] = {cmd.BYTECODE: cmd for cmd in (
        DefTerminate, DefJointParent, DefDeformer, DefMeshRendering, DefDeformerMember
    )}

    def __init__(self, parent: _MDL0_SER_T = None, offset = 0):
        super().__init__(parent, offset)
        self.cmds: list[DefinitionCommand] = []

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def unpack(self, data: bytes):
        cmdOffset = self.offset
        while True:
            cmdType = self._COMMAND_TYPES[data[cmdOffset]]
            if cmdType is DefTerminate:
                break
            cmd = cmdType.unpack(data, cmdOffset + 1)
            self.cmds.append(cmd)
            cmdOffset += cmd.size()
        return self

    def size(self):
        return sum(cmd.size() for cmd in self.cmds) + 1 # + 1 for terminate cmd added on pack

    def pack(self):
        return b"".join(cmd.pack() for cmd in self.cmds) + DefTerminate().pack()


class Joint(Tree):
    """Defines a transformation (and a few other properties) used for displaying MDL0 meshes."""

    def __init__(self, parent: Self = None, name: str = None):
        super().__init__(parent)
        self.name = name
        self.isVisible = True
        self.bbMode = BillboardMode.OFF
        self.bbParent: Joint = None
        self.segScaleComp = False
        self._srt = tf.Transformation(3)
        self._mtxCache: dict[tf.AbsMtxGenerator, np.ndarray] = {}
        self._invMtxCache: dict[tf.AbsMtxGenerator, np.ndarray] = {}
        self._d = Deformer({self: 1.0})

    @property
    def deformer(self):
        """Deformer made up of only this joint."""
        return self._d

    @property
    def scale(self):
        """Scale of this joint."""
        return self._srt.s

    @property
    def rot(self):
        """Rotation of this joint."""
        return self._srt.r

    @property
    def trans(self):
        """Translation of this joint."""
        return self._srt.t

    def setSRT(self, s: np.ndarray = None, r: np.ndarray = None, t: np.ndarray = None):
        """Update any of this joint's transformation vectors with new values."""
        self._srt.set(s, r, t)
        for joint in self.deepChildren():
            joint._mtxCache = {}
            joint._invMtxCache = {}

    def setMtxCache(
        self,
        mtxGen: type[tf.AbsMtxGenerator],
        mtx: np.ndarray | None = None,
        inv: np.ndarray | None = None,
    ):
        """Overwrite any of this joint's cached matrices with new values.
        
        This is unnecessary in most cases, as these caches are automatically recalculated when
        necessary. It's used when importing in case the imported matrices don't match up with the
        transforms (which can happen in models created with other tools, e.g., maybe BrawlCrate?)
        """
        if mtx is not None:
            self._mtxCache[mtxGen] = mtx
        if inv is not None:
            self._invMtxCache[mtxGen] = inv

    def mtx(self, model: "MDL0"):
        """Absolute matrix for this joint, calculated based on a model's settings."""
        mtxGen = model.mtxGen3D
        try:
            return self._mtxCache[mtxGen].copy()
        except KeyError:
            mtx = mtxGen.absMtx([(j._srt, j.segScaleComp) for j in self.ancestors()])
            self._mtxCache[mtxGen] = mtx
            return mtx

    def invMtx(self, model: "MDL0"):
        """Inverse absolute matrix for this joint, calculated based on a model's settings."""
        mtxGen = model.mtxGen3D
        try:
            return self._invMtxCache[mtxGen].copy()
        except KeyError:
            mtx = np.linalg.inv(self.mtx(model))
            self._invMtxCache[mtxGen] = mtx
            return mtx

    def addChild(self, child: Self):
        super().addChild(child)
        for joint in child.deepChildren():
            joint._mtxCache = {}
            joint._invMtxCache = {}

    def removeChild(self, child: Self):
        super().removeChild(child)
        for joint in child.deepChildren():
            joint._mtxCache = {}
            joint._invMtxCache = {}


class JointFlags(Flag):
    """Flags that store info about various joint characteristics."""
    NO_TRANSFORMATION = auto()
    NO_TRANSLATION = auto()
    NO_ROTATION = auto()
    NO_SCALE = auto()
    HAS_UNIFORM_SCALE = auto()
    SEG_SCALE_COMP = auto()
    CHILD_HAS_SSC = auto()
    HIERARCHICAL_SCALE = auto() # https://download.autodesk.com/global/docs/softimage2014/en_us/userguide/index.html?url=files/transforms_ScalingObjects.htm,topicNumber=d30e51306 pylint: disable=line-too-long
    IS_VISIBLE = auto()
    HAS_GEOMETRY = auto()
    HAS_BB_PARENT = auto()


class JointSerializer(Serializer[_MDL0_SER_T, Joint]):

    DATA_TYPE = Joint
    _STRCT = Struct(">IiiIIIII 3f 3f 3f 3f 3f iiiii 12f 12f")


class JointReader(JointSerializer["MDL0Reader"], Reader, StrPoolReadMixin):

    def __init__(self, parent: "MDL0Reader" = None, offset = 0):
        super().__init__(parent, offset)
        self._bbParentIdx: int = None
        self._parentOffset: int = None
        self._mtx: np.ndarray = None
        self._invMtx: np.ndarray = None

    def unpack(self, data: bytes):
        super().unpack(data)
        joint = self._data = Joint()
        unpackedData = self._STRCT.unpack_from(data, self.offset)
        flags = JointFlags(unpackedData[5])
        joint.isVisible = JointFlags.IS_VISIBLE in flags
        joint.segScaleComp = JointFlags.SEG_SCALE_COMP in flags
        joint.bbMode = BillboardMode(unpackedData[6])
        self._bbParentIdx = unpackedData[7] if JointFlags.HAS_BB_PARENT in flags else None
        joint.setSRT(unpackedData[8:11], unpackedData[11:14], unpackedData[14:17])
        self._mtx = np.reshape([*unpackedData[28:40], 0, 0, 0, 1], (4, 4))
        self._invMtx = np.reshape([*unpackedData[40:52], 0, 0, 0, 1], (4, 4))
        self._parentOffset = self.offset + unpackedData[23] if unpackedData[23] != 0 else None
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.name = self.parentSer.itemName(self)
        jrs = self.parentSer.section(JointReader).values() # all the model's joint readers
        if self._bbParentIdx is not None:
            self._data.bbParent = tuple(jrs)[self._bbParentIdx].getInstance()
        if self._parentOffset is not None:
            self._data.parent = next(j.getInstance() for j in jrs if j.offset == self._parentOffset)

    def updateMatrixCache(self, mtxGen: type[tf.AbsMtxGenerator]):
        self._data.setMtxCache(mtxGen, mtx=self._mtx, inv=self._invMtx)


class JointWriter(JointSerializer["MDL0Writer"], Writer, StrPoolWriteMixin):

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def _calcSize(self):
        return self._STRCT.size

    def _getFlags(self):
        joint = self._data
        flags = JointFlags(0)
        if joint.bbParent is not None:
            flags |= JointFlags.HAS_BB_PARENT
        deformIdx = self.parentSer.deformers.getInstance().index(joint.deformer)
        if deformIdx < self.parentSer.numGeometryDeformers:
            flags |= JointFlags.HAS_GEOMETRY
        if joint._srt.homoS:
            flags |= JointFlags.HAS_UNIFORM_SCALE
        if joint.segScaleComp:
            flags |= JointFlags.SEG_SCALE_COMP
        if any(j.segScaleComp for j in joint.children):
            flags |= JointFlags.CHILD_HAS_SSC
        if joint.isVisible:
            flags |= JointFlags.IS_VISIBLE
        noS = joint._srt.identityS
        noR = joint._srt.identityR
        noT = joint._srt.identityT
        if noS:
            flags |= JointFlags.NO_SCALE
        if noR:
            flags |= JointFlags.NO_ROTATION
        if noT:
            flags |= JointFlags.NO_TRANSLATION
        if noS and noR and noT:
            flags |= JointFlags.NO_TRANSFORMATION
        return flags

    def pack(self):
        # family references
        parent = self._data.parent
        firstChild = self._data.children[0] if self._data.children else None
        fam = (parent, firstChild)
        if parent is not None:
            selfIdx = parent.children.index(self._data) # index of this joint in parent's children
            nextSib = parent.children[selfIdx + 1] if selfIdx < len(parent.children) - 1 else None
            prevSib = parent.children[selfIdx - 1] if selfIdx > 0 else None
            fam += (nextSib, prevSib)
        else:
            fam += (None, None)
        writers = self.parentSer.section(JointWriter)
        famOffsets = (writers[j.name].offset - self.offset if j is not None else 0 for j in fam)
        # billboard parent
        bbParent = self._data.bbParent
        bbParentIdx = self.parentSer.itemIdx(writers[bbParent.name]) if bbParent is not None else 0
        # pack first 3 rows of absolute transform matrix & inverse (last row assumed to be 0 0 0 1)
        # note: these matrices don't seem to have any effect
        # (looks like the wii only cares about the srt vectors)
        mtx = self._data.mtx(self.parentSer.getInstance())
        inv = self._data.invMtx(self.parentSer.getInstance())
        mtx = mtx.flatten()[:12]
        inv = inv.flatten()[:12]
        minCoord = maxCoord = [0, 0, 0] # TODO: min/max coords
        return self._STRCT.pack(self._size, self.parentSer.offset - self.offset,
                                self.stringOffset(self._data.name) - self.offset,
                                self.parentSer.itemIdx(self),
                                self.parentSer.deformers.getInstance().index(self._data.deformer),
                                self._getFlags().value, self._data.bbMode.value, bbParentIdx,
                                *self._data.scale, *self._data.rot, *self._data.trans,
                                *minCoord, *maxCoord, *famOffsets, 0, *mtx, *inv)


class Deformer(Mapping[Joint, float]):
    """Mapping of weighted joints w/ a combined transformation matrix."""

    def __init__(self, weights: dict[Joint, float] = None):
        self._weights = weights if weights is not None else {}

    def mtx(self, model: "MDL0") -> np.ndarray:
        """Matrix for this deformer, calculated based on a model's settings."""
        return np.sum(j.mtx(model) * w for j, w in self._weights.items())

    def pose(self, model: "MDL0", pose: dict[Joint, np.ndarray]):
        """Matrix for this deformer based on a pose.

        This matrix converts geometry (presumably already transformed by mtx()) to a new pose,
        defined by the "pose" argument. This argument should be a dict that points each
        joint to its relative posed matrix.
        """
        return np.sum(pose[j] @ j.invMtx(model) * w for j, w in self._weights.items())

    @property
    def joints(self):
        return tuple(self._weights.keys())

    @property
    def weights(self):
        return tuple(self._weights.values())

    def __len__(self):
        return len(self._weights)

    def __getitem__(self, key):
        return self._weights[key]

    def __iter__(self):
        return iter(self._weights)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._weights)})"

    def __contains__(self, key):
        return key in self._weights

    def __hash__(self):
        return hash(tuple(self._weights.items()))

    def __eq__(self, other):
        return isinstance(other, Deformer) and self._weights == other._weights


class DeformerArrayReader(Reader["MDL0Reader", list[Deformer]]):

    def __init__(self, parent: "MDL0Reader" = None, offset = 0):
        super().__init__(parent, offset)
        self._arr: list[dict[int, float]] = []

    def unpack(self, data: bytes):
        super().unpack(data)
        offset = self.offset
        size = int.from_bytes(data[offset : offset + 4], "big")
        arr = Struct(">" + "i" * size).unpack_from(data, offset + 4)
        # get joints from indices; -1 indices will be set on definition apply, make none for now
        self._arr = [{i: 1.0} if i != -1 else None for i in arr]
        return self

    def applyDef(self, cmd: DefDeformer):
        """Update one entry in this array based on a deformer definition command."""
        self._arr[cmd.dIdx] = {next(iter(self._arr[i])): w for i, w in cmd.weights.items()}

    def _updateInstance(self):
        super()._updateInstance()
        jnts = tuple(self.parentSer.section(JointReader).values())
        self._data = [Deformer({jnts[j].getInstance(): w for j, w in d.items()}) for d in self._arr]


class DeformerArrayWriter(Writer["MDL0Writer", list[Deformer]]):

    def __init__(self, parent: "MDL0Writer" = None, offset = 0):
        super().__init__(parent, offset)
        self.indices: dict[Deformer, int] = {}

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    @property
    def _struct(self):
        return Struct(">I" + "i" * len(self._data))

    def _calcSize(self):
        return self._struct.size

    def fromInstance(self, data: list[Deformer]):
        super().fromInstance(data)
        self.indices = {df: i for i, df in enumerate(data)}
        return self

    def pack(self):
        joints = self.parentSer.getInstance().rootJoint.deepChildren()
        jointIdcs = {j: i for i, j in enumerate(joints)}
        deformIdcs = (jointIdcs[d.joints[0]] if len(d) == 1 else -1 for d in self._data)
        return self._struct.pack(len(self._data), *deformIdcs)


class VertexAttrGroup():
    """Group of items for some vertex attribute, like positions or normals."""

    ATTR_TYPE = gx.VertexAttr
    MAX_LEN = maxBitVal(16)

    def __init__(self, name: str = None, arr: np.ndarray = None, attr: ATTR_TYPE = None):
        self.name = name
        if arr is None:
            self._arr = np.ndarray((0, self.ATTR_TYPE.PADDED_COUNT))
        else:
            self.setArr(arr)
        self.attr = attr
        """GX attribute descriptor to use when packing. Set to None for automatic generation."""

    def __len__(self):
        return len(self._arr)

    @property
    def arr(self):
        """Vertex data for this group."""
        return self._arr

    def setArr(self, arr: np.ndarray):
        """Set this group's data with another array, which is copied.

        If its entries are too long (e.g., 4 elements when only 3 are supported), they're cropped.
        If entries are too short, they're padded to the proper length.
        """
        self._arr = self.ATTR_TYPE.pad(arr[..., :self.ATTR_TYPE.PADDED_COUNT])

    def genAttr(self):
        """Generate a GX vertex attribute descriptor for this group based on its data."""
        return self.ATTR_TYPE()

    def getAttr(self):
        """Return this group's attribute descriptor, if it has one.

        Otherwise, generate and return a new one (not assigned to the group automatically)."""
        return self.attr if self.attr else self.genAttr()


class StdVertexAttrGroup(VertexAttrGroup):
    """Standard vertex attribute group template followed by most attribute types."""
    ATTR_TYPE = gx.StdVertexAttr

    def genAttr(self):
        attr = super().genAttr()
        arr = self._arr
        arrMax = np.max(arr)
        arrMin = np.min(arr)
        isUnsigned = arrMin >= 0
        dtypes = (
            self.ATTR_TYPE.DataType.UINT8 if isUnsigned else self.ATTR_TYPE.DataType.INT8,
            self.ATTR_TYPE.DataType.UINT16 if isUnsigned else self.ATTR_TYPE.DataType.INT16
        )
        for dtype in dtypes:
            fmt = dtype.fmt
            scale = 16
            if arrMax > 0:
                maxAllowed = np.iinfo(fmt).max
                allowedScale = int(maxAllowed / arrMax).bit_length() - 1
                if allowedScale == -1:
                    continue
                scale = min(scale, allowedScale)
            if arrMin < 0:
                minAllowed = np.iinfo(fmt).min
                allowedScale = int(minAllowed / arrMin).bit_length() - 1
                if allowedScale == -1:
                    continue
                scale = min(scale, allowedScale)
            convertedArr = (arr * (1 << scale)).round().astype(fmt)
            if np.allclose(arr, convertedArr / (1 << scale)):
                attr.dtype = dtype
                attr.scale = scale
                break
        return attr


class VertexAttrGroupSerializer(Serializer[_MDL0_SER_T, VertexAttrGroup]):

    _STRCT = Struct(">IiiiIIIBBH")


class VertexAttrGroupReader(VertexAttrGroupSerializer["MDL0Reader"], Reader, StrPoolReadMixin):

    def __init__(self, parent: "MDL0Reader" = None, offset = 0):
        super().__init__(parent, offset)
        self._attr: gx.VertexAttr = None

    def unpack(self, data: bytes):
        super().unpack(data)
        unpackedData = self._STRCT.unpack_from(data, self.offset)
        dataOffset = unpackedData[2] + self.offset
        self._data = self.DATA_TYPE()
        self._attr = self._unpackAttr(unpackedData[5:9])
        arr = self._attr.unpackBuffer(data[dataOffset:], unpackedData[9])
        self._data.setArr(arr)
        self._data.attr = self._attr
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.name = self.parentSer.itemName(self)

    @classmethod
    @abstractmethod
    def _unpackAttr(cls, params: tuple) -> gx.VertexAttr:
        """Generate this group's attr from the parameters stored in its packed form."""


class VertexAttrGroupWriter(VertexAttrGroupSerializer["MDL0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "MDL0Writer" = None, offset = 0):
        super().__init__(parent, offset)
        self._attr: gx.VertexAttr = None

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    @abstractmethod
    def _packAttr(self) -> tuple:
        """Generate the set of parameters for this group's attr as they're stored when packed."""

    def _headSize(self):
        """Return the size (in bytes) of this group's header (up to its data)."""
        return self._STRCT.size

    def _calcSize(self):
        return super()._calcSize()

    def fromInstance(self, data: VertexAttrGroup):
        super().fromInstance(data)
        self._attr = self._data.getAttr()
        self._size = self._headSize() + pad(self._attr.calcBufferSize(len(self._data)), 32)
        return self

    def _packHeader(self):
        """Pack the header of this group (up to its data)."""
        return self._STRCT.pack(self._size, self.parentSer.offset - self.offset,
                                self._headSize(), self.stringOffset(self._data.name) - self.offset,
                                self.parentSer.itemIdx(self), *self._packAttr(), len(self._data))

    def pack(self):
        return self._packHeader() + pad(self._attr.packBuffer(self._data.arr), 32)


class StdVertexAttrGroupReader(VertexAttrGroupReader):

    @classmethod
    def _unpackAttr(cls, params: tuple):
        return cls.DATA_TYPE.ATTR_TYPE(params[1], params[0], params[2])


class StdVertexAttrGroupWriter(VertexAttrGroupWriter):

    def _packAttr(self):
        attr: gx.StdVertexAttr = self._attr
        return (attr.ctype.value, attr.dtype.value, attr.scale, attr.stride)


class MinMaxVertexAttrGroupWriter(VertexAttrGroupWriter):
    """Writer for a vertex group that stores min/max values on pack.

    This is used for vertex groups that have data which isn't normalized by nature. For instance,
    color values are always between 0-1, so this isn't necessary for them, but positions can have
    any value, so it gets used there.
    """
    _MM_STRCT_SIZE = 32 # size of min/max struct, regardless of dims

    @classmethod
    @property
    def _MM_STRCT(cls):
        """Struct that holds min/max values for this group.

        (Varies between group types based on the maximum number of dimensions allowed).
        """
        maxDims = cls.DATA_TYPE.ATTR_TYPE.CompType.maxDims()
        padAmount = cls._MM_STRCT_SIZE - 4 * 2 * maxDims
        return Struct(f">{maxDims}f {maxDims}f {padAmount}x")

    def _headSize(self):
        return super()._headSize() + self._MM_STRCT.size

    def _packHeader(self):
        arr = self._data.arr
        return super()._packHeader() + self._MM_STRCT.pack(*arr.min(0), *arr.max(0))


class PsnGroup(StdVertexAttrGroup):
    """Group of vertex positions."""
    ATTR_TYPE = gx.PsnAttr

    def genAttr(self):
        attr = super().genAttr()
        arr = self._arr
        padVal = attr.PAD_VAL
        if np.allclose(arr[:, 2], padVal): # (else implicitly xyz)
            attr.ctype = gx.PsnAttr.CompType.XY
        return attr


class PsnGroupReader(StdVertexAttrGroupReader):
    DATA_TYPE = PsnGroup


class PsnGroupWriter(StdVertexAttrGroupWriter, MinMaxVertexAttrGroupWriter):
    DATA_TYPE = PsnGroup


class NrmGroup(StdVertexAttrGroup):
    """Group of vertex normals."""
    ATTR_TYPE = gx.NrmAttr

    def __init__(self, name: str = None, arr: np.ndarray = None):
        super().__init__(name, arr)
        self.isNBT = False

    def genAttr(self):
        attr = super().genAttr()
        arr = self._arr
        if self.isNBT: # (else implicitly n)
            if len(arr) == arr.size / 9:
                attr.ctype = gx.NrmAttr.CompType.NBT
            else:
                attr.ctype = gx.NrmAttr.CompType.NBT_SPLIT
        return attr


class NrmGroupReader(StdVertexAttrGroupReader):
    DATA_TYPE = NrmGroup

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data.isNBT = self._attr.dtype is not gx.NrmAttr.CompType.N
        return self


class NrmGroupWriter(StdVertexAttrGroupWriter):
    DATA_TYPE = NrmGroup


class ClrGroup(VertexAttrGroup):
    """Group of vertex colors."""
    ATTR_TYPE = gx.ClrAttr

    def genAttr(self):
        attr = super().genAttr()
        arr = self._arr
        padVal = attr.PAD_VAL
        dtypes = gx.ClrAttr.DataType
        if np.allclose(arr[:, 3], padVal): # rgb
            if np.allclose(arr[:, :3], dtypes.RGB565.colorFmt.quantize(arr[:, :3])):
                attr.dtype = dtypes.RGB565
            else:
                attr.dtype = dtypes.RGB8
        else: # rgba
            for dtype in (dtypes.RGBA4, dtypes.RGBA6):
                if np.allclose(arr, dtype.colorFmt.quantize(arr)):
                    attr.dtype = dtype
                    break
        return attr


class ClrGroupReader(VertexAttrGroupReader):
    DATA_TYPE = ClrGroup

    @classmethod
    def _unpackAttr(cls, params: tuple):
        return cls.DATA_TYPE.ATTR_TYPE(params[1])


class ClrGroupWriter(VertexAttrGroupWriter):
    DATA_TYPE = ClrGroup

    def _packAttr(self):
        attr: gx.ClrAttr = self._attr
        return (attr.ctype.value, attr.dtype.value, attr.stride, 0)


class UVGroup(StdVertexAttrGroup):
    """Group of vertex UVs."""
    ATTR_TYPE = gx.UVAttr

    def genAttr(self):
        attr = super().genAttr()
        arr = self._arr
        padVal = attr.PAD_VAL
        if np.allclose(arr[:, 1], padVal): # (else implicitly uv)
            attr.ctype = gx.UVAttr.CompType.U
        return attr


class UVGroupReader(StdVertexAttrGroupReader):
    DATA_TYPE = UVGroup


class UVGroupWriter(StdVertexAttrGroupWriter, MinMaxVertexAttrGroupWriter):
    DATA_TYPE = UVGroup


class TexFlags(Flag):
    ENABLED = auto()
    NO_SCALE = auto()
    NO_ROTATION = auto()
    NO_TRANSLATION = auto()


class Texture():
    """Reference to an image and some settings that control its rendering for a material."""

    def __init__(self):
        self.imgName: str = None
        self.pltName: str = None
        self.wrapModes = [WrapMode.CLAMP, WrapMode.CLAMP] # wrap modes for u and v, respectively
        self.minFilter = MinFilter.LINEAR
        self.magFilter = MagFilter.LINEAR
        self.lodBias = 0
        self.maxAnisotropy = MaxAnisotropy.ONE
        self.clampBias = False
        self.texelInterpolate = False
        self._srt = tf.Transformation(2)
        self._mtxCache: dict[tf.MtxGenerator, np.ndarray] = {}
        self.mapMode = TexMapMode.UV
        self.coordIdx = 0 # idx in attribute for coords (which uv to use)
        self.usedCam = -1
        self.usedLight = -1

    def __eq__(self, other: "Texture"):
        return (
            isinstance(other, Texture)
            and self.imgName == other.imgName
            and self.pltName == other.pltName
            and self.wrapModes == other.wrapModes
            and self.minFilter == other.minFilter
            and self.magFilter == other.magFilter
            and self.lodBias == other.lodBias
            and self.maxAnisotropy == other.maxAnisotropy
            and self.clampBias == other.clampBias
            and self.texelInterpolate == other.texelInterpolate
            and self._srt == other._srt
            and self.mapMode == other.mapMode
            and self.coordIdx == other.coordIdx
            and self.usedCam == other.usedCam
            and self.usedLight == other.usedLight
        )

    @property
    def scale(self):
        """Scale of this texture."""
        return self._srt.s

    @property
    def rot(self):
        """Rotation of this texture."""
        return self._srt.r

    @property
    def trans(self):
        """Translation of this texture."""
        return self._srt.t

    @property
    def identityScale(self):
        """Whether this texture's scale is the default value."""
        return self._srt.identityS

    @property
    def identityRot(self):
        """Whether this texture's rotation is the default value."""
        return self._srt.identityR

    @property
    def identityTrans(self):
        """Whether this texture's translation is the default value."""
        return self._srt.identityT

    def setSRT(self, s: np.ndarray = None, r: np.ndarray = None, t: np.ndarray = None):
        """Update any of this texture's transformation vectors with new values."""
        self._srt.set(s, r, t)
        self._mtxCache = {}

    def mtx(self, mat: "Material"):
        """Transformation matrix for this texture, calculated based on a material's settings."""
        mtxGen = mat.mtxGen
        try:
            return self._mtxCache[mtxGen].copy()
        except KeyError:
            mtx = mtxGen.genMtx(self._srt)
            self._mtxCache[mtxGen] = mtx
            return mtx


class TextureSerializer(Serializer[_MAT_SER_T, Texture]):

    DATA_TYPE = Texture
    _STRCT = Struct(">iiiiIIIIIIfI?? 2x")


class TextureReader(TextureSerializer["MaterialReader"], Reader, StrPoolReadMixin):

    @classmethod
    def size(cls):
        """Size of a texture in bytes."""
        return cls._STRCT.size

    def unpack(self, data: bytes):
        unpackedData = self._STRCT.unpack_from(data, self.offset)
        self._data = tex = Texture()
        if unpackedData[0] != 0:
            tex.imgName = self.readString(data, self.offset + unpackedData[0])
        if unpackedData[1] != 0:
            tex.pltName = self.readString(data, self.offset + unpackedData[1])
        tex.wrapModes = [WrapMode(wm) for wm in unpackedData[6:8]]
        tex.minFilter = MinFilter(unpackedData[8])
        tex.magFilter = MagFilter(unpackedData[9])
        tex.lodBias = unpackedData[10]
        tex.maxAnisotropy = MaxAnisotropy(unpackedData[11])
        tex.clampBias = unpackedData[12]
        tex.texelInterpolate = unpackedData[13]
        return self


class TextureWriter(TextureSerializer["MaterialWriter"], Writer, StrPoolWriteMixin):

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def getStrings(self):
        strs = {self._data.imgName, self._data.pltName}
        strs.discard(None)
        return strs

    def _calcSize(self):
        return self._STRCT.size

    def getFlags(self):
        """TexFlags for this writer's texture."""
        flags = TexFlags.ENABLED
        if self._data.identityScale:
            flags |= TexFlags.NO_SCALE
        if self._data.identityRot:
            flags |= TexFlags.NO_ROTATION
        if self._data.identityTrans:
            flags |= TexFlags.NO_TRANSLATION
        return flags

    def pack(self):
        tex = self._data
        texIdx = next(i for i, t in enumerate(self.parentSer.getInstance().textures) if t is tex)
        return self._STRCT.pack(calcOffset(self.offset, self.stringOffset(self._data.imgName)),
                                calcOffset(self.offset, self.stringOffset(self._data.pltName)),
                                0, 0, texIdx, texIdx, *(m.value for m in tex.wrapModes),
                                tex.minFilter.value, tex.magFilter.value, tex.lodBias,
                                tex.maxAnisotropy.value, tex.clampBias, tex.texelInterpolate)


class IndTexture():
    """Settings for an indirect texture. 4 are stored with a material.

    More settings for these textures are set in TEV configs and in their individual stages.
    """

    def __init__(self, mode = IndTexMode.WARP, lightIdx = -1,
                 coordScales: list[gx.IndCoordScalar] = None):
        self.mode = mode
        self.lightIdx = lightIdx
        self.coordScales = coordScales if coordScales else [gx.IndCoordScalar.DIV_1] * 2

    def __eq__(self, other: "IndTexture"):
        return (
            isinstance(other, IndTexture)
            and self.mode == other.mode
            and self.lightIdx == other.lightIdx
            and self.coordScales == other.coordScales
        )


class IndTransform():
    """Matrix for an indirect texture that skews its warping effects."""

    def __init__(self):
        self._srt = tf.Transformation(2)
        self._mtxCache: np.ndarray = None

    @property
    def scale(self):
        """Scale of this indirect texture transform."""
        return self._srt.s

    @property
    def rot(self):
        """Rotation of this indirect texture transform."""
        return self._srt.r

    @property
    def trans(self):
        """Translation of this indirect texture transform."""
        return self._srt.t

    def setSRT(self, s: np.ndarray = None, r: np.ndarray = None, t: np.ndarray = None):
        """Update any of this transformation's vectors with new values."""
        self._srt.set(s, r, t)
        self._mtxCache = None

    @property
    def mtx(self) -> np.ndarray:
        """Matrix for this transformation."""
        if self._mtxCache is None:
            self._mtxCache = tf.IndMtxGen2D.genMtx(self._srt)
        return self._mtxCache

    @classmethod
    def fromMtxSettings(cls, settings: list[gx.IndMtxSettings]):
        """Generate an indirect transform from a list of GX indirect matrix settings registers."""
        # calculate matrix scale (see comments in gx.IndMtxSettings for more info)
        scale = 0
        for setting in settings:
            scale |= (setting.bits.scale << ((setting.idx % 3) * 2))
        scale += gx.IndMtxSettings.SCALE_MIN
        # read matrix values
        mtx = np.identity(3)
        mtx[:, :2] = [[item * (2 ** scale) for item in setting.bits.items] for setting in settings]
        mtx = mtx.T
        # generate srt from matrix, and make new indirect transform based on it
        newTransform: cls = cls()
        newTransform._srt = tf.decompose(mtx)
        return newTransform

    def toMtxSettings(self, mtxIdx: int):
        """Generate a list of GX indirect matrix settings registers from this transform.

        A matrix index is required that indicates the index of the matrix in XF memory (0-2).
        """
        maxVal = gx.IndMtxSettings.MAX_VAL
        minScale = gx.IndMtxSettings.SCALE_MIN
        regs: list[gx.IndMtxSettings] = []
        mtx = self.mtx[:2].copy()
        scale = maxBitVal(6) + minScale
        # highest matrix value allowed on pack is .999... so set scale such that all are below that
        absMtx = np.abs(mtx)
        while np.max(absMtx / (2 ** (scale - 1))) <= maxVal and scale > minScale:
            scale -= 1
        # iterate through cols to construct matrix cmds
        for colIdx, col in enumerate(mtx.T):
            reg = gx.IndMtxSettings(mtxIdx * 3 + colIdx)
            reg.bits.items = col / (2 ** scale)
            reg.bits.scale = ((scale - gx.IndMtxSettings.SCALE_MIN) >> (colIdx * 2)) & 0b11
            regs.append(reg)
        return regs

    def __eq__(self, other: "IndTransform"):
        return isinstance(other, IndTransform) and self._srt == other._srt


class LightChannel():
    """Controls material lighting, applied to vertex colors before they're used in TEV configs."""

    class ColorEnableFlags(Flag):
        """Flags in light channels that enable different color sources/outputs."""
        DIFFUSE_ALPHA = auto()
        DIFFUSE_COLOR = auto()
        AMBIENT_ALPHA = auto()
        AMBIENT_COLOR = auto()
        RASTER_ALPHA = auto()
        RASTER_COLOR = auto()

    class ColorControlFlags(Flag):
        """Light channel flags indicating the sources of the material and ambient colors.

        Also has some flags regarding SCN0 light enabling, but these are always disabled (and get
        enabled on runtime).
        """
        DIFFUSE_FROM_VERTEX = auto() # if false, from register
        DIFFUSE_ENABLE = auto()
        LIGHT_0_ENABLE = auto()
        LIGHT_1_ENABLE = auto()
        LIGHT_2_ENABLE = auto()
        LIGHT_3_ENABLE = auto()
        AMBIENT_FROM_VERTEX = auto() # if false, from register
        DIFFUSE_SIGNED = auto()
        DIFFUSE_CLAMPED = auto()
        ATTENUATION_ENABLE = auto()
        ATTENUATION_SPOTLIGHT = auto()
        LIGHT_4_ENABLE = auto()
        LIGHT_5_ENABLE = auto()
        LIGHT_6_ENABLE = auto()
        LIGHT_7_ENABLE = auto()

    def __init__(self):
        self.enabledColors = self.ColorEnableFlags(0b111111)
        self.difColor = [0.0, 0.0, 0.0, 1.0]
        self.ambColor = [0.0] * 4
        self.colorControl = self.ColorControlFlags(0)
        self.alphaControl = self.ColorControlFlags(0)

    def __eq__(self, other: "LightChannel"):
        return (
            isinstance(other, LightChannel)
            and self.enabledColors == other.enabledColors
            and self.difColor == other.difColor
            and self.ambColor == other.ambColor
            and self.colorControl == other.colorControl
            and self.alphaControl == other.alphaControl
        )


class LightChannelSerializer(Serializer[_MAT_SER_T, LightChannel]):

    DATA_TYPE = LightChannel
    _STRCT = Struct(">I 4B 4B II")


class LightChannelReader(LightChannelSerializer["MaterialReader"], Reader):

    @classmethod
    def size(cls):
        """Size of a light channel in bytes."""
        return cls._STRCT.size

    def unpack(self, data: bytes):
        self._data = lc = LightChannel()
        unpackedData = self._STRCT.unpack_from(data, self.offset)
        lc.enabledColors = LightChannel.ColorEnableFlags(unpackedData[0])
        lc.difColor = [normBitVal(c, 8) for c in unpackedData[1:5]]
        lc.ambColor = [normBitVal(c, 8) for c in unpackedData[5:9]]
        lc.colorControl = LightChannel.ColorControlFlags(unpackedData[9])
        lc.alphaControl = LightChannel.ColorControlFlags(unpackedData[10])
        return self


class LightChannelWriter(LightChannelSerializer["MaterialWriter"], Writer):

    def _calcSize(self):
        return self._STRCT.size

    def pack(self):
        lc = self._data
        colors = (denormBitVal(v, 8) for c in (lc.difColor, lc.ambColor) for v in c)
        return self._STRCT.pack(lc.enabledColors.value, *colors,
                                lc.colorControl.value, lc.alphaControl.value)


class Material():
    """Uses a TEV config with some additional settings to control the way meshes are rendered."""

    def __init__(self, name: str = None, tevConfig: "TEVConfig" = None):
        super().__init__()
        self.name = name
        self.tevConfig = tevConfig
        self.renderGroup = RenderGroup.OPA
        self.cullMode = gx.CullMode.NONE
        self.indTextures: list[IndTexture] = [IndTexture() for _ in range(gx.MAX_INDIRECTS)]
        self.indSRTs: list[IndTransform] = []
        self.textures: list[Texture] = []
        self.lightChans: list[LightChannel] = []
        self.alphaTestSettings = gx.AlphaTestSettings.ValStruct()
        self.depthSettings = gx.DepthSettings.ValStruct()
        self.blendSettings = gx.BlendSettings.ValStruct()
        self.constAlphaSettings = gx.ConstAlphaSettings.ValStruct()
        self.standColors = [[0.0] * 4 for _ in range(gx.MAX_TEV_STAND_COLORS)]
        self.constColors = [[0.0] * 4 for _ in range(gx.MAX_TEV_CONST_COLORS)]
        self.lightSet = -1
        self.fogSet = -1
        self.mtxGen: type[tf.MtxGenerator] = tf.MayaMtxGen2D

    def isDuplicate(self, other: "Material"):
        """Return True if this material has identical settings to another."""
        return (
            isinstance(other, Material)
            and self.name == other.name
            and (
                (self.tevConfig is None and other.tevConfig is None)
                or self.tevConfig.isDuplicate(other.tevConfig)
            )
            and self.renderGroup == other.renderGroup
            and self.cullMode == other.cullMode
            and self.indTextures == other.indTextures
            and self.indSRTs == other.indSRTs
            and self.textures == other.textures
            and self.lightChans == other.lightChans
            and self.alphaTestSettings == other.alphaTestSettings
            and self.depthSettings == other.depthSettings
            and self.blendSettings == other.blendSettings
            and self.constAlphaSettings == other.constAlphaSettings
            and self.standColors == other.standColors
            and self.constColors == other.constColors
            and self.lightSet == other.lightSet
            and self.fogSet == other.fogSet
            and self.mtxGen == other.mtxGen
        )

    @property
    def earlyDepthTest(self):
        """True if depth testing can be done before texture processing for this material.

        False if this isn't the case (i.e., texture processing can lead to pixel discarding)
        (i.e., the alpha test is utilized).
        """
        logic = self.alphaTestSettings.logic
        comps = self.alphaTestSettings.comps
        if logic is gx.AlphaLogicOp.AND:
            return all(c is gx.CompareOp.ALWAYS for c in comps)
        if logic is gx.AlphaLogicOp.OR:
            return any(c is gx.CompareOp.ALWAYS for c in comps)
        if logic is gx.AlphaLogicOp.XOR:
            return gx.CompareOp.ALWAYS in comps and gx.CompareOp.NEVER in comps
        if logic is gx.AlphaLogicOp.XNOR:
            return (all(c is gx.CompareOp.ALWAYS for c in comps)
                    or all(c is gx.CompareOp.NEVER for c in comps))


class MaterialSerializer(Serializer[_MDL0_SER_T, Material]):

    DATA_TYPE = Material
    _MAIN_STRCT = Struct(">IiiI B 3x BBBBI?bbx 4B 4b iIiiii 4x 256x 4x 96x II")
    _TEX_SRT_STRCT = Struct(">2f f 2f")
    _TEX_MTX_STRCT = Struct(">bbB? 12f")
    _CMD_STRCT = Struct(">224s 160s")


class MaterialReader(MaterialSerializer["MDL0Reader"], Reader, StrPoolReadMixin):

    def __init__(self, parent: "MDL0Reader" = None, offset = 0):
        super().__init__(parent, offset)
        self._tevConfigOffset: int = None

    def unpack(self, data: bytes):
        unpackedData = self._MAIN_STRCT.unpack_from(data, self.offset)
        # unpack simple properties
        self._data = mat = Material()
        mat.renderGroup = RenderGroup(unpackedData[4])
        mat.cullMode = gx.CullMode(unpackedData[9])
        mat.lightSet, mat.fogSet = unpackedData[11:13]
        self._tevConfigOffset = self.offset + unpackedData[21] if unpackedData[21] != 0 else None
        mat.mtxGen = self.parentSer.MTX_GEN_TYPES_2D[unpackedData[28]]
        # calculate offsets for more complex things to unpack
        texSrtOffset = self.offset + self._MAIN_STRCT.size
        texMtxOffset = texSrtOffset + self._TEX_SRT_STRCT.size * gx.MAX_TEXTURES
        lightingOffset = texMtxOffset + self._TEX_MTX_STRCT.size * gx.MAX_TEXTURES
        texListOffset = self.offset + unpackedData[23]
        cmdOffset = self.offset + unpackedData[26]
        # unpack light channels
        lightingEndOffset = lightingOffset + LightChannelReader.size() * unpackedData[6]
        for lcOffset in range(lightingOffset, lightingEndOffset, LightChannelReader.size()):
            mat.lightChans.append(LightChannelReader(self, lcOffset).unpack(data).getInstance())
        # read gx commands
        packedCmds = data[cmdOffset : cmdOffset + self._CMD_STRCT.size]
        cmds = [cmd for cmd in gx.read(packedCmds) if isinstance(cmd, gx.LoadReg)]
        mat.alphaTestSettings = gx.LoadReg.filterRegs(cmds, gx.AlphaTestSettings)[0].bits
        mat.depthSettings = gx.LoadReg.filterRegs(cmds, gx.DepthSettings)[0].bits
        mat.blendSettings = gx.LoadReg.filterRegs(cmds, gx.BlendSettings)[0].bits
        mat.constAlphaSettings = gx.LoadReg.filterRegs(cmds, gx.ConstAlphaSettings)[0].bits
        # indirect textures
        indScls = [s for r in gx.LoadReg.filterRegs(cmds, gx.IndCoordScale) for s in r.bits.scales]
        indInfo = zip(unpackedData[13:17], unpackedData[17:21], indScls)
        mat.indTextures = [IndTexture(IndTexMode(m), l, s) for m, l, s in indInfo]
        # indirect transforms: create one from each set of 3
        indMtxSettings = gx.LoadReg.filterRegs(cmds, gx.IndMtxSettings)
        for settings in (indMtxSettings[i : i + 3] for i in range(0, len(indMtxSettings), 3)):
            mat.indSRTs.append(IndTransform.fromMtxSettings(settings))
        colorRegs = gx.LoadReg.filterRegs(cmds, gx.TEVColorReg)
        # color registers
        for reg in colorRegs:
            odd = reg.idx % 2 # even - ra, odd - bg
            cIdx = reg.idx // 2
            # there's no standard color 0 (that's the final output), so indices for that start at 1
            c = mat.constColors[cIdx] if reg.bits.isConstant else mat.standColors[cIdx - 1]
            if odd:
                c[2] = reg.bits.b
                c[1] = reg.bits.g
            else:
                c[0] = reg.bits.r
                c[3] = reg.bits.a
        # textures
        for texIdx, coordSettings in enumerate(gx.LoadReg.filterRegs(cmds, gx.TexCoordSettings)):
            # texture info is stored in 3 different places:
            # main texture
            texOffset = texListOffset + texIdx * TextureReader.size()
            tex: Texture = TextureReader(self, texOffset).unpack(data).getInstance()
            mat.textures.append(tex)
            # srt
            srtOffset = texSrtOffset + texIdx * self._TEX_SRT_STRCT.size
            srtInfo = self._TEX_SRT_STRCT.unpack_from(data, srtOffset)
            tex.setSRT(srtInfo[0:2], srtInfo[2:3], srtInfo[3:5])
            # matrix & mapping info
            mtxOffset = texMtxOffset + texIdx * self._TEX_MTX_STRCT.size
            mtxInfo = self._TEX_MTX_STRCT.unpack_from(data, mtxOffset)
            tex.usedCam, tex.usedLight = mtxInfo[0:2]
            tex.mapMode = TexMapMode(mtxInfo[2])
            if tex.mapMode is TexMapMode.UV:
                tex.coordIdx = coordSettings.bits.texCoordSrc.value - gx.TexCoordSource.UV_0.value
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.name = self.parentSer.itemName(self)
        if self._tevConfigOffset is not None:
            tevConfigReaders = self.parentSer.section(TEVConfigReader).values()
            tevConfigReader = next(t for t in tevConfigReaders if t.offset == self._tevConfigOffset)
            self._data.tevConfig = tevConfigReader.getInstance()


class MaterialWriter(MaterialSerializer["MDL0Writer"], Writer, StrPoolWriteMixin):

    def __init__(self, parent: "MDL0Writer" = None, offset = 0):
        super().__init__(parent, offset)
        self.textures: list[TextureWriter] = []

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        # texture writer offsets & material size are both dependent on material offset
        for texWriter in self.textures:
            texWriter.offset = texWriter.offset - self._offset + o
        self._offset = o
        self._size = self._calcSize()

    def getStrings(self):
        return set().union(*(tex.getStrings() for tex in self.textures))

    def fromInstance(self, data: Material):
        super().fromInstance(data)
        offset = self.offset + self._preTexSize()
        for tex in self._data.textures:
            writer = TextureWriter(self, offset).fromInstance(tex)
            self.textures.append(writer)
            offset += writer.size()
        return self

    @classmethod
    def _preTexSize(cls):
        """Size (in bytes) of a packed material up to its textures."""
        size = cls._MAIN_STRCT.size
        size += cls._TEX_SRT_STRCT.size * gx.MAX_TEXTURES
        size += cls._TEX_MTX_STRCT.size * gx.MAX_TEXTURES
        size += LightChannelReader.size() * gx.MAX_CLR_ATTRS
        return size

    def _calcSize(self):
        size = self._preTexSize() + sum(writer.size() for writer in self.textures)
        return pad(size, 32, self.offset - self.parentSer.offset) + self._CMD_STRCT.size

    def _getColorRegs(self):
        """Get this writer's material's standard/constant colors in their GX registers.

        Returns a list where the first entry is a list of standard registers, and the second is a
        list of constant registers.
        """
        mat = self._data
        writeClrRegs = []
        constClrRegs = []
        for clrIdx, color in enumerate(mat.standColors + mat.constColors):
            isConstant = clrIdx >= len(mat.standColors)
            regs = None
            if not isConstant:
                regs = writeClrRegs
                clrIdx += 1 # there's no standard color 0 (that's the final output), so start at 1
            else:
                regs = constClrRegs
                clrIdx -= len(mat.standColors)
            for cmdIdx in range(2): # 2 cmds per color, each w/ 2 of its components
                cmd = gx.TEVColorReg(clrIdx * 2 + cmdIdx)
                cmd.bits.isConstant = isConstant
                if cmdIdx == 0:
                    cmd.bits.r = color[0]
                    cmd.bits.a = color[3]
                if cmdIdx == 1:
                    cmd.bits.b = color[2]
                    cmd.bits.g = color[1]
                # for some reason, the first cmd for non-const colors appears thrice
                regs += [cmd] * (3 if not isConstant and cmdIdx == 1 else 1)
        return [writeClrRegs, constClrRegs]

    def _getTexCoordCmds(self):
        """Get XF load commands for this writer's material's textures' coordinate settings.

        Returns a list with two commands for each texture, alternating between coord settings and
        post effect settings.
        """
        regs: list[gx.LoadXF] = []
        for i, tex in enumerate(self._data.textures):
            coordSettings = gx.TexCoordSettings(i)
            postEffectSettings = gx.PostEffectSettings(i)
            cSets = coordSettings.bits
            eSets = postEffectSettings.bits
            eSets.mtxIdx = i * 3
            if tex.mapMode is TexMapMode.UV: # tex coords are 2d
                cSets.projectionType = gx.TexProjectionType.ST
                cSets.inputForm = gx.TexInputForm.AB11
                cSets.texCoordSrc = gx.TexCoordSource(tex.coordIdx + gx.TexCoordSource.UV_0.value)
            else: # tex coords are projection-based
                cSets.projectionType = gx.TexProjectionType.STQ
                cSets.inputForm = gx.TexInputForm.ABC1
            if tex.mapMode is TexMapMode.PROJECTION:
                cSets.texCoordSrc = gx.TexCoordSource.POSITION
            elif tex.mapMode in (TexMapMode.ENV_CAM, TexMapMode.ENV_LIGHT, TexMapMode.ENV_SPEC):
                cSets.texCoordSrc = gx.TexCoordSource.NORMAL
                eSets.normalize = True
            # always 5 in nsmbw, no idea why
            cSets.embossSrc = 5
            for settings in (coordSettings, postEffectSettings):
                regs.append(gx.LoadXF(settings))
        return regs

    def _texFlags(self):
        """Return an int representing some flags for each of this material's textures.

        Used for packing to bytes.
        """
        flags = 0
        for i, texWriter in enumerate(self.textures):
            flags |= (texWriter.getFlags().value << (i * 4))
        return flags

    def pack(self):
        mat = self._data
        # pack textures
        packedSRTs = packedMtcs = packedTexs = b""
        dummyTexWriter = TextureWriter(self).fromInstance(Texture())
        for texWriter in fillList(self.textures, gx.MAX_TEXTURES, dummyTexWriter):
            # textures are always packed w/ an identity matrix for some reason
            mtx = np.identity(4).flatten()[:12]
            tex = texWriter.getInstance()
            mapMode = tex.mapMode.value
            packedSRTs += self._TEX_SRT_STRCT.pack(*tex.scale, *tex.rot, *tex.trans)
            packedMtcs += self._TEX_MTX_STRCT.pack(tex.usedCam, tex.usedLight, mapMode, True, *mtx)
            if texWriter is not dummyTexWriter:
                packedTexs += texWriter.pack()
        # pack light channels
        filledLCs = fillList(mat.lightChans, gx.MAX_CLR_ATTRS, LightChannel())
        packedLCs = b"".join(LightChannelWriter(self).fromInstance(lc).pack() for lc in filledLCs)
        # pack cmds: this becomes a list of lists of cmds, where each subset is padded to 16 on pack
        cmds = []
        # start w/ the easy ones
        mainCmds = [gx.AlphaTestSettings(v=mat.alphaTestSettings),
                    gx.DepthSettings(v=mat.depthSettings),
                    gx.BPMask(v=gx.BPMask.ValStruct(0b000000001111111111100011)),
                    gx.BlendSettings(v=mat.blendSettings),
                    gx.ConstAlphaSettings(v=mat.constAlphaSettings)]
        cmds.append(mainCmds)
        # color registers
        cmds += self._getColorRegs()
        # indirect stuff
        cmds.append([gx.NOP(16)]) # 16 extra bytes of padding between color regs and indirect cmds
        indCmds = []
        for i, texs in enumerate(zip(mat.indTextures[::2], mat.indTextures[1::2])):
            # for each pair of indirect textures, make a command w/ their coord scales
            cmd = gx.IndCoordScale(i)
            cmd.bits.scales = (tex.coordScales for tex in texs)
            indCmds.append(cmd)
        mtxCmds = [[cmd for cmd in indSRT.toMtxSettings(i)] for i, indSRT in enumerate(mat.indSRTs)]
        if mtxCmds:
            # first ind mtx is grouped w/ coord scales; others are separate
            # (unknown if others should be grouped together or separate from one another,
            # as more than 2 matrices are never used in nsmbw - grouping together seems to work
            # though, at least on dolphin)
            indCmds += mtxCmds[0]
        cmds.append(indCmds)
        cmds.append([cmd for mtx in mtxCmds[1:] for cmd in mtx])
        # process and pack commands
        packedCmds = b""
        for subset in cmds:
            subset = [gx.LoadBP(cmd) if isinstance(cmd, gx.BPReg) else cmd for cmd in subset]
            packedCmds += pad(gx.pack(subset), 16)
        # bp cmds followed by xf () cmds
        packedCmds = self._CMD_STRCT.pack(packedCmds, gx.pack(self._getTexCoordCmds()))
        # pack header & return
        offset = self.offset
        texOffset = self._MAIN_STRCT.size + len(packedSRTs + packedMtcs + packedLCs)
        cmdOffset = pad(texOffset + len(packedTexs), 32, offset - self.parentSer.offset)
        numTexs = len(self.textures)
        numStages = len(mat.tevConfig.stages) if mat.tevConfig is not None else 0
        mtxGen = self.parentSer.MTX_GEN_TYPES_2D.index(self._data.mtxGen)
        tevCfgWriters = self.parentSer.section(TEVConfigWriter).values()
        try:
            tevCfgOffset = next(t for t in tevCfgWriters if t.getInstance() is mat.tevConfig).offset
        except StopIteration: # no tev config for this material
            tevCfgOffset = offset
        head = self._MAIN_STRCT.pack(self._size, self.parentSer.offset - offset,
                                     self.stringOffset(mat.name) - offset,
                                     self.parentSer.itemIdx(self), mat.renderGroup.value, numTexs,
                                     len(mat.lightChans), numStages, len(mat.indSRTs),
                                     mat.cullMode.value, mat.earlyDepthTest,
                                     mat.lightSet, mat.fogSet,
                                     *(t.mode.value for t in mat.indTextures),
                                     *(t.lightIdx for t in mat.indTextures),
                                     tevCfgOffset - offset, numTexs, texOffset if numTexs else 0,
                                     0, 0, cmdOffset, self._texFlags(), mtxGen)
        mainBody = head + packedSRTs + packedMtcs + packedLCs + packedTexs
        return pad(mainBody, 32, offset - self.parentSer.offset) + packedCmds


@dataclass
class ColorSwap():
    """Part of a TEV config that lets you swap color channels - read more in gx.TEVColorSettings."""

    r: gx.ColorChannel = gx.ColorChannel.R
    g: gx.ColorChannel = gx.ColorChannel.G
    b: gx.ColorChannel = gx.ColorChannel.B
    a: gx.ColorChannel = gx.ColorChannel.A

    def fromReg(self, reg: gx.TEVColorSettings):
        """Modify this color swap based on data from a color settings BP register."""
        odd = reg.idx % 2
        if not odd:
            self.r = reg.bits.swapR
            self.g = reg.bits.swapG
        else:
            self.b = reg.bits.swapB
            self.a = reg.bits.swapA

    def toReg(self, reg: gx.TEVColorSettings):
        """Modify a color settings BP register based on this color swap's data."""
        odd = reg.idx % 2
        if not odd:
            reg.bits.swapR = self.r
            reg.bits.swapG = self.g
        else:
            reg.bits.swapB = self.b
            reg.bits.swapA = self.a

    def __eq__(self, other: "ColorSwap"):
        return self.r == other.r and self.g == other.g and self.b == other.b and self.a == other.a


class TEVStage():
    """Defines a set of calculations used in TEV rendering."""

    def __init__(self):
        self.constColorSel: gx.TEVConstSel = gx.TEVConstSel.VAL_8_8
        self.constAlphaSel: gx.TEVConstSel = gx.TEVConstSel.VAL_8_8
        self.texIdx = self.texCoordIdx = gx.MAX_TEXTURES - 1
        self.rasterSel = gx.TEVRasterSel.ZERO
        self.indSettings = gx.TEVIndSettings.ValStruct()
        self.colorParams = gx.TEVColorParams.ValStruct()
        self.alphaParams = gx.TEVAlphaParams.ValStruct()

    def usesTex(self):
        """Return true if this stage uses a texture, and false otherwise."""
        return (gx.TEVColorArg.TEX_COLOR in self.colorParams.args or
                gx.TEVColorArg.TEX_ALPHA in self.colorParams.args or
                gx.TEVAlphaArg.TEX_ALPHA in self.alphaParams.args)

    def usesInd(self):
        """Return true if this stage uses an indirect texture operation, and false otherwise."""
        return self.indSettings.mtxIdx is not gx.IndMtxIdx.NONE

    def fromColorSettings(self, reg: gx.TEVColorSettings, odd = False):
        """Modify this stage's const indices based on data from a BP register.

        Since this register contains data for two stages, the "odd" parameter can be used to set
        whether this stage is the even one (first) or the odd one (second).
        """
        self.constColorSel = reg.bits.constColorSels[odd]
        self.constAlphaSel = reg.bits.constAlphaSels[odd]

    def fromSources(self, reg: gx.TEVSources, odd = False):
        """Modify this stage's texture/coord/raster sources based on data from a BP register.

        Since this register contains data for two stages, the "odd" parameter can be used to set
        whether this stage is the even one (first) or the odd one (second).
        """
        self.texIdx = reg.bits.texIdcs[odd]
        self.texCoordIdx = reg.bits.texCoordIdcs[odd]
        self.rasterSel = reg.bits.rasterSels[odd]

    def __eq__(self, other: "TEVStage"):
        return (self.constColorSel == other.constColorSel and
                self.constAlphaSel == other.constAlphaSel and
                self.texIdx == other.texIdx and
                self.texCoordIdx == other.texCoordIdx and
                self.rasterSel == other.rasterSel and
                self.indSettings == other.indSettings and
                self.colorParams == other.colorParams and
                self.alphaParams == other.alphaParams)


class TEVConfig():
    """Configuration for the Wii's Texture EnVironment, which controls material rendering.

    The TEV renders in stages, going sequentially through them all until everything's been
    calculated and some output color's been reached. There can be up to 16.
    """

    def __init__(self):
        super().__init__()
        self.stages: list[TEVStage] = []
        self.colorSwaps = [ColorSwap() for _ in range(gx.MAX_COLOR_SWAPS)]
        self.indSources = gx.IndSources.ValStruct()

    def isDuplicate(self, other: "TEVConfig"):
        """Return True if this TEV config has identical settings to another."""
        return (
            isinstance(other, TEVConfig)
            and self.stages == other.stages
            and self.colorSwaps == other.colorSwaps
            and self.indSources == other.indSources
        )


class TEVConfigSerializer(Serializer[_MDL0_SER_T, TEVConfig]):

    DATA_TYPE = TEVConfig
    _STRCT = Struct(">IiIB 3x 8b 8x 480s")


class TEVConfigReader(TEVConfigSerializer["MDL0Reader"], Reader, StrPoolReadMixin):

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = TEVConfig()
        unpackedData = self._STRCT.unpack_from(data, self.offset)
        # parse gx bp commands to generate this tev config's properties
        cmds = [cmd for cmd in gx.read(unpackedData[12]) if isinstance(cmd, gx.LoadBP)]
        # indirect sources
        self._data.indSources = gx.LoadReg.filterRegs(cmds, gx.IndSources)[0].bits
        # swap settings
        # the swap and const settings are distinguished here based on this data's structure:
        # the first 8 cmds (besides masks) are color settings that set up the swap table
        # after that, all cmds for the 1st two stages are all put together, followed by next 2, etc
        # color settings commands get used again in this section, this time for setting the consts
        # masks are used for this behavior, but since we know how it's structured we can just
        # grab the first 8 (num color swaps * 2) for swaps and the last 8 for consts
        swapSettings = gx.LoadReg.filterRegs(cmds, gx.TEVColorSettings)[:gx.MAX_COLOR_SWAPS * 2]
        for i, swapSetting in enumerate(swapSettings):
            swap = self._data.colorSwaps[i // 2]
            swap.fromReg(swapSetting)
        # stages
        constSettings = gx.LoadReg.filterRegs(cmds, gx.TEVColorSettings)[gx.MAX_COLOR_SWAPS * 2:]
        sourceParams = gx.LoadReg.filterRegs(cmds, gx.TEVSources)
        colorParams = gx.LoadReg.filterRegs(cmds, gx.TEVColorParams)
        alphaParams = gx.LoadReg.filterRegs(cmds, gx.TEVAlphaParams)
        indSettings = gx.LoadReg.filterRegs(cmds, gx.TEVIndSettings)
        usedStages = unpackedData[3]
        self._data.stages = [TEVStage() for _ in range(usedStages)]
        for i, stage in enumerate(self._data.stages):
            odd = i % 2 != 0 # is this an odd stage? used for reading cmds that affect 2 stages
            stage.fromColorSettings(constSettings[i // 2], odd)
            stage.fromSources(sourceParams[i // 2], odd)
            stage.colorParams = colorParams[i].bits
            stage.alphaParams = alphaParams[i].bits
            stage.indSettings = indSettings[i].bits
        return self


class TEVConfigWriter(TEVConfigSerializer["MDL0Writer"], Writer, StrPoolWriteMixin):

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def _calcSize(self):
        return self._STRCT.size

    def pack(self):
        # this becomes a list of lists of cmds, where each subset is padded to 16 on pack
        # the first set is main commands for the whole config, then the rest are for pairs of stages
        cmds = []
        # pack main cmds for the whole tev config (color swaps, indirect sources)
        swapMasks = (gx.BPMask(v=gx.BPMask.ValStruct(0b1111)) for _ in range(8))
        swapRegs = [gx.TEVColorSettings(i) for i in range(gx.MAX_COLOR_SWAPS * 2)]
        for i, swap in enumerate(self._data.colorSwaps):
            for j in range(2): # 2 cmds per swap
                swap.toReg(swapRegs[i * 2 + j])
        indSrcReg = gx.IndSources(v=self._data.indSources)
        mainCmds = [reg for pair in zip(swapMasks, swapRegs) for reg in pair] + [indSrcReg]
        cmds.append(mainCmds)
        # pack commands for stage pairs
        # (note that an extra stage may be created as a dummy so that they can be packed as such)
        evenStages = fillList(self._data.stages, 2, TEVStage())
        for pairIdx, pair in enumerate(zip(evenStages[::2], evenStages[1::2])):
            pairCmds = [] # commands for this pair of stages
            # constant colors
            pairCmds.append(gx.BPMask(v=gx.BPMask.ValStruct(0b111111111111111111110000)))
            constReg = gx.TEVColorSettings(pairIdx)
            constReg.bits.constAlphaSels = [stage.constAlphaSel for stage in pair]
            constReg.bits.constColorSels = [stage.constColorSel for stage in pair]
            pairCmds.append(constReg)
            # texture/color/coord sources
            srcParams = gx.TEVSources(pairIdx)
            srcParams.bits.texIdcs = [stage.texIdx for stage in pair]
            srcParams.bits.texCoordIdcs = [stage.texCoordIdx for stage in pair]
            srcParams.bits.texEnables = [stage.usesTex() for stage in pair]
            srcParams.bits.rasterSels = [stage.rasterSel for stage in pair]
            pairCmds.append(srcParams)
            # cmds for individual stages
            stageCmds = []
            for stageIdx, stage in enumerate(pair):
                # only make these cmds for real stages (not the possible dummy)
                if id(stage) in (id(s) for s in self._data.stages):
                    idx = pairIdx * 2 + stageIdx
                    personalCmds = []
                    personalCmds.append(gx.TEVColorParams(idx, v=stage.colorParams))
                    personalCmds.append(gx.TEVAlphaParams(idx, v=stage.alphaParams))
                    personalCmds.append(gx.TEVIndSettings(idx, v=stage.indSettings))
                    stageCmds.append(personalCmds)
                else:
                    stageCmds.append([gx.NOP(5)] * 3)
            # interleave cmds for each stage in the pair
            pairCmds += [cmd for stage in zip(*stageCmds) for cmd in stage]
            # add cmds
            cmds.append(pairCmds)
        # process and pack commands
        packedCmds = b""
        for subset in cmds:
            subset = [gx.LoadBP(cmd) if isinstance(cmd, gx.BPReg) else cmd for cmd in subset]
            packedCmds += pad(gx.pack(subset), 16)
        # get used texture indices
        usedTexIdcs = set()
        for stage in self._data.stages:
            if stage.usesTex():
                usedTexIdcs.add(stage.texIdx)
                if stage.usesInd():
                    usedTexIdcs.add(self._data.indSources.texIdcs[stage.indSettings.indirectID])
        texIdcs = (t if t in usedTexIdcs else -1 for t in range(gx.MAX_TEXTURES))
        # finally, pack the header and we're done
        return self._STRCT.pack(self._size, self.parentSer.offset - self.offset,
                                self.parentSer.itemIdx(self), len(self._data.stages),
                                *texIdcs, packedCmds)


class DrawGroup():
    """Group of draw commands and deformers they're allowed to use."""

    def __init__(self, deformers: list[Deformer] = None, cmds: list[gx.DrawPrimitives] = None):
        self.deformers = deformers if deformers is not None else []
        self.cmds = cmds if cmds is not None else []


class Mesh():
    """Brings the other MDL0 types together to form a part of the physical model."""

    def __init__(self, name: str = None):
        super().__init__()
        self.name = name
        self.vertGroups: dict[type[VertexAttrGroup], dict[int, VertexAttrGroup]] = {
            PsnGroup: {},
            NrmGroup: {},
            ClrGroup: {},
            UVGroup: {}
        }
        self.mat: Material = None
        self.visJoint: Joint = None # joint that determines this object's visibility
        self.drawGroups: list[DrawGroup] = []
        self.drawPrio = 0 # 1 = max priority, 255 = min priority, 0 = arbitrarily decided
        self.singleBind: Deformer = None # deformer to use if none in draw groups

    @property
    def cmds(self):
        """Generator for all commands used in this mesh's draw groups."""
        return (cmd for dg in self.drawGroups for cmd in dg.cmds)

    def numFaces(self):
        """Number of faces in this mesh."""
        return sum(cmd.numFaces() for cmd in self.cmds)

    def numVerts(self):
        """Number of vertices in this mesh."""
        return sum(len(cmd) for cmd in self.cmds)

    def getDeformers(self, ignoreSingle = False) -> set[Deformer]:
        """Return a set of the deformers used by this mesh.

        Optionally, you can choose not to count single-bind usage.
        """
        dfs = {self.singleBind} if self.singleBind is not None and not ignoreSingle else set()
        dfs |= {d for dg in self.drawGroups for d in dg.deformers}
        return dfs

    def hasPsnMtcs(self):
        """Return true if this mesh has commands for loading position matrices in its vertices.

        (i.e., it uses weight-based rigging rather than just the single bind)
        """
        return any(len(dg.deformers) > 0 for dg in self.drawGroups)

    def hasNrmMtcs(self):
        """Return true if this mesh has commands for loading normal matrices in its vertices.

        (i.e., it uses weight-based rigging and has custom normals)
        """
        return self.hasPsnMtcs() and len(self.vertGroups[NrmGroup]) > 0

    def hasTexMtcs(self):
        """Return true if this mesh has commands for loading texture matrices in its vertices.

        (i.e., it uses weight-based rigging and has textures based on positions or normals)
        """
        return self.hasPsnMtcs() and any(t.mapMode is not TexMapMode.UV for t in self.mat.textures)

    def getVertexDef(self):
        """Generate a vertex definition for this mesh based on the attributes it uses."""
        # note the commented lines in this function
        # they represent a different approach for figuring out the necessary vertex stride -
        # basing index size for each attribute on the highest used index rather than group size
        # this can lead to lossless compression better than the original models, but since they use
        # the group size method, i use the group size method
        vDef = gx.VertexDef()
        # matrices
        if self.hasPsnMtcs():
            vDef.psnMtcs[0].dec = gx.MtxAttrDec.IDX8
            for mtxDef, tex in zip(vDef.texMtcs, self.mat.textures):
                if tex.mapMode is not TexMapMode.UV:
                    mtxDef.dec = gx.MtxAttrDec.IDX8
        # other attributes, based on attr groups
        # bigIdcs = np.any(np.concatenate([cmd.vertData for cmd in self.cmds]) > maxBitVal(8), 0)
        attrPairs = ((self.vertGroups[PsnGroup], vDef.psns), (self.vertGroups[NrmGroup], vDef.nrms),
                     (self.vertGroups[ClrGroup], vDef.clrs), (self.vertGroups[UVGroup], vDef.uvs))
        # attrIdx = gx.PSN_ATTR_IDX
        for groups, defs in attrPairs:
            for i, g in groups.items():
                d = defs[i]
                d.fmt = g.getAttr()
                d.dec = gx.StdAttrDec.IDX8 if len(g) <= maxBitVal(8) else gx.StdAttrDec.IDX16
                # d.dec = gx.StdAttrDec.IDX16 if bigIdcs[attrIdx + i] else gx.StdAttrDec.IDX8
            # attrIdx += len(defs)
        return vDef


class MeshSerializer(Serializer[_MDL0_SER_T, Mesh]):

    DATA_TYPE = Mesh
    _STRCT = Struct(">Iii 4s 4s 4s 2I i 2I i IIiIII 14h i")
    _DEF_INFO_OFFSET = 24 # offset to def info in header
    _DATA_INFO_OFFSET = 36 # offset to data info in header
    _DEF_SIZE = 224


class MeshReader(MeshSerializer["MDL0Reader"], Reader, StrPoolReadMixin):

    def __init__(self, parent: "MDL0Reader" = None, offset = 0):
        super().__init__(parent, offset)
        self._matIdx: int = None
        self._visJointIdx: int = None
        self._singleBindIdx: int = None
        self._drawGroupDeformerIdcs: list[list[int]] = [] # deformer idcs for each draw group
        self._vertGroupIdcs: dict[type[VertexAttrGroup], list[int]] = {
            PsnGroup: [-1] * gx.MAX_PSN_ATTRS,
            NrmGroup: [-1] * gx.MAX_NRM_ATTRS,
            ClrGroup: [-1] * gx.MAX_CLR_ATTRS,
            UVGroup: [-1] * gx.MAX_UV_ATTRS
        }

    def applyDef(self, cmd: DefMeshRendering):
        """Update one entry in this array based on a deformer definition command."""
        self._matIdx = cmd.matIdx
        self._visJointIdx = cmd.visJointIdx
        self._data.drawPrio = cmd.drawPrio

    def unpack(self, data: bytes):
        super().unpack(data)
        unpackedData = self._STRCT.unpack_from(data, self.offset)
        self._data = Mesh()
        self._singleBindIdx = unpackedData[2] if unpackedData[2] != -1 else None
        # get data groups
        self._vertGroupIdcs[PsnGroup] = unpackedData[18:19]
        self._vertGroupIdcs[NrmGroup] = unpackedData[19:20]
        self._vertGroupIdcs[ClrGroup] = unpackedData[20:22]
        self._vertGroupIdcs[UVGroup] = unpackedData[22:30]
        # read vertex def
        vertexDefSize = unpackedData[7]
        vertexDefOffset = self.offset + self._DEF_INFO_OFFSET + unpackedData[8]
        vertexDefBytes = data[vertexDefOffset : vertexDefOffset + vertexDefSize]
        vertexDef = gx.VertexDef()
        for cmd in gx.read(vertexDefBytes):
            if isinstance(cmd, gx.LoadCP):
                cmd.reg.bits.applyTo(vertexDef)
        # read vertex data
        vertexDataSize = unpackedData[10]
        vertexDataOffset = self.offset + self._DATA_INFO_OFFSET + unpackedData[11]
        vertexDataBytes = data[vertexDataOffset : vertexDataOffset + vertexDataSize]
        latestGroup = DrawGroup() # most recently created draw group
        latestDeformers = []
        for cmd in vertexDef.read(vertexDataBytes):
            if isinstance(cmd, gx.DrawPrimitives):
                latestGroup.cmds.append(cmd)
            elif isinstance(cmd, gx.LoadPsnMtx):
                # load commands are found at the start of draw groups
                # so, if a load command is read and the latest group isn't new, make a new one
                if len(latestGroup.cmds) > 0:
                    self._data.drawGroups.append(latestGroup)
                    self._drawGroupDeformerIdcs.append(latestDeformers)
                    latestGroup = DrawGroup()
                    latestDeformers = []
                latestDeformers.append(cmd.idx)
        if len(latestGroup.cmds) > 0:
            self._data.drawGroups.append(latestGroup)
            self._drawGroupDeformerIdcs.append(latestDeformers)
        return self

    def _updateInstance(self):
        super()._updateInstance()
        m = self._data
        m.name = self.parentSer.itemName(self)
        # material
        mats = tuple(self.parentSer.section(MaterialReader).values())
        m.mat = mats[self._matIdx].getInstance()
        # visibility joint
        joints = tuple(self.parentSer.section(JointReader).values())
        m.visJoint = joints[self._visJointIdx].getInstance()
        # deformers
        deformers = self.parentSer.deformers.getInstance()
        if self._singleBindIdx is not None:
            m.singleBind = deformers[self._singleBindIdx]
        for dg, deformerIdcs in zip(m.drawGroups, self._drawGroupDeformerIdcs):
            dg.deformers = [deformers[i] for i in deformerIdcs]
        # vertex data groups
        for reader in (PsnGroupReader, NrmGroupReader, ClrGroupReader, UVGroupReader):
            gType = reader.DATA_TYPE
            grps = tuple(self.parentSer.section(reader).values())
            idcs = self._vertGroupIdcs[gType]
            m.vertGroups[gType] = {i: grps[g].getInstance() for i, g in enumerate(idcs) if g != -1}


class MeshWriter(MeshSerializer["MDL0Writer"], Writer, StrPoolWriteMixin):

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def _packDeformers(self):
        """Pack this mesh's deformers via references to the main weight array."""
        mainArr = self.parentSer.deformers.getInstance()
        selfArr = self._data.getDeformers(True)
        numDeformers = len(selfArr)
        if numDeformers == 0:
            return b""
        indices = (i for i, d in enumerate(mainArr) if d in selfArr) # sort by index into main arr
        return Struct(">I" + "H" * numDeformers).pack(numDeformers, *indices)

    def _calcSize(self):
        mesh = self._data
        # deformer array size
        numDeformers = len(self._data.getDeformers(True))
        dfArrSize = Struct(">I" + "H" * numDeformers).size if numDeformers > 0 else 0
        # size of each draw command
        vertexDef = mesh.getVertexDef()
        vertexStride = vertexDef.stride
        drawSize = sum(len(c) * vertexStride + 3 for g in mesh.drawGroups for c in g.cmds)
        # size of matrix load commands: 5 bytes * (1-3 commands) * # deformers per group
        numLoads = sum((mesh.hasTexMtcs(), mesh.hasPsnMtcs(), mesh.hasNrmMtcs())) # num cmds per d
        loadSize = 5 * numLoads # size (bytes) of load cmds for one d, get all in next line
        loadSize *= sum(len(g.deformers) for g in mesh.drawGroups)
        # total padded size of all commands
        vertexDataSize = pad(drawSize + loadSize, 32)
        return pad(self._STRCT.size + dfArrSize, 32) + self._DEF_SIZE + vertexDataSize

    def pack(self):
        mesh = self._data
        # pack deformers
        packedDeformers = self._packDeformers()
        preCmdSize = pad(self._STRCT.size + len(packedDeformers), 32) # size of everything pre-cmds
        # pack vertex def
        # note that there's some weird padding here - i just apply it manually through nop commands
        # bc i think that's clearer than a bunch of weird pad() calls would be
        vDef = mesh.getVertexDef()
        decs = vDef.getDecs()
        counts = vDef.getCounts()
        vertexDefCmds = [gx.NOP(10)] + decs + [counts, gx.NOP(1)] + vDef.getFmts()
        packedVertexDef = pad(gx.pack(vertexDefCmds), self._DEF_SIZE)
        # pack vertex data
        drawCmds = []
        hasPsnMtcs = mesh.hasPsnMtcs()
        hasTexMtcs = mesh.hasTexMtcs()
        hasNrmMtcs = mesh.hasNrmMtcs()
        allDeformerIdcs = self.parentSer.deformers.indices
        for dg in mesh.drawGroups:
            groupDeformers = dg.deformers
            deformerIdcs = [allDeformerIdcs[d] for d in groupDeformers]
            if hasTexMtcs:
                drawCmds += [gx.LoadTexMtx(mtxIdx=i, idx=e) for i, e in enumerate(deformerIdcs)]
            if hasPsnMtcs:
                drawCmds += [gx.LoadPsnMtx(mtxIdx=i, idx=e) for i, e in enumerate(deformerIdcs)]
            if hasNrmMtcs:
                drawCmds += [gx.LoadNrmMtx(mtxIdx=i, idx=e) for i, e in enumerate(deformerIdcs)]
            drawCmds += dg.cmds
        packedVertexData = pad(vDef.pack(drawCmds), 32)
        # pack data group indices
        groupIdcs = ()
        for writer in (PsnGroupWriter, NrmGroupWriter, ClrGroupWriter, UVGroupWriter):
            groupType = writer.DATA_TYPE
            allGroups = tuple(self.parentSer.section(writer))
            used = mesh.vertGroups[groupType]
            slots = range(groupType.ATTR_TYPE.MAX_ATTRS)
            groupIdcs += tuple(allGroups.index(used[s].name) if s in used else -1 for s in slots)
        # create main header and pack
        singleBind = -1 if mesh.singleBind is None else allDeformerIdcs[mesh.singleBind]
        header = self._STRCT.pack(self._size, self.parentSer.offset - self.offset, singleBind,
                                  *(d.reg.bits.pack() for d in decs), counts.reg.bits.pack(),
                                  self._DEF_SIZE,
                                  self._DEF_SIZE, # this is supposed to be the size of the vertex data minus padding; not sure how to calculate this properly, as it apparently depends on the number of vertex attributes used (usually 128, but see bgA_A402), but this seems to work fine
                                  preCmdSize - self._DEF_INFO_OFFSET,
                                  len(packedVertexData), len(packedVertexData),
                                  preCmdSize + self._DEF_SIZE - self._DATA_INFO_OFFSET,
                                  vDef.getFlags(), 0, self.stringOffset(mesh.name) - self.offset,
                                  self.parentSer.itemIdx(self), mesh.numVerts(), mesh.numFaces(),
                                  *groupIdcs, -1, -1, self._STRCT.size)
        return pad(header + packedDeformers, 32) + packedVertexDef + packedVertexData


class ResourceLinkWriter(AddressedSerializable["MDL0Writer"], Writable["MDL0Writer"]):
    """Writes data that points an external resource to where it's used in this model."""

    _HEAD_STRCT = Struct(">I")
    _ENTRY_STRCT = Struct(">ii")

    def __init__(self, parent: "MDL0Writer" = None, offset = 0, texs: list[Texture] = None):
        super().__init__(parent, offset)
        self.textures = texs if texs is not None else []

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, o: int):
        self._offset = o

    def size(self):
        return self._HEAD_STRCT.size + self._ENTRY_STRCT.size * len(self.textures)

    def pack(self):
        output = self._HEAD_STRCT.pack(len(self.textures))
        mws = self.parentSer.section(MaterialWriter).values() # all the model's material writers
        for tex in self.textures:
            mw, tw = next((m, t) for m in mws for t in m.textures if t.getInstance() is tex)
            matOffset = calcOffset(self.offset, mw.offset)
            texOffset = calcOffset(self.offset, tw.offset)
            output += self._ENTRY_STRCT.pack(matOffset, texOffset)
        return output


class TextureLinkWriter(ResourceLinkWriter):
    """Writes data that points an external image to where it's used in this model."""


class PaletteLinkWriter(ResourceLinkWriter):
    """Writes data that points an external palette to where it's used in this model."""


class MDL0(Subfile):
    """BRRES subfile for models."""

    _VALID_VERSIONS = (11, )

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.rootJoint: Joint = None
        self.vertGroups: dict[type[VertexAttrGroup], list[VertexAttrGroup]] = {
            PsnGroup: [], NrmGroup: [], ClrGroup: [], UVGroup: []
        }
        self.tevConfigs: list[TEVConfig] = []
        self.mats: list[Material] = []
        self.meshes: list[Mesh] = []
        self.mtxGen3D: type[tf.AbsMtxGenerator] = tf.StdMtxGen3D


class MDL0Serializer(SubfileSerializer[BRRES_SER_T, MDL0]):

    DATA_TYPE = MDL0
    FOLDER_NAME = "3DModels(NW4R)"
    MAGIC = b"MDL0"

    MTX_GEN_TYPES_3D = (tf.StdMtxGen3D, tf.XSIMtxGen3D, tf.MayaMtxGen3D)
    MTX_GEN_TYPES_2D = (tf.MayaMtxGen2D, tf.XSIMtxGen2D, tf.MaxMtxGen2D)

    # order in which the sections are stored in files
    _SEC_DATA_ORDER: tuple[type] = (
        TextureLinkWriter, PaletteLinkWriter, Definition, Joint, None, None,
        Material, TEVConfig, Mesh, PsnGroup, NrmGroup, ClrGroup, UVGroup, None
    )
    # the order of their dicts is different from the order in which their data actually appears
    _SEC_DICT_ORDER: tuple[type] = (
        Definition, Joint, PsnGroup, NrmGroup, ClrGroup, UVGroup,
        None, None, Material, TEVConfig, Mesh, TextureLinkWriter, PaletteLinkWriter, None
    )

    _SEC_OFFSET_STRCT = Struct(">15i")
    _HEAD_STRCT = Struct(">IiIIIIiI???BI 3f 3f")
    _SEC_DATA_PAD_AMOUNT = 4 # each section's data is padded to this amount


class MDL0Reader(MDL0Serializer, SubfileReader):

    _SEC_READABLES: dict[type, type[Readable]] = {
        Definition: Definition,
        Joint: JointReader,
        PsnGroup: PsnGroupReader,
        NrmGroup: NrmGroupReader,
        ClrGroup: ClrGroupReader,
        UVGroup: UVGroupReader,
        Material: MaterialReader,
        TEVConfig: TEVConfigReader,
        Mesh: MeshReader
    }

    def __init__(self, parent: BRRESReader = None, offset = 0):
        super().__init__(parent, offset)
        self.deformers: DeformerArrayReader = DeformerArrayReader(self)
        self._sections: dict[type[Readable], dict[str, Readable]] = {}

    def section(self, secType: type[_RBL_T]) -> dict[str, _RBL_T]:
        """Section of this model for some readable type.

        For a section type not present, return an empty dict.
        """
        try:
            return self._sections[secType]
        except KeyError:
            return {}

    def itemName(self, item: Readable):
        """Name of some readable item in its respective section of this model."""
        return getKey(self._sections[type(item)], item)

    def _getSectionInstances(self, secType: type[Reader]):
        """Get a list containing the data instances for a section of this writer's MDL0."""
        return [reader.getInstance() for reader in self.section(secType).values()]

    def unpack(self, data: bytes):
        super().unpack(data)
        secOffsetOffset = self.offset + self._CMN_STRCT.size
        headerOffset = secOffsetOffset + self._SEC_OFFSET_STRCT.size
        deformerOffset = headerOffset + self._HEAD_STRCT.size
        secOffsets = self._SEC_OFFSET_STRCT.unpack_from(data, secOffsetOffset)
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, headerOffset)
        self._data.mtxGen3D = self.MTX_GEN_TYPES_3D[unpackedHeader[2]]
        if self._data.mtxGen3D is tf.XSIMtxGen3D:
            raise ValueError("3D XSI transformations not currently supported")
        # unpack sections & deformers
        for t, secDictOffset in zip(self._SEC_DICT_ORDER, secOffsets):
            if secDictOffset != 0 and t in self._SEC_READABLES:
                dictReader = DictReader(self, self.offset + secDictOffset).unpack(data)
                readableT = self._SEC_READABLES[t]
                self._sections[readableT] = dictReader.readEntries(data, readableT)
        self.deformers = DeformerArrayReader(self, deformerOffset).unpack(data)
        # apply definitions
        meshes: tuple[MeshReader] = tuple(self._sections.get(MeshReader, {}).values())
        for d in self._sections.get(Definition, {}).values():
            for cmd in d.cmds:
                if isinstance(cmd, DefDeformer): # create multi-weight deformer
                    self.deformers.applyDef(cmd)
                elif isinstance(cmd, DefMeshRendering): # set mesh rendering settings
                    meshes[cmd.meshIdx].applyDef(cmd)
        return self

    def _updateInstance(self):
        super()._updateInstance()
        if JointReader in self._sections:
            # set root joint
            joints: list[Joint] = self._getSectionInstances(JointReader)
            rootJoints = [j for j in joints if not j.parent]
            self._data.rootJoint = rootJoints[0]
            if len(rootJoints) > 1:
                raise ValueError("BRRES models cannot have multiple root joints")
            # make joints use unpacked matrices by default (rather than recalculating)
            for jointReader in self._sections[JointReader].values():
                jointReader.updateMatrixCache(self._data.mtxGen3D)
        self._data.tevConfigs = unique(self._getSectionInstances(TEVConfigReader))
        self._data.mats = self._getSectionInstances(MaterialReader)
        self._data.meshes = self._getSectionInstances(MeshReader)
        for groupReader in (PsnGroupReader, NrmGroupReader, ClrGroupReader, UVGroupReader):
            self._data.vertGroups[groupReader.DATA_TYPE] = self._getSectionInstances(groupReader)

class MDL0Writer(MDL0Serializer, SubfileWriter):

    _SEC_WRITABLES: dict[type, type[Writable]] = {
        Definition: Definition,
        Joint: JointWriter,
        PsnGroup: PsnGroupWriter,
        NrmGroup: NrmGroupWriter,
        ClrGroup: ClrGroupWriter,
        UVGroup: UVGroupWriter,
        Material: MaterialWriter,
        TEVConfig: TEVConfigWriter,
        Mesh: MeshWriter,
        TextureLinkWriter: TextureLinkWriter,
        PaletteLinkWriter: PaletteLinkWriter
    }

    def __init__(self, parent: BRRESWriter = None, offset = 0):
        super().__init__(parent, offset)
        self.deformers: DeformerArrayWriter = DeformerArrayWriter(self)
        self._sections: dict[type[Writable], DictWriter[Writable]] = {}
        self._numGeometryDeformers = 0

    @property
    def numGeometryDeformers(self):
        """Number of deformers in this model that are used by geometry."""
        return self._numGeometryDeformers

    def section(self, secType: type[_WBL_T]) -> dict[str, _WBL_T]:
        """Section of this model for some writable type.

        For a section type not present, return an empty dict.
        """
        try:
            return self._sections[secType].getInstance()
        except KeyError:
            return {}

    def itemIdx(self, item: Writable):
        """Index of some writable item in its respective section of this model. -1 if not found."""
        try:
            return unique(self._sections[type(item)].getInstance().values()).index(item)
        except (KeyError, ValueError):
            return -1

    def _generateDefs(self) -> dict[str, Definition]:
        """Generate a dict of definitions for this writer's model."""
        # joint parent & deformer defs
        nodeTree = Definition()
        nodeMix = Definition()
        dfs = self.deformers.getInstance()
        dIdcs = {d.joints[0]: i for i, d in enumerate(dfs) if len(d) == 1} # single-joint df indices
        jointIdcs = {j: i for i, j in enumerate(self._data.rootJoint.deepChildren())}
        for joint, i in jointIdcs.items():
            parentIdx = dIdcs[joint.parent] if joint.parent is not None else 0
            nodeTree.cmds.append(DefJointParent(i, parentIdx))
            for d in dfs: # if joint's in a multi-weight deformer, add a cmd to node mix
                if len(d) > 1 and joint in d:
                    nodeMix.cmds.append(DefDeformerMember(dIdcs[joint], i))
                    break
        for i, d in enumerate(dfs):
            if len(d) > 1: # for multi-weight deformers, add joint weight defs
                nodeMix.cmds.append(DefDeformer(i, {dIdcs[j]: w for j, w in d.items()}))
        # object drawing defs (sorted by render order first, then material index)
        drawOpa = Definition()
        drawXlu = Definition()
        meshIdcs = {m: i for i, m in enumerate(self._data.meshes)}
        matIdcs = {m: i for i, m in enumerate(self._data.mats)}
        meshes = self._data.meshes
        meshes = sorted(meshes, key=lambda m: (m.drawPrio, matIdcs[m.mat]))
        for m in meshes:
            cmd = DefMeshRendering(matIdcs[m.mat], meshIdcs[m], jointIdcs[m.visJoint], m.drawPrio)
            drawDef = drawOpa if m.mat.renderGroup is RenderGroup.OPA else drawXlu
            drawDef.cmds.append(cmd)
        # return brres dict of defs that have commands
        defs = {"NodeTree": nodeTree, "NodeMix": nodeMix, "DrawOpa": drawOpa, "DrawXlu": drawXlu}
        return {k: v for k, v in defs.items() if len(v.cmds) > 0}

    def _generateLinks(self, linkType: type[_LINK_T], texProp: str) -> dict[str, _LINK_T]:
        """Generate a dict of resource links for this writer's model.

        Requires the type of link and the name of the texture attribute for this resource type.
        """
        links: dict[Subfile, _LINK_T] = {} # all links created, one for each texture
        # generate entries, w/ direct mat/tex references at first
        for mat in self._data.mats:
            for tex in mat.textures:
                resName = getattr(tex, texProp)
                if resName is not None:
                    try:
                        link = links[resName]
                    except KeyError:
                        link = links[resName] = linkType(self)
                    link.textures.append(tex)
        return dict(sorted(links.items())) # sort alphabetically

    def _generateSections(self) -> dict[type[Writable], dict[str, Writable]]:
        """Get a dict w/ writables for all the sections of a MDL0.

        Empty sections are excluded from the output dict.
        """
        model = self._data
        secs: dict[type[Writable], dict[str, Writable]] = {}
        # definitions
        secs[Definition] = self._generateDefs()
        # joints
        joints = model.rootJoint.deepChildren()
        secs[JointWriter] = {j.name: JointWriter(self).fromInstance(j) for j in joints}
        # vertex attr groups
        for groupWriterT in (PsnGroupWriter, NrmGroupWriter, ClrGroupWriter, UVGroupWriter):
            groups = model.vertGroups[groupWriterT.DATA_TYPE]
            secs[groupWriterT] = {g.name: groupWriterT(self).fromInstance(g) for g in groups}
        # materials
        secs[MaterialWriter] = {m.name: MaterialWriter(self).fromInstance(m) for m in model.mats}
        # tev configs
        cfs = {t: TEVConfigWriter(self).fromInstance(t) for t in model.tevConfigs}
        secs[TEVConfigWriter] = {m.name: cfs[m.tevConfig] for m in model.mats if m.tevConfig in cfs}
        # meshes
        secs[MeshWriter] = {m.name: MeshWriter(self).fromInstance(m) for m in model.meshes}
        # resource links
        secs[TextureLinkWriter] = self._generateLinks(TextureLinkWriter, "imgName")
        secs[PaletteLinkWriter] = self._generateLinks(PaletteLinkWriter, "pltName")
        # exclude empty sections & return
        return {t: d for t, d in secs.items() if len(d) > 0}

    def getStrings(self):
        return set().union(*(dictWriter.getStrings() for dictWriter in self._sections.values()))

    def _calcSize(self):
        return super()._calcSize()

    def fromInstance(self, data: MDL0):
        super().fromInstance(data)
        offset = self.offset
        offset += self._CMN_STRCT.size + self._SEC_OFFSET_STRCT.size + self._HEAD_STRCT.size
        # generate deformer array
        jointIdcs = {j: i for i, j in enumerate(data.rootJoint.deepChildren())}
        hasGeometry = unique(d for m in data.meshes for d in m.getDeformers())
        self._numGeometryDeformers = len(hasGeometry)
        dfSortInfo = {d: (len(d), jointIdcs[d.joints[0]]) for d in hasGeometry}
        deformers = sorted(hasGeometry, key=lambda d: dfSortInfo[d])
        for joint in jointIdcs: # joints w/o geometry come last
            deformer = joint.deformer
            if deformer not in deformers:
                deformers.append(deformer)
        self.deformers.fromInstance(deformers)
        offset += self.deformers.size()
        # generate sections & calculate offsets
        secs = self._generateSections()
        for t, d in secs.items():
            self._sections[t] = dictWriter = DictWriter(self, offset).fromInstance(d)
            offset += dictWriter.size()
        for sec in keyVals(secs, (self._SEC_WRITABLES.get(t) for t in self._SEC_DATA_ORDER)):
            offsetInDict = 0
            for writer in unique(sec.values()):
                writer.offset = offset + offsetInDict
                offsetInDict += writer.size()
            offset += pad(offsetInDict, self._SEC_DATA_PAD_AMOUNT)
        # store size & return
        self._size = offset - self.offset
        return self

    def pack(self):
        headerOffset = self._CMN_STRCT.size + self._SEC_OFFSET_STRCT.size
        # pack top of header & deformer array
        packedHeader = super().pack()
        packedDeformers = self.deformers.pack()
        # pack sections
        secs = {t: d for t, d in self._sections.items() if len(d.getInstance()) > 0}
        wblsDictOrder = tuple(self._SEC_WRITABLES.get(t) for t in self._SEC_DICT_ORDER)
        wblsDataOrder = tuple(self._SEC_WRITABLES.get(t) for t in self._SEC_DATA_ORDER)
        secsDictOrder = keyVals(secs, wblsDictOrder)
        secsDataOrder = keyVals(secs, wblsDataOrder)
        packedSecs = b"".join(s.pack() for s in secsDictOrder)
        for sec in secsDataOrder:
            packedSec = b"".join(i.pack() for i in unique(sec.getInstance().values()))
            packedSecs += pad(packedSec, self._SEC_DATA_PAD_AMOUNT)
        # pack section offsets & name offset
        allDicts = keyValsDef(secs, wblsDictOrder, None) # dicts w/ absent types as none
        secOffsets = (d.offset - self.offset if d is not None else 0 for d in allDicts)
        nameOffset = self.stringOffset(self._data.name) - self.offset
        packedHeader += self._SEC_OFFSET_STRCT.pack(*secOffsets, nameOffset)
        # pack main header
        numVerts = sum(m.numVerts() for m in self._data.meshes)
        numFaces = sum(m.numFaces() for m in self._data.meshes)
        hasTexMtcs = False
        hasNrms = any(len(m.vertGroups[NrmGroup]) > 0 for m in self._data.meshes)
        hasTexMtcs = any(m.hasTexMtcs() for m in self._data.meshes)
        enableExtents = False # TODO: extents, deformer matrix mode
        minCoord = maxCoord = [0, 0, 0]
        try:
            # for mtxGen2D, get most common texture matrix generator
            # ideally models should just use one generator and then this will be that (retail models
            # are always like this), but in case of weird models just do this (i don't think this
            # property actually does anything)
            mtxGen2D = self.MTX_GEN_TYPES_2D.index(multimode(m.mtxGen for m in self._data.mats)[0])
        except IndexError:
            # model has no materials so just pick 0
            mtxGen2D = 0
        packedHeader += self._HEAD_STRCT.pack(self._HEAD_STRCT.size, -headerOffset,
                                              self.MTX_GEN_TYPES_3D.index(self._data.mtxGen3D),
                                              mtxGen2D, numVerts, numFaces, 0,
                                              self.numGeometryDeformers,
                                              hasNrms, hasTexMtcs, enableExtents,
                                              0, self._HEAD_STRCT.size,
                                              *minCoord, *maxCoord)
        # put it all together
        return packedHeader + packedDeformers + packedSecs
