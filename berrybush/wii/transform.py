# standard imports
from abc import ABC, abstractmethod
# 3rd party imports
import numpy as np


def readonlyView(arr: np.ndarray):
    """Return a readonly view of a numpy array."""
    output = arr.view()
    output.flags.writeable = False
    return output


# default scale, rotation, & translation values
IDENTITY_S = 1
IDENTITY_R = 0
IDENTITY_T = 0


class VectorTransformation():
    """Defines properties for a type of spatial transformation."""

    @classmethod
    @abstractmethod
    def dims(cls, n: int) -> int:
        """Number of dimensions required to apply this transformation in n-dimensional space."""

    @classmethod
    def verify(cls, v: np.ndarray, n: int):
        """Raise a TypeError if v lacks the shape needed for this transformation in n dimensions."""
        if len(v) != cls.dims(n):
            raise TypeError(f"{cls} in {n}D expected {cls.dims(n)} dimensions but got {len(v)}")

    @classmethod
    @abstractmethod
    def mtx(cls, v: np.ndarray) -> np.ndarray:
        """Generate a matrix from a vector for this transformation.
        
        The input array can also be an array of vectors, with any shape.
        A matrix is created for each vector.
        """


class Scaling(VectorTransformation):
    """Scaling in n dimensions."""

    @classmethod
    def dims(cls, n: int):
        return n

    @classmethod
    def mtx(cls, v: np.ndarray):
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        # for each vector, make diagonal matrix w/ vector on diagonal (+ bottom-right entry of 1)
        extraShape, length = v.shape[:-1], v.shape[-1]
        if extraShape:
            mtcs = np.tile(np.identity(length + 1), (*extraShape, 1, 1))
            mtcs[..., :-1, :-1] = np.apply_along_axis(np.diagflat, -1, v)
            return mtcs
        elif length != 2:
            # this does the same thing as the code above, but optimized for the case of 1 vector
            # (since apply_along_axis is kinda slow, even for this case)
            return np.diagflat(np.append(v, 1))
        else:
            # we have one last optimization for 2d for the sake of fast srt0 animation evaluation
            # (diagflat is pretty fast, but this is faster)
            return np.array((
                (v[0], 0,    0),
                (0,    v[1], 0),
                (0,    0,    1)
            ))

    @classmethod
    def fromMtx(cls, mtx: np.ndarray) -> np.ndarray:
        """Return the scale of a SRT transformation matrix."""
        mtx = mtx[:-1, :-1]
        scale = np.linalg.norm(mtx, axis=1)
        if np.linalg.det(mtx) < 0:
            scale[0] *= -1
        return scale


class Rotation(VectorTransformation):
    """Rotation in n dimensions, defined using Euler angles in NW4R conventions."""

    @classmethod
    def dims(cls, n: int):
        return n * (n - 1) // 2

    @classmethod
    def mtx(cls, v: np.ndarray):
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        extraShape, length = v.shape[:-1], v.shape[-1]
        if length not in {1, 3}:
            raise TypeError("Rotation matrices only supported for 2D and 3D")
        v = -np.deg2rad(v)
        sin = np.sin(v)
        cos = np.cos(v)
        if length == 1 and not extraShape:
            # optimization for simple 2d rotations, which have high priority
            # since they can be animated in berrybush (srt0 animations)
            cos = cos.item()
            sin = sin.item()
            return np.array((
                ( cos, sin, 0),
                (-sin, cos, 0),
                ( 0,   0,   1)
            ))
        identities = np.tile(np.identity(4, v.dtype), (*extraShape, 1, 1))
        xMtcs = identities.copy()
        xMtcs[..., 1, 1] = cos[..., 0]
        xMtcs[..., 2, 1] = -sin[..., 0]
        xMtcs[..., 1, 2] = sin[..., 0]
        xMtcs[..., 2, 2] = cos[..., 0]
        if length == 1: # 2d
            return xMtcs[..., 1:, 1:]
        yMtcs = identities.copy()
        yMtcs[..., 0, 0] = cos[..., 1]
        yMtcs[..., 2, 0] = sin[..., 1]
        yMtcs[..., 0, 2] = -sin[..., 1]
        yMtcs[..., 2, 2] = cos[..., 1]
        zMtcs = identities.copy()
        zMtcs[..., 0, 0] = cos[..., 2]
        zMtcs[..., 1, 0] = -sin[..., 2]
        zMtcs[..., 0, 1] = sin[..., 2]
        zMtcs[..., 1, 1] = cos[..., 2]
        return zMtcs @ yMtcs @ xMtcs

    @classmethod
    def extractMtx(cls, mtx: np.ndarray):
        """Return the rotation matrix part of a SRT transformation matrix."""
        with np.errstate(divide="ignore", invalid="ignore"): # suppress 0 division warnings
            return np.nan_to_num(mtx[:-1, :-1] / Scaling.fromMtx(mtx).reshape(-1, 1))


class Translation(VectorTransformation):
    """Translation in n dimensions."""

    @classmethod
    def dims(cls, n: int):
        return n

    @classmethod
    def mtx(cls, v: np.ndarray):
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        extraShape, length = v.shape[:-1], v.shape[-1]
        if length == 2 and not extraShape:
            # optimization for simple 2d translations, which have high priority
            # since they can be animated in berrybush (srt0 animations)
            return np.array((
                (1, 0, v[0]),
                (0, 1, v[1]),
                (0, 0, 1)
            ))
        mtcs = np.tile(np.identity(length + 1), (*extraShape, 1, 1))
        mtcs[..., :length, length] = v
        return mtcs

    @classmethod
    def fromMtx(cls, mtx: np.ndarray) -> np.ndarray:
        """Return the translation of a SRT transformation matrix."""
        return mtx[:-1, -1].flatten()


def decompose(mtx: np.ndarray):
    """Decompose an n-dimensional matrix into its SRT components. (4x4 for 3D, 3x3 for 2D)"""
    n = len(mtx) - 1
    if n not in {2, 3}:
        raise TypeError("Decomposing only supported for 2D and 3D")
    scale = Scaling.fromMtx(mtx)
    trans = Translation.fromMtx(mtx)
    rotMtx = Rotation.extractMtx(mtx)
    rot = decompose2DRotation(rotMtx) if n == 2 else decompose3DRotation(rotMtx)
    return Transformation(n, scale, rot, trans)


def decompose3DRotation(mtx: np.ndarray) -> np.ndarray:
    """Get a set of Euler angles in NW4R conventions from a 3x3 3D rotation matrix.
        
    The input can also be an array of matrices, with any shape.
    A vector is created for each matrix.
    """
    # based on https://learnopencv.com/rotation-matrix-to-euler-angles/
    sy = np.sqrt(mtx[..., 0, 0] * mtx[..., 0, 0] + mtx[..., 1, 0] * mtx[..., 1, 0])
    singular = sy < 1e-6
    notSingular = np.invert(singular)
    decomposed = np.zeros((*mtx.shape[:-2], 3))
    decomposed[singular, 0] = np.arctan2(-mtx[..., 1, 2], mtx[..., 1, 1])[singular]
    decomposed[singular, 1] = np.arctan2(-mtx[..., 2, 0], sy)[singular]
    decomposed[notSingular, 0] = np.arctan2(mtx[..., 2, 1], mtx[..., 2, 2])[notSingular]
    decomposed[notSingular, 1] = np.arctan2(-mtx[..., 2, 0], sy)[notSingular]
    decomposed[notSingular, 2] = np.arctan2(mtx[..., 1, 0], mtx[..., 0, 0])[notSingular]
    return np.rad2deg(decomposed)


def decompose2DRotation(mtx: np.ndarray) -> np.ndarray:
    """Get an angle from a 2x2 2D rotation matrix."""
    return -np.rad2deg(np.arctan2(mtx[0, 1], mtx[0, 0]).reshape(1))


class Transformation():
    """Set of scale, rotation, and translation vectors in n-dimensional space."""

    def __init__(self, n: int, s: np.ndarray = None, r: np.ndarray = None, t: np.ndarray = None):
        self._ndims = n
        s = s if s is not None else np.repeat(IDENTITY_S, Scaling.dims(n))
        r = r if r is not None else np.repeat(IDENTITY_R, Rotation.dims(n))
        t = t if t is not None else np.repeat(IDENTITY_T, Translation.dims(n))
        self.set(s, r, t)

    @property
    def s(self):
        return self._sv

    @property
    def r(self):
        return self._rv

    @property
    def t(self):
        return self._tv

    @property
    def homoS(self):
        """True if this transformation's scale is homogenous (same value for every axis)."""
        return np.allclose(self._s, self._s[0])

    @property
    def homoR(self):
        """True if this transformation's rotation is homogenous (same value for every axis)."""
        return np.allclose(self._r, self._r[0])

    @property
    def homoT(self):
        """True if this transformation's translation is homogenous (same value for every axis)."""
        return np.allclose(self._t, self._t[0])

    @property
    def identityS(self):
        """True if this transformation's scale is the default value."""
        return np.allclose(self._s, IDENTITY_S)

    @property
    def identityR(self):
        """True if this transformation's rotation is the default value."""
        return np.allclose(self._r, IDENTITY_R)

    @property
    def identityT(self):
        """True if this transformation's translation is the default value."""
        return np.allclose(self._t, IDENTITY_T)

    @property
    def ndims(self):
        return self._ndims

    def set(self, s: np.ndarray = None, r: np.ndarray = None, t: np.ndarray = None):
        """Update any of this transformation's vectors with new values."""
        if s is not None:
            self._s = s if isinstance(s, np.ndarray) else np.array(s)
            self._sv = readonlyView(self._s)
            Scaling.verify(self._s, self._ndims)
        if r is not None:
            self._r = r if isinstance(r, np.ndarray) else np.array(r)
            self._rv = readonlyView(self._r)
            Rotation.verify(self._r, self._ndims)
        if t is not None:
            self._t = t if isinstance(t, np.ndarray) else np.array(t)
            self._tv = readonlyView(self._t)
            Translation.verify(self._t, self._ndims)

    def __eq__(self, other: "Transformation"):
        return (
            isinstance(other, Transformation)
            and np.array_equal(self._s, other._s)
            and np.array_equal(self._r, other._r)
            and np.array_equal(self._t, other._t)
        )


class MtxGenerator(ABC):
    """Defines a way to generate a matrix from some set of transformation vectors."""

    @classmethod
    @abstractmethod
    def genMtx(cls, srt: Transformation) -> np.ndarray:
        """Generate a matrix for a transformation."""


class AbsMtxGenerator(MtxGenerator):
    """Matrix generator that lets you make absolute matrices for parented transforms."""

    @classmethod
    @abstractmethod
    def absMtx(cls, srts: list[tuple[Transformation, bool]]) -> np.ndarray:
        """Generate an absolute matrix for some series of transformations using this generator.

        Each transformation is associated with a bool that indicates whether to apply segment
        scale compensation for that transformation.

        The transformations should be organized such that the first is the parent of the second,
        which is the parent of the third, etc. (Whatever "parent" means is dictated by the
        generator itself)
        """


class StdMtxGen3D(AbsMtxGenerator):

    @classmethod
    def genMtx(cls, srt: Transformation):
        return Translation.mtx(srt.t) @ Rotation.mtx(srt.r) @ Scaling.mtx(srt.s)

    @classmethod
    def absMtx(cls, srts: list[tuple[Transformation, bool]]):
        mtx = np.identity(4)
        parent = None
        for srt, segScaleComp in srts:
            if segScaleComp and parent is not None:
                cmp = np.linalg.inv(Scaling.mtx(parent.s)) # parent compensation matrix
                mtx = mtx @ Translation.mtx(srt.t) @ cmp @ Rotation.mtx(srt.r) @ Scaling.mtx(srt.s)
            else:
                mtx = mtx @ cls.genMtx(srt)
            parent = srt
        return mtx


# as far as i can tell, maya is exactly the same as standard? both work w/ segment scale compensate
class MayaMtxGen3D(StdMtxGen3D):
    pass


class XSIMtxGen3D(AbsMtxGenerator):

    @classmethod
    def genMtx(cls, srt: Transformation):
        raise NotImplementedError("3D XSI transformations not currently supported")

    @classmethod
    def absMtx(cls, srts: list[tuple[Transformation, bool]]):
        raise NotImplementedError("3D XSI transformations not currently supported")


# notes about conventions described in comments:

# everything is based on how the texture appears
# "top-left clockwise" means texture rotates clockwise about its top-left corner
# "left/up are +" means as you increase x/y translation, the texture moves left and up
# "top-left" for scale means the texture scales from its top-left corner

# "default" refers to how the texture appears w/ +y as up axis for positive transformations
# (this is also just the convention for indirect transforms)


class MayaMtxGen2D(MtxGenerator):

    _TRANS_COMPENSATE_1 = Translation.mtx((0, 1))
    _TRANS_COMPENSATE_2 = Translation.mtx((0, -1))
    _TRANS_COMPENSATE_3 = Translation.mtx((.5, .5))
    _TRANS_COMPENSATE_4 = Translation.mtx((-.5, -.5))

    @classmethod
    def genMtx(cls, srt: Transformation):
        return (
            np.identity(3)
            # scale (default top-left, maya bottom-left)
            @ cls._TRANS_COMPENSATE_1
            @ Scaling.mtx(srt.s)
            @ cls._TRANS_COMPENSATE_2
            # translation (default left/up are +, maya right/up are +)
            @ Translation.mtx([-srt.t[0], srt.t[1]])
            # rotation (default top-left counterclockwise, maya center clockwise)
            @ cls._TRANS_COMPENSATE_3
            @ Rotation.mtx(-srt.r)
            @ cls._TRANS_COMPENSATE_4
        )


class XSIMtxGen2D(MtxGenerator):

    _TRANS_COMPENSATE_1 = Translation.mtx((0, 1))
    _TRANS_COMPENSATE_2 = Translation.mtx((0, -1))

    @classmethod
    def genMtx(cls, srt: Transformation):
        return (
            np.identity(3)
            # scale (default top-left, xsi bottom-left)
            @ cls._TRANS_COMPENSATE_1
            @ Scaling.mtx(srt.s)
            @ cls._TRANS_COMPENSATE_2
            # rotation (default top-left counterclockwise, xsi bottom-left counterclockwise)
            @ cls._TRANS_COMPENSATE_1
            @ Rotation.mtx(srt.r)
            @ cls._TRANS_COMPENSATE_2
            # translation (default left/up are +, xsi right/up are +)
            @ Translation.mtx([-srt.t[0], srt.t[1]])
        )


class MaxMtxGen2D(MtxGenerator):

    _TRANS_COMPENSATE_1 = Translation.mtx((.5, .5))
    _TRANS_COMPENSATE_2 = Translation.mtx((-.5, -.5))

    @classmethod
    def genMtx(cls, srt: Transformation):
        return (
            np.identity(3)
            # scale (default top-left, 3ds max center)
            @ cls._TRANS_COMPENSATE_1
            @ Scaling.mtx(srt.s)
            @ cls._TRANS_COMPENSATE_2
            # rotation (default top-left counterclockwise, 3ds max center clockwise)
            @ cls._TRANS_COMPENSATE_1
            @ Rotation.mtx(-srt.r)
            @ cls._TRANS_COMPENSATE_2
            # translation (default left/up are +, 3ds max right/up are +)
            @ Translation.mtx([-srt.t[0], srt.t[1]])
        )


class IndMtxGen2D(MtxGenerator):

    @classmethod
    def genMtx(cls, srt: Transformation):
        return Translation.mtx(srt.t) @ Rotation.mtx(srt.r) @ Scaling.mtx(srt.s)
