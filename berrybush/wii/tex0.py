# standard imports
from abc import abstractmethod
from struct import Struct
# 3rd party imports
import numpy as np
# internal imports
from . import dxt1lookups
from .gx import ColorFormat
from .subfile import BRRES_SER_T, Subfile, SubfileSerializer, SubfileReader, SubfileWriter


def blockSplit(px: np.ndarray, blkDims: tuple):
    """Split 3D (y, x, color channels) image data into blocks of the specified dimensions."""
    # https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    chans = px.shape[2]
    dims = (px.shape[1], px.shape[0])
    px = px.reshape(dims[1] // blkDims[1], blkDims[1], -1, blkDims[0], chans)
    px = px.swapaxes(1,2)
    return px.reshape(-1, blkDims[1], blkDims[0], chans)


def unBlockSplit(px: np.ndarray, dims: tuple):
    """Unsplit 3D (y, x, color channels) image data from blocks into the specified dimensions."""
    # https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    chans = px.shape[3]
    blkDims = (px.shape[2], px.shape[1])
    px = px.reshape(dims[1] // blkDims[1], -1, blkDims[1], blkDims[0], chans)
    px = px.swapaxes(1,2)
    return px.reshape(*dims[::-1], chans)


class ImageFormat():
    """Wii image format for encoding & decoding image data stored in blocks."""

    _BLK_DIMS: np.ndarray # block dims in px

    @classmethod
    def roundDims(cls, dims: np.ndarray):
        """Round image dimensions based on this format's block size."""
        return np.ceil(dims / cls._BLK_DIMS).astype(np.integer) * cls._BLK_DIMS

    @classmethod
    @abstractmethod
    def imgSize(cls, dims: np.ndarray) -> int:
        """Get the size of an image in bytes based on its dimensions."""

    @classmethod
    @abstractmethod
    def importImg(cls, data: bytes, dims: np.ndarray) -> np.ndarray:
        """Import an array of pixels (top->bottom, left->right) from bytes."""

    @classmethod
    @abstractmethod
    def exportImg(cls, px: np.ndarray) -> bytes:
        """Export an array of pixels (top->bottom, left->right) to bytes."""

    @classmethod
    def adjustImg(cls, px: np.ndarray) -> np.ndarray:
        """Adjust an image based on this format by exporting & re-importing it.

        (e.g., convert to grayscale for IA8, or perform block compression for CMPR)"""
        return cls.importImg(cls.exportImg(px), px.shape[:2][::-1])


class RawImageFormat(ImageFormat):
    """Image format that stores uncompressed data."""

    _PX_FMT: ColorFormat # format for color channels

    @classmethod
    def imgSize(cls, dims: np.ndarray):
        # image must first be resized so blocks fit w/o overflow
        return cls.roundDims(dims).prod() * cls._PX_FMT.stride

    @classmethod
    def importImg(cls, data: bytes, dims: np.ndarray):
        roundedDims = cls.roundDims(dims)
        numBlocks = np.prod(roundedDims / cls._BLK_DIMS, dtype=np.integer)
        px = cls._PX_FMT.unpack(data)
        # unflatten to blocks and do any format-specific work before import
        px = px.reshape(numBlocks, *cls._BLK_DIMS[::-1], -1)
        px = cls._prepForImport(px)
        # add block shape back in case it was removed, then remove it and return
        px = px.reshape(-1, *cls._BLK_DIMS[::-1], px.shape[-1])
        px = unBlockSplit(px, roundedDims)
        return px[:dims[1], :dims[0]]

    @classmethod
    def exportImg(cls, px: np.ndarray):
        dims = np.array(px.shape[:2])[::-1]
        # pad image dimensions to multiples of block dimensions using 0s
        padAmounts = cls.roundDims(dims) - dims
        px = np.pad(px, ((0, padAmounts[1]), (0, padAmounts[0]), (0, 0)))
        # split into blocks, do format-specific work, and flatten & pack
        px = blockSplit(px, cls._BLK_DIMS)
        px = cls._prepForExport(px)
        return cls._PX_FMT.pack(px.reshape(-1, cls._PX_FMT.nchans))

    @classmethod
    @abstractmethod
    def _prepForImport(cls, px: np.ndarray):
        """Take an array of image data separated into blocks and prepare it for import.

        This method is called by the main import method after data has been put into a 1D array of
        2D blocks of pixels (each with channels based on this format, making for 4 total
        dimensions). After the work done here, image data is taken out of block form and returned.
        It's immediately reshaped, so this method can mess with the shape.

        The default implementation normalizes each color's values based on the # of bits given to
        each channel by this format. After calling it (if necessary), subclasses should implement
        their means of converting the image data from their format into the unpacked format
        (whatever that may be - in most cases, rgba).
        """
        return cls._PX_FMT.normalize(px)


    @classmethod
    @abstractmethod
    def _prepForExport(cls, px: np.ndarray):
        """Take an array of image data separated into blocks and prepare it for export.

        This method is called by the main export method after data has been put into a 1D array of
        2D blocks of pixels (each with split into channels, making 4 total dimensions). After the
        work done here, image data is flattened from the blocks and packed to bytes. Because it's
        immediately flattened, this method can mess with the shape.

        The default implementation denormalizes each color's values based on the # of bits given to
        each channel by this format. Before calling it (if necessary), subclasses should implement
        their means of converting the image data from the unpacked format (whatever that may be -
        in most cases, rgba) into their packed format.
        """
        return cls._PX_FMT.denormalize(px.reshape(-1, cls._PX_FMT.nchans))


class ListableImageFormat(RawImageFormat):
    """Raw image format that can also be stored in the form of a color list."""

    @classmethod
    def listSize(cls, length: np.ndarray):
        """Get the size of a color list based on its length."""
        return length * cls._PX_FMT.stride

    @classmethod
    def importList(cls, data: bytes, length = -1):
        return cls._prepForImport(cls._PX_FMT.unpack(data, length))

    @classmethod
    def exportList(cls, px: np.ndarray):
        return cls._PX_FMT.pack(cls._prepForExport(px))

    @classmethod
    @abstractmethod
    def _prepForImport(cls, px: np.ndarray):
        """Take an image data array (any shape - maybe list, image, etc.) & prepare it for import.

        The default implementation normalizes each color's values based on the # of bits given to
        each channel by this format. After calling it (if necessary), subclasses should implement
        their means of converting the image data from their format into the unpacked format
        (whatever that may be - in most cases, rgba).
        """
        return super()._prepForImport(px)

    @classmethod
    @abstractmethod
    def _prepForExport(cls, px: np.ndarray):
        """Take an image data array (any shape - maybe list, image, etc.) & prepare it for export.

        The default implementation denormalizes each color's values based on the # of bits given to
        each channel by this format. Before calling it (if necessary), subclasses should implement
        their means of converting the image data from the unpacked format (whatever that may be -
        in most cases, rgba) into their packed format.
        """
        return super()._prepForExport(px)


def grayscale(px: np.ndarray):
    """Return an array of image data converted to grayscale.

    (Output has input shape w/o last dimension, filled with grayscale values)"""
    return 0.2989 * px[..., 0] + 0.5870 * px[..., 1] + 0.1140 * px[..., 2]


class I4(RawImageFormat):
    _BLK_DIMS = np.array((8, 8))
    # color format stride in bytes must be an int, so treat this format like a 2-channel format and
    # do some hacks in prepForImport and imgSize to adjust for this
    _PX_FMT = ColorFormat(4, 4)

    @classmethod
    def imgSize(cls, dims: np.ndarray):
        return super().imgSize(dims) // 2 # adjust for pixel format hack

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        px = super()._prepForImport(px.reshape(-1, cls._PX_FMT.nchans)).reshape(-1, 1)
        return (px[..., (0, 0, 0, 0)])

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        # convert to grayscale & normalize
        return super()._prepForExport(grayscale(px))


class I8(RawImageFormat):
    _BLK_DIMS = np.array((8, 4))
    _PX_FMT = ColorFormat(8)

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        px = super()._prepForImport(px)
        return (px[..., (0, 0, 0, 0)])

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        # convert to grayscale & normalize
        return super()._prepForExport(grayscale(px))


class IA4(RawImageFormat):
    _BLK_DIMS = np.array((8, 4))
    _PX_FMT = ColorFormat(4, 4)

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        px = super()._prepForImport(px)
        return (px[..., (1, 1, 1, 0)])

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        # convert to grayscale & normalize
        return super()._prepForExport(np.stack((px[..., 3], grayscale(px)), -1))


class IA8(ListableImageFormat):
    _BLK_DIMS = np.array((4, 4))
    _PX_FMT = ColorFormat(8, 8)

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        px = super()._prepForImport(px)
        return (px[..., (1, 1, 1, 0)])

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        # convert to grayscale & normalize
        return super()._prepForExport(np.stack((px[..., 3], grayscale(px)), -1))


class RGB565(ListableImageFormat):
    _BLK_DIMS = np.array((4, 4))
    _PX_FMT = ColorFormat(5, 6, 5)

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        px = super()._prepForImport(px)
        return np.concatenate((px, np.ones((*px.shape[:-1], 1))), -1)

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        # trim off alpha channel & normalize
        return super()._prepForExport(px[..., :3])


class RGB5A3(ListableImageFormat):
    _BLK_DIMS = np.array((4, 4))
    _PX_FMT = ColorFormat(16)

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        px = px.astype(int)
        rgb = px[..., 0] >> 15 == 1
        argb = np.invert(rgb)
        output = np.empty((*px.shape[:-1], 4))
        output[rgb, :3] = (px[rgb] >> np.array([[10, 5, 0]]) & 0b11111) / 0b11111
        output[rgb, 3] = 1
        output[argb, :3] = (px[argb] >> np.array([[8, 4, 0]]) & 0b1111) / 0b1111
        # alpha max is (max 3 bit value) + 1 because if alpha is 1, it's just disabled
        output[argb, 3] = (px[argb, 0] >> 12 & 0b111) / 0b1000
        return output

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        rgb = px[..., 3] == 1
        argb = np.invert(rgb)
        # alpha max is (max 3 bit value) + 1 because if alpha is 1, it's just disabled
        px[..., 3] *= 0b1000
        px[argb, :3] *= 0b1111
        px[rgb, :3] *= 0b11111
        px = np.round(px).astype(int)
        output = np.zeros(px.shape[:-1], px.dtype) # px w/ each color as one 16 bit int
        output[rgb] = np.bitwise_or.reduce(px[rgb, :3] << np.array([[10, 5, 0]]), 1) | (1 << 15)
        output[argb] = np.bitwise_or.reduce(px[argb, :] << np.array([[8, 4, 0, 12]]), 1)
        return output.reshape(*output.shape, 1)


class RGBA8(RawImageFormat):
    _BLK_DIMS = np.array((4, 4))
    _PX_FMT = ColorFormat(8, 8, 8, 8)

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        # normalize
        px = super()._prepForImport(px)
        # for rgba, values are rearranged a bit within blocks for some reason
        numChans = 4
        blkSize = cls._BLK_DIMS.prod() * numChans # block size, in total color channel values
        px = px.reshape(-1, blkSize)
        midwayIdx = blkSize // 2
        copy = px.copy()
        px[:, 3::numChans] = copy[:, 0:midwayIdx:2] # a
        px[:, 0::numChans] = copy[:, 1:midwayIdx:2] # r
        px[:, 1::numChans] = copy[:, midwayIdx+0::2] # g
        px[:, 2::numChans] = copy[:, midwayIdx+1::2] # b
        return px.reshape(-1, numChans)

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        # for rgba, values are rearranged a bit within blocks for some reason
        numChans = 4
        blkSize = cls._BLK_DIMS.prod() * numChans # block size, in total color channel values
        px = px.reshape(-1, blkSize)
        midwayIdx = blkSize // 2
        copy = px.copy()
        px[:, 0:midwayIdx:2] = copy[:, 3::numChans] # a
        px[:, 1:midwayIdx:2] = copy[:, 0::numChans] # r
        px[:, midwayIdx+0::2] = copy[:, 1::numChans] # g
        px[:, midwayIdx+1::2] = copy[:, 2::numChans] # b
        # normalize
        return super()._prepForExport(px)


class PaletteImageFormat(RawImageFormat):
    """Image format that stores indices into a palette."""

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        return px.astype(f"u{cls._PX_FMT.stride}")

    @classmethod
    def _prepForExport(cls, px: np.ndarray):
        return px.astype(f"u{cls._PX_FMT.stride}")


class C4(PaletteImageFormat):
    _BLK_DIMS = np.array((8, 8))
    _PX_FMT = ColorFormat(4, 4) # this uses the same dirty hack as i4 for unpacking/packing pixels

    @classmethod
    def imgSize(cls, dims: np.ndarray):
        return super().imgSize(dims) // 2

    @classmethod
    def _prepForImport(cls, px: np.ndarray):
        return super()._prepForImport(px).reshape(-1, *cls._BLK_DIMS[::-1], 1)


class C8(PaletteImageFormat):
    _BLK_DIMS = np.array((8, 4))
    _PX_FMT = ColorFormat(8)


class C14X2(PaletteImageFormat):
    _BLK_DIMS = np.array((4, 4))
    _PX_FMT = ColorFormat(16)


class CMPR(ImageFormat):

    _BLK_DIMS = np.array((8, 8))
    _SUB_DIMS = np.array((4, 4))
    _PLT_FMT = ColorFormat(5, 6, 5)
    _IDX_FMT = ColorFormat(*((2, ) * 16))

    @classmethod
    def _batchCov(cls, arr: np.ndarray):
        n = arr.shape[2]
        m1 = arr - arr.sum(2, keepdims=True) / n
        return np.einsum("ijk,ilk->ijl", m1, m1) / (n - 1)

    @classmethod
    def imgSize(cls, dims: np.ndarray):
        return cls.roundDims(dims).prod() // 2

    @classmethod
    def importImg(cls, data: bytes, dims: tuple[int, int]):
        realDims = dims
        dims = cls.roundDims(dims)
        # process image data
        subs = np.frombuffer(data, ">u2").reshape(-1, 2, 2)
        plts = cls._PLT_FMT.unpack(np.ascontiguousarray(subs[:, 0]).tobytes()).reshape(-1, 2, 3)
        idcs = cls._IDX_FMT.unpack(np.ascontiguousarray(subs[:, 1]).tobytes()).reshape(-1, 16)
        plts = cls._PLT_FMT.normalize(plts)
        pxPlts = np.broadcast_to(plts, (16, *plts.shape)).swapaxes(0, 1) # plts view w/ entry per px
        subPx = np.empty((len(subs), 16, 4), float)
        # first, handle actual indices (not interpolation/transparent)
        isPltIdx = idcs < 2
        notPltIdx = np.logical_not(isPltIdx)
        subPx[isPltIdx, :3] = pxPlts[isPltIdx, idcs[isPltIdx]]
        # then, interpolation by thirds (1/3 for index 2, 2/3 for 3)
        subsGt = np.broadcast_to(subs[:, 0, 0] > subs[:, 0, 1], (16, subs.shape[0])).swapaxes(0, 1)
        is3Itp = np.logical_and(notPltIdx, subsGt)
        pxPlts3Itp = pxPlts[is3Itp]
        pxFacs3Itp = np.expand_dims(idcs[is3Itp] - 1, -1) / 3
        subPx[is3Itp, :3] = pxPlts3Itp[:, 0] + (pxPlts3Itp[:, 1] - pxPlts3Itp[:, 0]) * pxFacs3Itp
        # then, half interpolation (1/2 for both index 2 and 3; 3 is just transparent)
        is2Itp = np.logical_and(notPltIdx, np.logical_not(subsGt))
        pxPlts2Itp = pxPlts[is2Itp]
        subPx[is2Itp, :3] = pxPlts2Itp[:, 0] + (pxPlts2Itp[:, 1] - pxPlts2Itp[:, 0]) / 2
        # set alpha (1 unless we have half interpolation and indices are 3; in that case, 0)
        subPx[:, :, 3] = np.logical_not(np.logical_and(is2Itp, idcs == 3))
        # now we have all the pixel colors, just have to rearrange from sub shape then block shape
        blkDims = cls._BLK_DIMS
        subDims = cls._SUB_DIMS
        numSubs = blkDims // subDims # subs per block x/y
        nchans = subPx.shape[-1]
        px = subPx.reshape(-1, *subDims[::-1], nchans)
        px = px.reshape(-1, numSubs[1], numSubs[0], subDims[1], subDims[0], nchans).swapaxes(2, 3)
        px = px.reshape(-1, *blkDims[::-1], nchans)
        px = unBlockSplit(px, dims) # after stacking subs to make full blocks, remove block shape
        # finally, crop dims & return
        return px[:realDims[1], :realDims[0]]

    @classmethod
    def exportImg(cls, px: np.ndarray):
        original = px
        realDims = (px.shape[1], px.shape[0])
        dims = cls.roundDims(realDims)
        blkDims = cls._BLK_DIMS
        subDims = cls._SUB_DIMS
        # pad image dimensions to multiples of block dimensions using edge values
        padAmounts = dims - realDims
        px = np.pad(px, ((0, padAmounts[1]), (0, padAmounts[0]), (0, 0)), mode="edge")
        # split into tex0 blocks, then sub-blocks
        numSubs = blkDims // subDims # subs per block x/y
        nchans = 4
        px = blockSplit(px, blkDims) # we now have block formation, but need sub formation
        px = px.reshape(-1, numSubs[1], subDims[1], numSubs[0], subDims[0], nchans).swapaxes(2, 3)
        px = px.reshape(-1, np.prod(subDims), nchans)
        # dxt1 compression
        # this is a basic implementation of the "range fit" technique described here:
        # https://www.sjbrown.co.uk/posts/dxt-compression-techniques/
        # and implemented here:
        # https://github.com/castano/nvidia-texture-tools/blob/master/src/nvtt/QuickCompressDXT.cpp
        # the "cluster fit" technique is more accurate but just way too slow for python
        # just for reference, cluster fit is described in greater detail here:
        # https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/dxtc/doc/cuda_dxtc.pdf
        # and implemented here:
        # https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/3_Imaging/dxtc/dxtc.cu
        # note: there are ways to improve range fit which i haven't implemented. might eventually
        totalSubs = len(px)
        rgb = px[:, :, :3]
        a = px[:, :, 3]
        transparent = a < .5
        hasAlpha = np.any(transparent, axis=1)
        isOpaque = np.logical_not(hasAlpha)
        # first, handle single-color blocks
        # these are compressed using a lookup table with optimal palettes for each possible color
        # (since palettes are rgb565, but the interpolated values have 8-bit precision)
        # since values are interpolated based on either halves or thirds, there are two such tables
        isSgl = np.all(np.all(px == px[:, :1], axis=2), axis=1)
        notSgl = np.logical_not(isSgl)
        sglA = hasAlpha[isSgl]
        sglO = isOpaque[isSgl]
        bounds = np.empty((2, totalSubs, 1, 3))
        boundsSgl = np.empty((2, np.count_nonzero(isSgl), 1, 3))
        sglLookupIdcs = np.round(px[isSgl, 0, :3] * 255).astype(np.uint8)[:, :, None]
        boundsSglA = np.take_along_axis(dxt1lookups.PLT_3, sglLookupIdcs[sglA], axis=0)
        boundsSglO = np.take_along_axis(dxt1lookups.PLT_4, sglLookupIdcs[sglO], axis=0)
        boundsSgl[:, sglA, 0] = boundsSglA.transpose(2, 0, 1)
        boundsSgl[:, sglO, 0] = boundsSglO.transpose(2, 0, 1)
        bounds[:, isSgl] = boundsSgl
        rgbns = rgb[notSgl]
        # now for everything else: get best fit lines for each sub & get extrema on them
        try:
            bestFits = np.linalg.svd((rgbns - np.mean(rgbns, axis=1).reshape(-1, 1, 3)))[2][:, 0]
        except (SystemError, ValueError) as e:
            # this happens rarely for large images because "dgesdd fails to init"
            # just raise this error for now
            raise RuntimeError("Not enough memory for CMPR compression. "
                               "Please try again or consider using smaller images") from e
        projections = np.einsum("ijk,ik->ij", rgbns, bestFits)
        # alternative covariant method
        # cov = cls._batchCov(rgbns.swapaxes(1, 2))
        # eigvals, eigvecs = np.linalg.eig(cov)
        # dominantEigvalIdcs = np.argmax(eigvals, axis=1)[:, None, None]
        # bestFits = np.take_along_axis(eigvecs, dominantEigvalIdcs, 1).squeeze(1)
        boundIdcs = np.array((np.argmax(projections, axis=1), np.argmin(projections, axis=1)))
        boundsNSgl = np.take_along_axis(rgbns[None], boundIdcs[:, :, None, None], 2)
        # adjust bounds for rgb565 limitations
        boundsNSgl.clip(0, 1, out=boundsNSgl)
        denormNSgl = cls._PLT_FMT.denormalize(boundsNSgl.swapaxes(0, 1).reshape(-1, 3)).round()
        boundsNSgl[:] = cls._PLT_FMT.normalize(denormNSgl).reshape(-1, 2, 1, 3).swapaxes(0, 1)
        # get palettes & ensure colors are ordered properly to indicate whether alpha enabled
        bounds[:, notSgl] = cls._PLT_FMT.normalize(denormNSgl).reshape(-1, 2, 1, 3).swapaxes(0, 1)
        denorm = cls._PLT_FMT.denormalize(bounds.swapaxes(0, 1).reshape(-1, 3))
        packedPltVals = np.fromstring(cls._PLT_FMT.pack(denorm), ">u2").reshape(-1, 2)
        boundsO = bounds[:, isOpaque]
        plts = np.empty((4, *bounds.shape[1:]))
        plts[:2] = bounds
        plts3 = plts.copy()
        plts3[2:] = (bounds[0] + bounds[1]) / 2
        plts[2:, isOpaque] = ((2 * boundsO[0] + boundsO[1]) / 3, (boundsO[0] + 2 * boundsO[1]) / 3)
        plts[2:, hasAlpha] = plts3[2:, hasAlpha]
        needsSwap = np.zeros(len(packedPltVals), dtype=bool)
        needsSwap[isOpaque] = packedPltVals[isOpaque, 0] <= packedPltVals[isOpaque, 1]
        needsSwap[hasAlpha] = packedPltVals[hasAlpha, 0] > packedPltVals[hasAlpha, 1]
        plts[:, needsSwap] = plts[:, needsSwap][(1, 0, 3, 2),] # swap indices 0/1 and 2/3
        packedPltVals[needsSwap] = packedPltVals[needsSwap, ::-1]
        # for opaque subs, check if 3-interpolation would be better than 4-interpolation
        plts3[:2] = plts[:2]
        opaqueNSgl = np.logical_and(isOpaque, notSgl)
        difs4 = rgb[opaqueNSgl] - plts[:, opaqueNSgl]
        difs3 = rgb[opaqueNSgl] - plts3[:, opaqueNSgl]
        errors4 = np.einsum("ijkl,ijkl->ijk", difs4, difs4)
        errors3 = np.einsum("ijkl,ijkl->ijk", difs3, difs3)
        shouldUse3 = np.zeros(totalSubs, dtype=bool)
        shouldUse3[opaqueNSgl] = errors3.min(0).sum(1) < errors4.min(0).sum(1)
        plts[:, shouldUse3] = plts3[:, shouldUse3][(1, 0, 3, 2), ]
        packedPltVals[shouldUse3] = packedPltVals[shouldUse3, ::-1]
        # compute indices (closest palette entry for each color, or just 3 if transparent)
        difs = rgb - plts
        dots = np.einsum("ijkl,ijkl->ijk", difs, difs)
        idcs = np.argmin(dots, 0)
        idcs[transparent] = 3
        # compression is done! pack to bytes & return
        # initially, all palettes are followed by all indices; use swapaxes to interleave
        packed = packedPltVals.tobytes() + cls._IDX_FMT.pack(idcs)
        return np.frombuffer(packed, np.uint32).reshape(2, -1).swapaxes(0, 1).tobytes()


class TEX0(Subfile):
    """BRRES subfile for textures."""

    _VALID_VERSIONS = (1, 3)

    def __init__(self, name: str = None, version = -1):
        super().__init__(name, version)
        self.fmt: type[ImageFormat] = None
        self.images: list[np.ndarray] = []

    @property
    def dims(self):
        return np.array(self.images[0].shape[:2])[::-1]

    @property
    def numMipmaps(self):
        """Number of mipmaps in this image (main image not included)."""
        return len(self.images) - 1

    @property
    def isPaletteIndices(self):
        """True if this image contains indices into a palette."""
        return issubclass(self.fmt, PaletteImageFormat)

    @property
    def isRGBA(self):
        """True if this image contains RGBA data.
        
        Note that it might not have an RGBA format; this refers to the unpacked data, not packed.
        (For instance, RGB565 data is unpacked as RGBA data w/ all 1s for alpha; this makes it easy
        to change formats)
        """
        return not issubclass(self.fmt, PaletteImageFormat)

    def mipmapDims(self, mipmapLevel = 0):
        """Dimensions (px) for a given mipmap level (0 = full image, 1 = half size, etc)."""
        return self.dims >> mipmapLevel


class TEX0Serializer(SubfileSerializer[BRRES_SER_T, TEX0]):

    DATA_TYPE = TEX0
    FOLDER_NAME = "Textures(NW4R)"
    MAGIC = b"TEX0"

    _HEAD_STRCT = Struct(">iiIHHIIffii 8x")
    _IMG_FORMATS: tuple[type[ImageFormat]] = (
        I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA8, None, C4, C8, C14X2, None, None, None, CMPR
    )

    def _mipmapSize(self, mipmapLevel):
        """Size of a mipmap ((level 0 = full image, 1 = half size, etc)) in bytes."""
        return self._data.fmt.imgSize(self._data.mipmapDims(mipmapLevel))


class TEX0Reader(TEX0Serializer, SubfileReader):

    def unpack(self, data: bytes):
        super().unpack(data)
        file = self._data
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset + self._CMN_STRCT.size)
        dims = unpackedHeader[3:5]
        file.fmt: type[ImageFormat] = self._IMG_FORMATS[unpackedHeader[5]]
        if file.fmt is None:
            raise NotImplementedError("Unsupported image format detected")
        dataOffset = unpackedHeader[0]
        dataSize = self.offset
        if dataOffset > 0:
            # add dummy image for the sake of calculating mipmap dimensions
            file.images.append(np.empty(dims[::-1]))
            # unpack image data
            for mmLevel in range(unpackedHeader[6]):
                dataOffset += dataSize
                dataSize = self._mipmapSize(mmLevel)
                mmData = data[dataOffset : dataOffset + dataSize]
                file.images.append(file.fmt.importImg(mmData, file.mipmapDims(mmLevel)))
            # remove dummy image
            file.images.pop(0)
        return self


class TEX0Writer(TEX0Serializer, SubfileWriter):

    def _calcSize(self):
        dataSize = sum(self._mipmapSize(m) for m in range(len(self._data.images)))
        return self._CMN_STRCT.size + self._HEAD_STRCT.size + dataSize

    def pack(self):
        tex = self._data
        packedHeader = self._HEAD_STRCT.pack(self._CMN_STRCT.size + self._HEAD_STRCT.size,
                                             self.stringOffset(self._data.name) - self.offset,
                                             0, *tex.dims, self._IMG_FORMATS.index(tex.fmt),
                                             len(tex.images), 0, tex.numMipmaps, 0, 0)
        packedData = b"".join(tex.fmt.exportImg(img) for img in tex.images)
        return super().pack() + packedHeader + packedData
