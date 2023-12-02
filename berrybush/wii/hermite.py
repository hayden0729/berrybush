import numpy as np


def h0eval(t: np.ndarray):
    """Evaluate the first Hermite basis function at time(s) t."""
    return 2 * (t ** 3) - 3 * (t ** 2) + 1


def h1eval(t: np.ndarray):
    """Evaluate the second Hermite basis function at time(s) t."""
    return -2 * (t ** 3) + 3 * (t ** 2)


def h2eval(t: np.ndarray):
    """Evaluate the third Hermite basis function at time(s) t."""
    return (t ** 3) - 2 * (t ** 2) + t


def h3eval(t: np.ndarray):
    """Evaluate the fourth Hermite basis function at time(s) t."""
    return (t ** 3) - (t ** 2)


def hp1eval(t: np.ndarray):
    """Evaluate the derivative of the first Hermite basis function at time(s) t."""
    return 6 * (t ** 2) - 6 * t


def hp2eval(t: np.ndarray):
    """Evaluate the derivative of the second Hermite basis function at time(s) t."""
    return -6 * (t ** 2) + 6 * t


def hp3eval(t: np.ndarray):
    """Evaluate the derivative of the third Hermite basis function at time(s) t."""
    return 3 * (t ** 2) - 4 * t + 1


def hp4eval(t: np.ndarray):
    """Evaluate the derivative of the fourth Hermite basis function at time(s) t."""
    return 3 * (t ** 2) - 2 * t


def generateBasisLookup(n: int):
    """Generate a lookup table of length n for the Hermite basis functions & their derivatives."""
    t = np.linspace(0, 1, n, dtype=np.float64)
    return np.array((
        (h0eval(t), h1eval(t), h2eval(t), h3eval(t)),
        (hp1eval(t), hp2eval(t), hp3eval(t), hp4eval(t))
    ))


# note: 64 bytes per entry, so 2^16 entries -> 4 MB
# plus one extra entry, since it has to include both endpoints
# (dividing based on a power of 2 is 100% arbitrary,
# but probably better than a power of 2 minus one)
BASIS_LOOKUP = generateBasisLookup(2 ** 16 + 1)


def interpolateCurve(curve: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Hermite curve interpolation. Return the X/Y/tangent values at some X on the curve.

    Multiple positions, and optionally multiple curves (one per position), may be provided
    for batch calculations.

    A single curve is made up of two points, each with an X, Y, and tangent value.
    (For instance, `[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]` defines 2 curves)
    """
    x1, y1, t1 = curve[..., 0, 0], curve[..., 0, 1], curve[..., 0, 2]
    x2, y2, t2 = curve[..., 1, 0], curve[..., 1, 1], curve[..., 1, 2]
    span = x2 - x1
    # handle span of 0 to prevent division by 0 problems
    # whether it's a scalar or array,
    # do stuff that will arbitrarily result in taking the left control point
    if np.isscalar(span):
        if span == 0:
            return np.repeat(curve[0, np.newaxis], len(x), axis=0)
    else:
        span[span == 0] = 1
    # https://www.cubic.org/docs/hermite.htm
    t = (x - x1) / span
    i = ((BASIS_LOOKUP.shape[-1] - 1) * t + .5).astype(np.int64)
    h, hp = BASIS_LOOKUP[:, :, i]
    # note: this hermite implementation is non-parameterized, so tangents are scalars,
    # but they're usually vectors (that's the case in the link above)
    # the parameterized version would have tangents [span, span * t1] and [span, span * t2]
    # since the formula we're using (linked) is for the parameterized verson,
    # we have to multiply t1 & t2 by span
    # this multiplication is also referenced here:
    # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_an_arbitrary_interval
    result = np.empty((*x.shape, 3))
    result[..., 0] = x
    result[..., 1] = y1 * h[0] + y2 * h[1] + (t1 * h[2] + t2 * h[3]) * span
    result[..., 2] = (y1 * hp[0] + y2 * hp[1]) / span + t1 * hp[2] + t2 * hp[3]
    return result


def interpolateSpline(spline: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Get the X, Y, and tangent values for a Hermite spline at the given X values.
    
    (A spline is an array of control points, each with an X, Y, and tangent value)
    """
    interpolated = np.empty((*positions.shape, 3))
    interpolated[:, 0] = positions
    # first, determine which positions are in bounds
    outOfBoundsL = positions < spline[0, 0]
    outOfBoundsR = positions > spline[-1, 0]
    outOfBounds = np.logical_or(outOfBoundsL, outOfBoundsR)
    inBounds = np.logical_not(outOfBounds)
    # handle out-of-bounds positions
    interpolated[outOfBoundsL, 1] = spline[0, 1]
    interpolated[outOfBoundsR, 1] = spline[-1, 1]
    interpolated[outOfBounds, 2] = 0
    # handle in-bounds positions
    i = np.searchsorted(spline[:, 0], positions[inBounds]) # each position's next ctrl point index
    curves = np.empty((len(i), 2, 3)) # each position's corresponding curve
    curves[:, 0] = spline[i - 1]
    curves[:, 1] = spline[i]
    interpolated[inBounds] = interpolateCurve(curves, positions[inBounds])
    return interpolated


def simplifySpline(spline: np.ndarray, maxError: float, precision: float = 1) -> np.ndarray:
    """Simplify a Hermite spline, enforcing a maximum error tested throughout the curve's domain.

    The error is tested at a set of X values no more than `precision` units apart from one another
    that includes both endpoints.

    (e.g., if the spline starts & ends on integer X values, and `precision = 1`, the error will be
    tested at every integer X value in the domain)
    """
    numPositions = int(np.ceil((spline[-1, 0] - spline[0, 0]) / precision)) * precision + 1
    positions = np.linspace(spline[0, 0], spline[-1, 0], numPositions)
    # maxError *= np.ptp(spline[..., 1]) # uncomment for a "relative" error rather than absolute
    simplified = simplifySplineRough(interpolateSpline(spline, positions), maxError)
    # since we do simplification based on the interpolated, it's possible (though rare) that
    # the "simplified" version will actually be larger than the original
    # in that case, just return the original
    return min(spline, simplified, key=len)


def simplifySplineRough(points: np.ndarray, maxError: float) -> np.ndarray:
    """Simplify a Hermite spline, enforcing a maximum error at the original control points."""
    # define the new curve using the first & last points of the current spline, then test errors
    newCurve = np.empty((2, 3)) # this is faster than np.stack((points[0], points[-1]))
    newCurve[0] = points[0]
    newCurve[1] = points[-1]
    newPoints = interpolateCurve(newCurve, points[:, 0])
    errors = np.abs(newPoints[:, 1] - points[:, 1])
    maxErrorIndex = errors.argmax()
    # at the point of maximum error, if max error is exceeded, use that point as a pivot and
    # repeat the process on both sides recursively - eventually you get a simplified spline
    if errors[maxErrorIndex] > maxError:
        left = simplifySplineRough(points[:maxErrorIndex + 1], maxError)
        right = simplifySplineRough(points[maxErrorIndex:], maxError)
        return np.concatenate((left, right[1:]))
    else:
        return newCurve
    # note that this is loosely based on philip j schneider's bezier fitting algorithm,
    # but it's dramatically simpler because we use scalar tangents rather than vectors
    # (which has consequences such as making t and y both functions of x and makes everything nice)
    # (this is also mentioned in hermite() above)


# print(simplifySpline(np.array([[0, 0, 1], [2, 2, 1], [4, 4, 1]]), 0.01)) # demo
