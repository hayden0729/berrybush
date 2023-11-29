import numpy as np


def interpolateCurve(curve: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Hermite curve interpolation. Return the X/Y/tangent values at some X on the curve.

    Multiple positions, and optionally multiple curves (one per position), may be provided
    for batch calculations.

    A single curve is made up of two points, each with an X, Y, and tangent value.
    (For instance, `[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]` defines 2 curves)
    """
    x1 = curve[..., 0, 0]
    y1 = curve[..., 0, 1]
    t1 = curve[..., 0, 2]
    x2 = curve[..., 1, 0]
    y2 = curve[..., 1, 1]
    t2 = curve[..., 1, 2]
    span = x2 - x1
    # https://www.cubic.org/docs/hermite.htm
    fac = (x - x1) / span
    fac2 = fac ** 2
    fac3 = fac ** 3
    h1 = 2 * fac3 - 3 * fac2 + 1
    h2 = -2 * fac3 + 3 * fac2
    h3 = fac3 - 2 * fac2 + fac
    h4 = fac3 - fac2
    h1p = 6 * fac2 - 6 * fac
    h2p = -6 * fac2 + 6 * fac
    h3p = 3 * fac2 - 4 * fac + 1
    h4p = 3 * fac2 - 2 * fac
    # this hermite implementation is non-parameterized, so tangents are scalars,
    # but they're usually vectors (that's the case in the link above)
    # the parameterized version would have tangents [span, span * t1] and [span, span * t2]
    # since the formula we're using (linked) is for the parameterized verson,
    # we have to multiply t1 & t2 by span
    # this multiplication is also referenced here:
    # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_an_arbitrary_interval
    y = h1 * y1 + h2 * y2 + (h3 * t1 + h4 * t2) * span
    yp = (h1p * y1 + h2p * y2 + (h3p * t1 + h4p * t2) * span) / span
    interpolated = np.stack((x, y, yp), axis=-1)
    # finally, where span is 0, arbitrarily take right control point (we get nan otherwise)
    noSpan = span == 0
    if np.isscalar(noSpan):
        # if there's only one curve, we can't use the general method bc noSpan is just a boolean
        # so, we handle it this way instead
        if noSpan:
            interpolated[x == curve[1, 0]] = curve[:, 1]
    else:
        # general case - we have a bunch of curves, so noSpan will vary for each position
        interpolated[noSpan] = curve[noSpan, 1]
    return interpolated


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
    curves = np.stack((spline[i - 1], spline[i]), axis=1) # each position's corresponding curve
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
    return simplifySplineRough(interpolateSpline(spline, positions), maxError)


def simplifySplineRough(points: np.ndarray, maxError: float) -> np.ndarray:
    """Simplify a Hermite spline, enforcing a maximum error at the original control points."""
    # define the new curve using the first & last points of the current spline, then test errors
    newCurve = np.stack((points[0], points[-1]))
    newPoints = interpolateCurve(newCurve, points[:, 0])
    errors = np.abs(newPoints[:, 1] - points[:, 1])
    maxErrorIndex = np.argmax(errors)
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
