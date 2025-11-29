"""
Helper functions for modeling bursts as 2D gausian functions.

Authors: James Bonaiuto <james.bonaiuto@isc.cnrs.fr>
         Maciej Szul <maciej.szul@isc.cnrs.fr>

Adaptation: Sotirios Papadopoulos <sotirios.papadopoulos@univ-lyon1.fr>
"""

import numpy as np


# 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):

    return np.exp(
        -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0))
    )


# Overlap
def overlap(a, b):

    return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]


# FWHM: non-overlapping
def fwhm_burst_norm(TF, peak):

    right_loc = np.nan
    cand = np.where(TF[peak[0], peak[1] :] <= TF[peak] / 2)[0]
    if len(cand):
        right_loc = cand[0]

    up_loc = np.nan
    cand = np.where(TF[peak[0] :, peak[1]] <= TF[peak] / 2)[0]
    if len(cand):
        up_loc = cand[0]

    left_loc = np.nan
    cand = np.where(TF[peak[0], : peak[1]] <= TF[peak] / 2)[0]
    if len(cand):
        left_loc = peak[1] - cand[-1]

    down_loc = np.nan
    cand = np.where(TF[: peak[0], peak[1]] <= TF[peak] / 2)[0]
    if len(cand):
        down_loc = peak[0] - cand[-1]

    if down_loc is np.nan:
        down_loc = up_loc
    if up_loc is np.nan:
        up_loc = down_loc
    if left_loc is np.nan:
        left_loc = right_loc
    if right_loc is np.nan:
        right_loc = left_loc

    horiz = np.nanmin([left_loc, right_loc])
    vert = np.nanmin([up_loc, down_loc])
    right_loc = horiz
    left_loc = horiz
    up_loc = vert
    down_loc = vert

    return right_loc, left_loc, up_loc, down_loc
