"""Timestamp-only IMU synchronization, in nanoseconds and SI units."""
from bisect import bisect_left
import numpy as np


def interpolate_imu(gyro, accel, start_ns, end_ns, max_gap_ns=50_000_000):
    """Return (start,end] gyro samples with bracketed linear accel interpolation.

    No extrapolation or sample duplication. Raises on missing data or large gaps.
    The caller retains future gyro samples and one acceleration before the boundary.
    """
    if len(accel) < 2:
        raise ValueError('Need two acceleration samples')
    at = [a[0] for a in accel]
    if any(b <= a for a, b in zip(at, at[1:])):
        raise ValueError('Non-monotonic acceleration timestamps')
    gt = [g[0] for g in gyro]
    if any(b <= a for a, b in zip(gt, gt[1:])):
        raise ValueError('Non-monotonic gyro timestamps')
    result=[]
    for t,g in gyro:
        if not start_ns < t <= end_ns:
            continue
        i=bisect_left(at,t)
        if i < len(at) and at[i]==t:
            a=np.asarray(accel[i][1])
        else:
            if i==0 or i==len(at):
                raise ValueError('Acceleration does not bracket gyro timestamp')
            lo,hi=accel[i-1],accel[i]
            if hi[0]-lo[0] > max_gap_ns:
                raise ValueError('Acceleration gap exceeds 50 ms')
            weight=(t-lo[0])/(hi[0]-lo[0])
            a=(1-weight)*np.asarray(lo[1])+weight*np.asarray(hi[1])
        if not np.isfinite([*g,*a]).all():
            raise ValueError('Nonfinite IMU data')
        result.append([int(t),*map(float,g),*map(float,a)])
    if not result:
        raise ValueError('No gyro samples in camera interval')
    coverage=[start_ns,*[r[0] for r in result],end_ns]
    if any(b-a > max_gap_ns for a,b in zip(coverage,coverage[1:])):
        raise ValueError('Gyro gap exceeds 50 ms')
    return result
