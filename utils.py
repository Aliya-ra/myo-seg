import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List

def find_single_file(folder: Path, pattern: str) -> Path:
    """
    Search a folder for a given file pattern
    - if none found => raise error (missing data)
    - if more than 1 found => raise error (ambiguous data)
    - if exactly 1 found => return the path
    """
    cands = sorted(folder.glob(pattern))

    if len(cands) == 0:
        raise FileNotFoundError(f"No files mathcing {pattern} in {folder}")
    
    if len(cands) > 1:
        names = ", ".join(p.name for p in cands)
        raise RuntimeError(f"Found {len(cands)} files for{pattern} in {folder}")
    
    return cands[0]


def read_image_and_array(path: Path) -> Tuple[sitk.Image, np.ndarray]:
    """
    Read a medical image and converts it to a numpy array.
    Returns (SimpleITK Image Object, Numpy Array)
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)

    return img, arr


def parse_centerline_txt(path: Path) -> np.ndarray:
    """
    Reads a centerlind.txt file containing (X, Y, Z).
    Returns a np array of shape (N, 3)
    """
    df = pd.read_csv(path, sep=None, engine="python", header=None)

    pts = df.iloc[:, :3].to_numpy(dtype=float)

    return pts


def rasterize_centerline(ref_image: sitk.Image, pts_mm: np.ndarray, radius_mm: float, step_mm: float) -> sitk.Image:
    """

    """
    size = ref_image.GetSize()
    spacing = ref_image. GetSpacing()

    # Create an image with the same dimension as the CT image
    seeds = sitk.Image(size, sitk.sitkUInt8)
    seeds.CopyInformation(ref_image)

    for a, b in zip(pts_mm[:-1], pts_mm[1:]):
        
        segment = b - a     # vector from point a to point b in mm.
        segment_len = float(np.linalg.norm(segment))    # Euclidian distance between a and b

        if segment_len == 0: continue   # it means the points are identical (so skip)

        # Calculate how many steps we need to cover all the points
        n_steps = max(1, int(math.ceil(segment_len / step_mm)))

        # Interpolation
        for t in np.linspace(0.0, 1.0, n_steps + 1):
            p = a + t * segment   # points between each (a,b) pair


            try:
                idx = ref_image.TransformPhysicalPointToIndex(tuple(map(float, p)))
                seeds[idx] = 1

            except Exception:
                pass # points outside the image

    rad_vox = tuple(int(max(1, math.ceil(radius_mm / s))) for s in spacing)

    thick_tube = sitk.BinaryDilate(
        seeds,
        kernelRadius = rad_vox,
        kernelType=sitk.sitkBall,
        foregroundValue=1,
        backgroundValue=0
    )

    return thick_tube