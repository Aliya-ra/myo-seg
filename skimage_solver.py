import numpy as np
import scipy.ndimage as nd
from skimage.segmentation import watershed
from typing import Tuple

class GeodesicTerritorySolver:
    def __init__(self, spacing_zyx: Tuple[float, float, float]):
        self.spacing = spacing_zyx

    def solve(self, myo_mask: np.ndarray, seed_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the Watershed algorithm with an aggressive buffer to capture 
        arteries floating in epicardial fat.
        """
        print("  [Solver] Running vectorized watershed...")

        # --- 1. Define the "Playable Area" (The Band) ---
        # FIX: Increased buffer from 5.0mm to 25.0mm
        # The RCA can be floating far from the myo wall due to fat/grooves.
        # We need a bridge wide enough to reach it.
        buffer_mm = 25.0 
        
        # Calculate iterations based on the smallest spacing (usually X or Y)
        # We use the min spacing to ensure the buffer is at LEAST 25mm in all directions
        min_spacing = min(self.spacing)
        iterations = int(buffer_mm / min_spacing)
        
        # Optimization: Use a slightly coarser structure for dilation to save time
        # but iterate enough times to cover the distance.
        struct = nd.generate_binary_structure(3, 1) 
        
        print(f"  -> Dilating myocardium by {buffer_mm}mm ({iterations} iterations)...")
        dilated_myo = nd.binary_dilation(myo_mask, structure=struct, iterations=iterations)
        
        # The final mask where water (artery labels) is allowed to flow:
        walkable_mask = dilated_myo | (seed_labels > 0)

        # --- DEBUG: CHECK CONNECTIVITY ---
        # This checks if the arteries are actually touching the playable area.
        # If this number is low, the buffer is still too small.
        # We check intersection of (Seeds) and (Dilated Myo).
        overlap = np.logical_and(dilated_myo, seed_labels > 0)
        if not np.any(overlap):
             print("  [WARNING] CRITICAL: Arteries are effectively disconnected from the heart!")
             print("            The buffer_mm is too small to reach the centerline.")

        # --- 2. Run Watershed ---
        # compactness=0 ensures the boundaries are driven by pure distance
        labels_full = watershed(
            image=np.zeros_like(myo_mask, dtype=np.uint8), 
            markers=seed_labels, 
            mask=walkable_mask,
            compactness=0
        )

        # --- 3. Cleanup ---
        # Crop back to the original True Myocardium
        final_labels = np.where(myo_mask, labels_full, 0).astype(np.int16)
        
        # Dummy distance map (watershed doesn't produce one natively)
        dummy_dist = np.zeros_like(final_labels, dtype=np.float32)

        return final_labels, dummy_dist