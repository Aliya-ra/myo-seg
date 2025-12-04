import math
import heapq
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

class GeodesicTerritorySolver:
    """
    Assigns every voxel in the myocardium to the nearest artery using a Stateful Djkstra algorithm
    """

    def __init__ (self, spacing_zyx: Tuple[float, float, float]):

        self.spacing = spacing_zyx
        self.neighbors = self._build_neighbor_offsets(spacing_zyx)

    def _build_neighbor_offsets(self, spacing: Tuple[float, float, float]) -> List[Tuple[int, int, int, float]]:
        """
        Returns a list of tuples: (dz, dy, dx, distance_mm) because the voxels are not isotropic (spacing differs in different dimentitons)
        For example:
        sx = 0.4004 mm  
        sy = 0.4004 mm  
        sz = 0.45 mm

        for all 26 neighbors in a 3d grid.
        """

        sz, sy, sx = spacing
        offsets = []

        # Loop through -1, 0, 1 in Z, Y, X directions
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue # Skip the center pixel (itself)
                    
                    # Euclidean distance in physical MM
                    dist = math.sqrt((dz * sz)**2 + (dy * sy)**2 + (dx * sx)**2)
                    offsets.append((dz, dy, dx, dist))
        
        return offsets
    

    def solve(self, myo_mask: np.ndarray, seed_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        State 0: outside Myocardium, allowed to move freely
        State 1: inside Myocardium, cannot go out anymore. (only allowed to move inside myo tissue)
        """

        shape = myo_mask.shape

        # initialize the distances with infinity
        dist_out = np.full(shape, np.inf, dtype=np.float32)
        dist_in = np.full(shape, np.inf, dtype=np.float32)
        
        labels_out = np.zeros(shape, dtype=np.int16)
        labels_in = np.zeros(shape, dtype=np.int16)

        # Priority Queue: (Distance, State, Z, Y, X)
        # State 0 = Outside, State 1 = Inside
        pq = [] # priority queue for Dijkstra algorithm (Distance, State, Z, Y, X)


        # ------- Main Algo --------
        # step 1: find all the voxels that are part of an artery
        seed_indices = np.argwhere(seed_labels > 0)

        print(f" [Solver] Initializing {len(seed_indices)} seed voxels...")

        for z, y, x in seed_indices:
            z, y, x = int(z), int(y), int(x)
            label = int(seed_labels[z, y, x])

            dist_out[z, y, x] = 0.0
            labels_out[z, y, x] = label
            heapq.heappush(pq, (0.0, 0, z, y, x))

            # If seed is ALSO physically inside the mask, initialize Inside State immediately
            if myo_mask[z, y, x]:
                dist_in[z, y, x] = 0.0
                labels_in[z, y, x] = label
                heapq.heappush(pq, (0.0, 1, z, y, x))

        

        # Progress bar using tqdm
        total_myo_voxels = np.count_nonzero(myo_mask)
        pbar = tqdm(total=total_myo_voxels, desc="  Computing Territories", unit="vox", mininterval=0.5, leave=False)
        finalized_count = 0


        # step 2: run Dijkstra
        while pq:
            d, state, z, y, x = heapq.heappop(pq)

            # Optimization: Check against the CORRECT distance map
            if state == 0:
                if d > dist_out[z, y, x]: continue
                curr_label = labels_out[z, y, x]
            else:
                if d > dist_in[z, y, x]: continue
                curr_label = labels_in[z, y, x]


                pbar.update(1)
                finalized_count += 1


                if finalized_count >= total_myo_voxels:
                    break

            # Check neighbors
            for dz, dy, dx, edge_len in self.neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx

                # Bounds check
                if not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                    continue
                
                new_dist = d + edge_len
                is_myo = myo_mask[nz, ny, nx]

                if state == 0:
                    # Logic for OUTSIDE
                    # 1. Try to walk to another Outside voxel
                    if new_dist < dist_out[nz, ny, nx]:
                        dist_out[nz, ny, nx] = new_dist
                        labels_out[nz, ny, nx] = curr_label
                        heapq.heappush(pq, (new_dist, 0, nz, ny, nx))
                    
                    # 2. Try to enter the Myocardium (state 0 -> 1)
                    if is_myo:
                        # We compare against dist_IN here
                        if new_dist < dist_in[nz, ny, nx]:
                            dist_in[nz, ny, nx] = new_dist
                            labels_in[nz, ny, nx] = curr_label
                            heapq.heappush(pq, (new_dist, 1, nz, ny, nx))

                elif state == 1:
                    # Logic for INSIDE
                    if is_myo:
                        if new_dist < dist_in[nz, ny, nx]:
                            dist_in[nz, ny, nx] = new_dist
                            labels_in[nz, ny, nx] = curr_label
                            heapq.heappush(pq, (new_dist, 1, nz, ny, nx))

        # Return the inside labels
        # fill any unreached myocardium with 0
        final_labels = labels_in
        final_dists = dist_in
        pbar.close()
        
        return final_labels, final_dists





        
    
