import math
import heapq
import numpy as np
from typing import Tuple, List

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
        dist_map = np.full(shape, np.inf, dtype=np.float32)   # to check: size of variables (flaot32 and int16 might not be necessary)
        label_map = np.zeros(shape, dtype=np.int16)

        pq = [] # priority queue for Dijkstra algorithm (Distance, State, Z, Y, X)


        # ------- Main Algo --------
        # step 1: find all the voxels that are part of an artery
        seed_indices = np.argwhere(seed_labels > 0)

        print(f" [Solver] Initializing {len(seed_indices)} seed voxels...")

        for z, y, x in seed_indices:
            z, y, x = int(z), int(y), int(x)
            label = int(seed_labels[z, y, x])

            dist_map[z, y, x] = 0.0         # the distance to the artery is 0 (the voxel is in the artery)
            label_map[z, y, x] = label      # assigns the corresponding label (e.g. LAD = 1, etc.)

        
        # step 2: run Dijkstra
        while pq:
            d, state, z, y, x = heapq.heappop(pq)

            if d > dist_map[z, y, x]:  # there is already a shorter path
                continue

            current_label = label_map[z, y, x]

            # check all 26 neighbors
            for dz, dy, dx, edge_len in self.neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx    # nz, ny, nx are the coordinates of the neighbor voxels

                # don't go off the image
                if not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                    continue

                new_dist = d + edge_len
                is_neighbor_myo = myo_mask[nz, ny, nx]


                # Case A: we are outside of myo
                if state == 0:
                    if new_dist < dist_map[nz, ny, nx]:
                        dist_map [nz, ny, nx] = new_dist
                        label_map[nz, ny, nx] = current_label

                        new_state = 1 if is_neighbor_myo else 0
                        heapq.heappush(pq, (new_dist, new_state, nz, ny, nx))
                
                # Case B: we are inside of myo
                elif state == 1:
                    if is_neighbor_myo:
                        if new_dist < dist_map[nz, ny, nx]:
                            dist_map [nz, ny, nx] = new_dist
                            label_map[nz, ny, nx] = current_label
                            heapq.heappush(pq, (new_dist, 1, nz, ny, nx))
        
        return label_map, dist_map





        
    
