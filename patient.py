import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Optional

import config
import utils
from solver import GeodesicTerritorySolver

class PatientCase:
    
    def __init__ (self, case_dir: Path):
        
        self.case_dir = case_dir
        self.case_id = case_dir.name

        self.img = None         # SITK image
        self.img_np = None      # np array
        self.myo = None
        self.myo_np = None
        self.spacing = None

        self.available_arteries = []  # List of strings ['LAD', etc.]
        self.cl_dir = None        # Path to the centerlines folder
        self.territory_map = None   # The final result


    
    def load_data(self):

        """
        Find the files and load them
        """
        img_file  = utils.find_single_file(self.case_dir / config.IMG_FOLDER, config.MHD_PATTERN)
        myo_file = utils.find_single_file(self.case_dir / config.MASK_FOLDER, config.MHD_PATTERN)

        self.img, self.img_np = utils.read_image_and_array(img_file)
        self.myo, self.myo_np = utils.read_image_and_array(myo_file)

        self.spacing = self.myo.GetSpacing()

        # find the centerlines
        for dname in config.CL_DIRS:
            if(self.case_dir / dname).exists():
                self.cl_dir = self.case_dir / dname
                break
        
        if not self.cl_dir:
            raise FileNotFoundError(f"No centerline folder found for {self.case_id}")
        

        self.available_arteries = []
        for artery in config.POSSIBLE_ARTERIES:
            if (self.cl_dir / f"{artery}.txt").exists():
                self.available_arteries.append(artery)



    def run_analysis(self) -> Dict:
        """
        The main function:
        Rasterize the centerlines => Solve => Save
        """

        if not self.available_arteries:
            print(f"Skipping patient {self.case_id}, no arteries found.")
            return{}
        
        seed_vol_np = np.zeros_like(self.myo_np, dtype=np.int16)

        artery_map = {name: i for i, name in enumerate(self.available_arteries,1)}

        for artery_name, idx in artery_map.items():

            # Parse the txt file
            pts = utils.parse_centerline_txt(self.cl_dir / f"{artery_name}.txt")

            # Draw the tube
            tube_img = utils.rasterize_centerline(self.img, pts, config.RASTER_RADIUS_MM, config.RASTER_STEP_MM)

            tube_arr = sitk.GetArrayFromImage(tube_img)
            seed_vol_np[tube_arr > 0] = idx

        
        # ----------Run the solver------------
        # Convert spacing to (z, y, x) order for numpy
        spacing_zyx = (self.spacing[2], self.spacing[1, self.spacing[0]])

        solver = GeodesicTerritorySolver(spacing_zyx)

        myo_mask_bool = (self.myo_np > 0)


        # Main Call
        self.territory_map, _ = solver.solve(myo_mask_bool, seed_vol_np)

        self._save_visualization(seed_vol_np)
        stats = self._calculate_statistics(artery_map, myo_mask_bool)

        return stats
    


    def _calculate_statistics(self, artery_map: Dict[str, int], myo_mask: np.ndarray) -> Dict:
        """
        Counts voxels and calculates the percentages.
        """
        # Volume of one voxel in mL = (sx * sy * sz) / 1000
        voxel_vol_ml = np.prod(self.spacing) / 1000.0
        
        results = {
            "case_id": self.case_id, 
            "available_arteries": self.available_arteries
        }
        
        total_vol = 0.0
        
        # Loop through each artery (LAD, RCA...)
        for art_name, idx in artery_map.items():
            # Count voxels that match this artery ID AND are inside myocardium
            mask = (self.territory_map == idx) & myo_mask
            vol = mask.sum() * voxel_vol_ml
            
            results[f"{art_name}_volume_ml"] = vol
            total_vol += vol
        
        results["total_volume_ml"] = total_vol
        
        # Compute percentages
        for art_name in self.available_arteries:
            vol = results[f"{art_name}_volume_ml"]
            pct = (vol / total_vol * 100) if total_vol > 0 else 0
            results[f"{art_name}_percent"] = pct
            
        return results
    


    def _save_visualization(self, seed_vol_np: np.ndarray):
        """
        Saves a .mhd file showing the territories + the original arteries.
        """
        # Start with the territory map
        combined = self.territory_map.copy()
        
        # include the arteries in the mask
        combined[seed_vol_np > 0] = seed_vol_np[seed_vol_np > 0]
        
        out_img = sitk.GetImageFromArray(combined.astype(np.uint8))
        out_img.CopyInformation(self.myo)
        
        save_path = self.case_dir / "territories_final.mhd"
        sitk.WriteImage(out_img, str(save_path))
        print(f"  [Saved] {save_path}")


