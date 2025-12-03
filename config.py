from pathlib import Path

# File Constants
ROOT_DIR = Path("CT_DATA")
MHD_PATTERN = "*.mhd"
IMG_FOLDER = "image_data_mhd"
MASK_FOLDER = "segmentation_mask"
CL_DIRS = ["centerlines_main_five", "centerlines_main"]

# Anatomical Constants
POSSIBLE_ARTERIES = ["LAD", "LCX", "OM", "RCA", "RPLB"]

# Algorithm Constants
RASTER_RADIUS_MM = 1.5
RASTER_STEP_MM = 0.5