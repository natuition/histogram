import cv2 as cv

config = {}

# ======================================================================================================================
# Paths
config["hist_database_path"] = "database\csv\database.npy"
config["patterns_dataset_dir"] = "database\\images"
config["query_image_path"] = "input\\prise8.jpg"
config["output_image_dir"] = "output\\"
config["output_image_name"] = "prise2 - AOI"
config["output_image_extension"] = ".jpg"

# ======================================================================================================================
# AOI Area of interest
config["aoi_top_border"] = 595
config["aoi_bottom_border"] = 1200
config["aoi_left_border"] = 854
config["aoi_right_border"] = 1734

# ======================================================================================================================
# Fragments
config["fragment_w"] = 120
config["fragment_h"] = 120
config["fragment_x_offset"] = 60
config["fragment_y_offset"] = 60

# ======================================================================================================================
# Histograms
config["hist_channels"] = [0, 1, 2]
config["hist_size"] = (8, 12, 3)
config["hist_range"] = [0, 180, 0, 256, 0, 256]
config["hist_comp_method"] = cv.HISTCMP_CHISQR

# for some hist comp algs lesser distance means more similar images, and vice versa
config["lesser_dist_more_similar"] = True
