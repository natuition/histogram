import cv2 as cv

config = {}


# ======================================================================================================================
# Paths
config["hist_database_path"] = "database\\database.npy"
config["patterns_dataset_dir"] = "database\\images"
config["query_image_path"] = "input\\prise8.jpg"
config["output_image_dir"] = "output\\"
config["output_image_name"] = "prise2 - AOI"
config["output_image_extension"] = ".jpg"


# ======================================================================================================================
# Image processing
# AOI - area of interest
config["aoi_top_border"] = 595
config["aoi_bottom_border"] = 1200
config["aoi_left_border"] = 854
config["aoi_right_border"] = 1734

config["fragment_w"] = 120
config["fragment_h"] = 120
config["fragment_x_offset"] = 60
config["fragment_y_offset"] = 60

config["hist_channels"] = [0, 1, 2]
config["hist_size"] = (8, 12, 3)
config["hist_range"] = [0, 180, 0, 256, 0, 256]
config["hist_comp_method"] = cv.HISTCMP_CHISQR

# for some hist comp algs lesser distance means more similar images, and vice versa
config["lesser_dist_more_similar"] = True


# ======================================================================================================================
# App config
# Modes:
# "database" - loads image and adds fragments from AOI to patterns and hist DB. Image sourse depends on use_camera
# setting
# "searching" - loads image and searches most unlike grass fragment and tries to extract this plant. Image source
# depends on use_camera setting
config["app_mode"] = "searching"
config["use_camera"] = False
config["camera_w"] = 2592
config["camera_h"] = 1944
config["camera_framerate"] = 32
config["cork_center_x"] = 1292
config["cork_center_y"] = 1172
config["one_mm_in_px"] = 4


# ======================================================================================================================
# Smoothie settings
config["smoothie_host"] = "169.254.232.224"
config["smoothie_xy_F"] = 1000
config["smoothie_z_F"] = 1000
config["extraction_z"] = 100
