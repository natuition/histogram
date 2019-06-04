from tarax.colordescriptor import ColorDescriptor
from tarax.searcher import Searcher
import glob
import cv2
import json
import numpy as np

# load config
with open("config/config.json", "r") as config_file:
    config = json.loads(config_file.read())

# initialize the color descriptor
# For our image search engine, we’ll be utilizing a 3D color histogram in the HSV color space with 8 bins for the Hue channel, 12 bins for the saturation channel, and 3 bins for the value channel, yielding a total feature vector of dimension 8 x 12 x 3 = 288.
# This means that for every image in our dataset, no matter if the image is 36 x 36 pixels or 2000 x 1800 pixels, all images will be abstractly represented and quantified using only a list of 288 floating point numbers.
# https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
cd = ColorDescriptor((8, 12, 3))


def get_aoi_areas(image_from_camera):
    """Cuts corkscrew from image from camera and dark borders, returns list of 3 images that are ready to split on fragments"""

    img_y_center, img_x_center = int(image_from_camera.shape[0] / 2), int(image_from_camera.shape[1] / 2)  # y_c = 972, x_c = 1296

    aoi_1_y_start = img_y_center - int(config["aoi_h"] / 2)
    aoi_1_y_end = img_y_center + config["from_center_to_cork_y"]
    aoi_1_x_start = img_x_center - int(config["aoi_w"] / 2)
    aoi_1_x_end = img_x_center + int(config["aoi_w"] / 2)

    aoi_2_y_start = aoi_1_y_end
    aoi_2_y_end = img_y_center + int(config["aoi_h"] / 2)
    aoi_2_x_start = aoi_1_x_start
    aoi_2_x_end = img_x_center - int(config["cork_w"] / 2)

    aoi_3_y_start = aoi_1_y_end
    aoi_3_y_end = aoi_2_y_end
    aoi_3_x_start = img_x_center + int(config["cork_w"] / 2)
    aoi_3_x_end = img_x_center + int(config["aoi_w"] / 2)

    # get areas of interest
    aoi_image_1 = image_from_camera[aoi_1_y_start:aoi_1_y_end, aoi_1_x_start:aoi_1_x_end]
    aoi_image_2 = image_from_camera[aoi_2_y_start:aoi_2_y_end, aoi_2_x_start:aoi_2_x_end]
    aoi_image_3 = image_from_camera[aoi_3_y_start:aoi_3_y_end, aoi_3_x_start:aoi_3_x_end]

    return [aoi_image_1, aoi_image_2, aoi_image_3]

    """
    # draw AOIs on src img
    # image = cv2.circle(image, (img_x_center, img_y_center), 5, (255, 0, 0), 2)
    image = cv2.rectangle(image, (aoi_1_x_start, aoi_1_y_start), (aoi_1_x_end, aoi_1_y_end), (255, 0, 0), 2)
    image = cv2.rectangle(image, (aoi_2_x_start, aoi_2_y_start), (aoi_2_x_end, aoi_2_y_end), (255, 0, 0), 2)
    image = cv2.rectangle(image, (aoi_3_x_start, aoi_3_y_start), (aoi_3_x_end, aoi_3_y_end), (255, 0, 0), 2)
    
    cv2.imwrite(config["output_image_dir"] +
                config["output_image_name"] + " 1" +
                config["output_image_extension"], aoi_image_1)
    cv2.imwrite(config["output_image_dir"] +
                config["output_image_name"] + " 2" +
                config["output_image_extension"], aoi_image_2)
    cv2.imwrite(config["output_image_dir"] +
                config["output_image_name"] + " 3" +
                config["output_image_extension"], aoi_image_3)
    cv2.imwrite(config["output_image_dir"] +
                config["output_image_name"] + " Orig with lines" +
                config["output_image_extension"], image_from_camera)
    """


def get_fragments(image):
    image_h, image_w = image.shape[:2]

    # if there will remain a piece of the image that is less than the shift distance flag
    additional_col_proc = image_w % config["fragment_x_offset"] != 0
    additional_line_proc = image_h % config["fragment_y_offset"] != 0
    offsets_x_cnt = image_w // config["fragment_x_offset"] - 1
    offsets_y_cnt = image_h // config["fragment_y_offset"] - 1
    cur_start_y = 0
    cur_end_y = config["fragment_h"]
    cur_start_x = 0
    cur_end_x = config["fragment_w"]

    # loop over image fragments
    fragments = []
    for line in range(offsets_y_cnt):
        for column in range(offsets_x_cnt):
            fragments.append(image[cur_start_y:cur_end_y, cur_start_x:cur_end_x])
            cur_start_x += config["fragment_x_offset"]
            cur_end_x += config["fragment_x_offset"]
            pass
        # if there remains a piece of the image (last column)
        if additional_col_proc:
            # taking first right fragment of the current line
            fragments.append(image[cur_start_y:cur_end_y, image_w - config["fragment_w"]:image_w])

        # set x coords to start (first column) and set y coords to next row
        cur_start_x = 0
        cur_end_x = config["fragment_w"]
        cur_start_y += config["fragment_y_offset"]
        cur_end_y += config["fragment_y_offset"]
        pass

    # if there remains a piece of the image (last line)
    if additional_line_proc:
        # set coords to last row (size_y - frag_y_size) and first column
        cur_start_x = 0
        cur_end_x = config["fragment_w"]
        cur_start_y = image_h - config["fragment_h"]
        cur_end_y = image_h

        for column in range(offsets_x_cnt):
            fragments.append(image[cur_start_y:cur_end_y, cur_start_x:cur_end_x])
            cur_start_x += config["fragment_x_offset"]
            cur_end_x += config["fragment_x_offset"]

        # if there remains a piece of the image (last column)
        if additional_col_proc:
            # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
            fragments.append(image[cur_start_y:cur_end_y, image_w - config["fragment_w"]:image_w])

    return fragments


def save_non_weed_patterns(patterns: list):
    i = 0
    for image in patterns:
        cv2.imwrite(config["patterns_dataset_dir"] + r"\nonweed " + str(i) + ".jpg", image)
        i += 1


def create_database_from_files():
    with open(config["hist_database_path"], "w") as output_file:
        # use glob to grab the image paths and loop over them
        for image_path in glob.glob(config["patterns_dataset_dir"] + "/*.jpg"):  # or "/*.png"
            # extract the image ID (i.e. the unique filename) from the image
            # path and load the image itself
            image_unique_name = image_path[image_path.rfind("/") + 1:]
            image = cv2.imread(image_path)

            # make image description and convert to str
            features = [str(f) for f in cd.describe(image)]
            output_file.write("%s,%s\n" % (image_unique_name, ",".join(features)))


def main():
    image = cv2.imread(config["query_image_path"])
    aoi_areas = get_aoi_areas(image)
    patterns = []

    for area in aoi_areas:
        patterns.extend(get_fragments(area))

    save_non_weed_patterns(patterns)

    create_database_from_files()


# temp test section
test_image = cv2.imread(config["query_image_path"])
sr = Searcher(config["hist_database_path"])


def temp_perf_test_func_to_execute():
    aoi_areas = get_aoi_areas(test_image)
    fragments = []

    for area in aoi_areas:
        fragments.extend(get_fragments(area))

    # take max of min
    result = ["Init value", 0]
    i = 1
    for fragment in fragments:
        print("Processing fragment ", i)
        i += 1
        histogram = cd.describe(fragment)
        key, dist = sr.search_best(histogram)
        if dist > result[1]:
            result[0], result[1] = key, dist

    # uncomment if you want to save results on HDD
    # CAUTION! saving WILL slow down algorythm, dont use when measuring
    """
    with open(config["output_image_dir"] + "most not similar fragment with patterns DB.txt", "w") as file:
        data = "Path: " + result[0] + "\nDist: " + result[1]
        file.write(data)
    """


def performance_test():
    import timeit
    execution_time = timeit.repeat(temp_perf_test_func_to_execute, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))


if __name__ == "__main__":
    performance_test()
