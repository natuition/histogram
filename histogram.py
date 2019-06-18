import cv2 as cv
import numpy as np
from config.local import *
import pickle
import uuid
import glob


def dump_database(database):
    with open(config["hist_database_path"], "wb") as file:
        pickle.dump(database, file)


def load_database():
    with open(config["hist_database_path"], "rb") as file:
        return pickle.load(file)


def save_frags_to_patterns_dir(fragments):
    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    for fragment in fragments:
        cv.imwrite(config["patterns_dataset_dir"] + "\\" + str(uuid.uuid4()) + ".jpg", fragment["img"])


def add_patterns_to_database(database: list):
    """Adds all patterns from patterns images directory to argument database"""

    # database is a list, each list's single item (record) structure is {"path", "hist"}
    for image_path in glob.glob(config["patterns_dataset_dir"] + "/*.jpg"):
        image = cv.imread(image_path)
        hist = calc_hist(image)
        database.append({"path": image_path, "hist": hist})
    return database


def calc_hist(image, mask=None):
    hist = cv.calcHist([cv.cvtColor(image, cv.COLOR_BGR2HSV)],
                       config["hist_channels"],
                       mask,
                       config["hist_size"],
                       config["hist_range"])
    return cv.normalize(hist, hist)


def find_most_unlike(query_hist, database):
    result = {"path": "Init value",
              "dist": 0}
    # bigger dist means that image is more unlike grass
    for record in database:
        dist = cv.compareHist(query_hist, record["hist"], config["hist_comp_method"])
        if config["lesser_dist_more_similar"]:
            if dist > result["dist"]:
                result["path"], result["dist"] = record["path"], dist
        else:
            if dist < result["dist"]:
                result["path"], result["dist"] = record["path"], dist
    return result


def _frag_process_columns(fragments: list, image, cur_end_y, last_col_processing):
    cur_start_y = cur_end_y - config["fragment_h"]

    for cur_end_x in range(config["aoi_left_border"] + config["fragment_w"],
                           config["aoi_right_border"],
                           config["fragment_x_offset"]):
        cur_start_x = cur_end_x - config["fragment_w"]
        fragments.append({
            "img": image[cur_start_y:cur_end_y, cur_start_x:cur_end_x],
            "start_x": cur_start_x,
            "start_y": cur_start_y,
            "end_x": cur_end_x,
            "end_y": cur_end_y
        })

    # if there remains a piece of the image (width) that is lesser than offset_x, we have to add it manually
    if last_col_processing:
        cur_start_x = config["aoi_right_border"] - config["fragment_w"]
        fragments.append({
            "img": image[cur_start_y:cur_end_y, cur_start_x:config["aoi_right_border"]],
            "start_x": cur_start_x,
            "start_y": cur_start_y,
            "end_x": config["aoi_right_border"],
            "end_y": cur_end_y
        })
    return fragments


def get_aoi_fragments(image):
    # flag if there remains a piece of the image that is less than the shift distance
    last_col_processing = config["aoi_right_border"] - config["aoi_left_border"] - config["fragment_w"] % config["fragment_x_offset"] != 0
    last_row_processing = config["aoi_bottom_border"] - config["aoi_top_border"] - config["fragment_h"] % config["fragment_y_offset"] != 0

    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    fragments = []
    # loop over image fragments (rows, cols)
    for cur_end_y in range(config["aoi_top_border"] + config["fragment_h"],
                           config["aoi_bottom_border"],
                           config["fragment_y_offset"]):
        _frag_process_columns(fragments, image, cur_end_y, last_col_processing)

    # if there remains a piece of the image (height) that is lesser than offset_y, we have to process that row manually
    if last_row_processing:
        cur_end_y = config["aoi_bottom_border"]
        _frag_process_columns(fragments, image, cur_end_y, last_col_processing)
    return fragments


def mark_fragment_on_img(image, fragment):
    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}

    # if we want to draw fragment itself (needs black background as arg image)
    # image[fragment["start_y"]:fragment["end_y"], fragment["start_x"]:fragment["end_x"]] = fragment["img"]

    # if we want to draw rectangle over fragment on source image
    image = cv.rectangle(image, (fragment["start_x"], fragment["start_y"]),
                         (fragment["end_x"], fragment["end_y"]), (255, 0, 0), 2)
    return image


def get_fragment_center_coords(fragment):
    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    x = fragment["start_x"] + config["fragment_w"] / 2
    y = fragment["start_y"] + config["fragment_h"] / 2
    return int(x), int(y)


def database_creator_mode():
    pass


def plant_searching_mode():
    pass


def main():
    # argprse!

    query_image = cv.imread(config["query_image_path"])

    # !!!
    query_h, query_w = query_image.shape[:2]
    print('Original Dimensions : ', query_image.shape)

    resized_h = 1200
    resized_w = int(resized_h * query_w / query_h)

    # resize image
    resized_image = cv.resize(query_image, (resized_w, resized_h), interpolation=cv.INTER_AREA)
    print('Resized Dimensions : ', resized_image.shape)

    # !!!
    weed_fragments = get_aoi_fragments()
    result_image = mark_fragment_on_img(resized_image, weed_fragments)
    print("Writing result image to", config["result"], "file.")
    cv.imwrite(config["output_image_dir"] +
                "result 1" +
                config["output_image_extension"], result_image)
    print("Done.")


if __name__ == "__main__":
    main()
