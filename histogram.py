#!/usr/bin/env python

from config.local import *
import cv2 as cv
import numpy as np
import pickle
import uuid
import glob
from connectors import SmoothieConnector

if config["use_camera"]:
    from picamera.array import PiRGBArray
    from picamera import PiCamera


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


def find_most_unlike_hist(query_hist, database: list):
    most_unlike = {"path": "Init value", "dist": 0}
    # bigger dist means that image is more unlike grass
    for record in database:
        dist = cv.compareHist(query_hist, record["hist"], config["hist_comp_method"])
        if config["lesser_dist_more_similar"]:
            if dist > most_unlike["dist"]:
                most_unlike["path"], most_unlike["dist"] = record["path"], dist
        else:
            if dist < most_unlike["dist"]:
                most_unlike["path"], most_unlike["dist"] = record["path"], dist
    return most_unlike


def find_most_unlike_frag(fragments: list, database: list):
    most_unlike_frag = None
    most_unlike_val = {"path": "Init value", "dist": 0}
    for fragment in fragments:
        cur_frag_val = find_most_unlike_hist(calc_hist(fragment["img"]), database)
        if config["lesser_dist_more_similar"]:
            if cur_frag_val["dist"] > most_unlike_val["dist"]:
                most_unlike_frag, most_unlike_val = fragment, cur_frag_val
        else:
            if cur_frag_val["dist"] < most_unlike_val["dist"]:
                most_unlike_frag, most_unlike_val = fragment, cur_frag_val
    return most_unlike_frag, most_unlike_val


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


def debug_mark_frag_on_img(image, fragment):
    """Draws rectangle over fragment (using fragment global coords) on source image"""

    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    image = cv.rectangle(image, (fragment["start_x"], fragment["start_y"]),
                         (fragment["end_x"], fragment["end_y"]), (255, 0, 0), 2)
    return image


def debug_draw_frag_on_img(image, fragment):
    """Draws fragment itself on the image (needs different from frag color background as arg image)"""

    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    image[fragment["start_y"]:fragment["end_y"], fragment["start_x"]:fragment["end_x"]] = fragment["img"]
    return image


def get_fragment_center_coords(fragment):
    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    x = fragment["start_x"] + config["fragment_w"] / 2
    y = fragment["start_y"] + config["fragment_h"] / 2
    return int(x), int(y)


def debug_image(image, fragment, x, y):
    image = debug_mark_frag_on_img(image, fragment)
    image = cv.circle(image, (x, y), 10, (255, 0, 0), thickness=2)
    cv.imwrite(config["output_image_dir"] + "fragment center result.jpg", image)


def run_database_mode():
    if config["use_camera"]:
        raise NotImplementedError("Camera usage code is not ready yet.")
    else:
        image = cv.imread(config["query_image_path"])

    fragments = get_aoi_fragments(image)
    save_frags_to_patterns_dir(fragments)
    try:
        database = load_database()
    except FileNotFoundError:
        database = []
    add_patterns_to_database(database)
    dump_database(database)


def run_searching_mode():
    if config["use_camera"]:
        raise NotImplementedError("Camera usage code is not ready yet.")
    else:
        image = cv.imread(config["query_image_path"])

    database = load_database()
    fragments = get_aoi_fragments(image)
    fragment, _ = find_most_unlike_frag(fragments, database)
    x, y = get_fragment_center_coords(fragment)

    # debug
    print(x, y)
    debug_image(image, fragment, x, y)


def main():
    if config["app_mode"] == "database":
        print("Starting in database creation mode.")
        run_database_mode()
    elif config["app_mode"] == "searching":
        print("Starting in searching mode.")
        run_searching_mode()
    else:
        print("Unknown mode error. Set correct mode in settings file .../config/local.py")
        exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
