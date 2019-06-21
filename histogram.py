#!/usr/bin/env python

from config.local import *
import cv2 as cv
import numpy as np
import pickle
import uuid
import glob
from connectors import SmoothieConnector
import time
import json
from multiprocessing import Value
import sys
import os

if config["use_camera"]:
    from picamera.array import PiRGBArray
    from picamera import PiCamera

CONFIG_LOCAL_PATH = "config/config_local.json"
NOT_SENT_MSG = "g-code wasn't sent to smoothie."

with open(CONFIG_LOCAL_PATH, "r") as file:
    config_ws_local = json.loads(file.read())

x_current = Value('i', 0)
y_current = Value('i', 0)
z_current = Value('i', 100)


def dump_database(database):
    with open(config["hist_database_path"], "wb") as file:
        pickle.dump(database, file)


def load_database():
    with open(config["hist_database_path"], "rb") as file:
        return pickle.load(file)


def save_frags_to_patterns_dir(fragments):
    # single fragment structure is {"img", "start_x", "start_y", "end_x", "end_y"}
    for fragment in fragments:
        sep = "/" if sys.version_info.minor == 5 else "\\"
        cv.imwrite(config["patterns_dataset_dir"] + sep + str(uuid.uuid4()) + ".jpg", fragment["img"])


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
    sep = "/" if sys.version_info.minor == 5 else "\\"
    output_full_path = config["output_image_dir"] + config["query_image_path"].split(sep)[-1] + " - result.jpg" \
        if config["use_camera"] \
        else "Image from camera - result.jpg"
    if not os.path.exists(config["output_image_dir"]):
        os.makedirs(config["output_image_dir"])
    cv.imwrite(output_full_path, image)


def read_until_contains(pattern, smc: SmoothieConnector):
    while True:
        response = smc.receive()
        if pattern in response:
            return response


def read_until_not(value, smc: SmoothieConnector):
    while True:
        response = smc.receive()
        if response != value:
            return response


def calibrate_axis(axis_current: Value, axis_label, axis_min_key, axis_max_key, smc: SmoothieConnector):
    distanse = 1000
    with axis_current.get_lock():
        if config_ws_local["{0}_AXIS_CALIBRATION_TO_MAX".format(axis_label)]:
            if not config_ws_local["USE_SMOOTHIE_CONNECTION_SIMULATION"]:
                smc.send("G28 {0}{1}".format(axis_label, distanse))
                read_until_contains("ok", smc)
            axis_current.value = config_ws_local[axis_max_key] - config_ws_local["AFTER_CALIBRATION_AXIS_OFFSET"]
        else:
            if not config_ws_local["USE_SMOOTHIE_CONNECTION_SIMULATION"]:
                smc.send("G28 {0}{1}".format(axis_label, -distanse))
                read_until_contains("ok", smc)
            axis_current.value = config_ws_local[axis_min_key] + config_ws_local["AFTER_CALIBRATION_AXIS_OFFSET"]

        # set fresh current coordinates on smoothie too
        if not config_ws_local["USE_SMOOTHIE_CONNECTION_SIMULATION"]:
            smc.send("G92 {0}{1}".format(axis_label, axis_current.value))
            read_until_contains("ok", smc)


def corkscrew_to_start_pos(smc: SmoothieConnector):
    # X axis calibration
    if config_ws_local["USE_X_AXIS_CALIBRATION"]:
        print("X axis calibration...", end=" ")
        calibrate_axis(x_current, "X", "X_MIN", "X_MAX", smc)
        print("Ok")

    # Y axis calibration
    if config_ws_local["USE_Y_AXIS_CALIBRATION"]:
        print("Y axis calibration...", end=" ")
        calibrate_axis(y_current, "Y", "Y_MIN", "Y_MAX", smc)
        print("Ok")

    # Z axis calibration
    if config_ws_local["USE_Z_AXIS_CALIBRATION"]:
        print("Z axis calibration...", end=" ")
        calibrate_axis(z_current, "Z", "Z_MIN", "Z_MAX", smc)
        print("Ok")


def switch_to_relative(smc: SmoothieConnector):
    smc.send("G91")
    return read_until_not(">", smc)


def validate_moving_key(value, key_name, key_min, key_max, key_min_name, key_max_name, current_value):
    """For F current_value must be 0"""

    if current_value + value > key_max:
        return "Command with {0}{1} goes beyond max acceptable range of {2} = {3}, " \
                   .format(key_name, value, key_max_name, key_max) + NOT_SENT_MSG
    if current_value + value < key_min:
        return "Command with {0}{1} goes beyond min acceptable range of {2} = {3}, " \
                   .format(key_name, value, key_min_name, key_min) + NOT_SENT_MSG
    return None


def move_cork_to_center(smc: SmoothieConnector):
    with x_current.getlock():
        # calc cork center coords and xy movement values for smoothie g-code
        center_x, center_y = int(config_ws_local["X_MAX"] / 2), int(config_ws_local["Y_MAX"] / 2)
        smc_x, smc_y = int(abs(x_current.value - center_x)), int(abs(y_current.value - center_y))
        if x_current.value > config["cork_center_x"]:
            smc_x = -smc_x
        if y_current.value > config["cork_center_y"]:
            smc_y = -smc_y
        x_current.value += smc_x
        y_current.value += smc_y
        g_code = "G0 X" + str(smc_x) + " Y" + str(smc_y) + " F" + str(config["smoothie_xy_F"])
        smc.send(g_code)
        read_until_contains("ok", smc)


def run_database_mode():
    if not config_ws_local["USE_SMOOTHIE_CONNECTION_SIMULATION"]:
        smc = SmoothieConnector(config["smoothie_host"], verbose=True)
        smc.connect()
        switch_to_relative(smc)
        corkscrew_to_start_pos(smc)
        move_cork_to_center(smc)

    if config["use_camera"]:
        print("Taking image...")
        camera = PiCamera()
        camera.resolution = (config["camera_w"], config["camera_h"])
        camera.framerate = config["camera_framerate"]
        raw_capture = PiRGBArray(camera, size=(config["camera_w"], config["camera_h"]))
        time.sleep(1)

        for frame in camera.capture_continuous(raw_capture, format="rgb"):  # bgr flag also returns rgb o_0
            image = cv.cvtColor(frame.array, cv.COLOR_RGB2BGR)
            raw_capture.truncate(0)
            # taking only single frame for now
            break
    else:
        print("Loading ", config["query_image_path"])
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
    if not config_ws_local["USE_SMOOTHIE_CONNECTION_SIMULATION"]:
        smc = SmoothieConnector(config["smoothie_host"], verbose=True)
        smc.connect()
        switch_to_relative(smc)
        corkscrew_to_start_pos(smc)
        move_cork_to_center(smc)

    if config["use_camera"]:
        print("Taking image...")
        camera = PiCamera()
        camera.resolution = (config["camera_w"], config["camera_h"])
        camera.framerate = config["camera_framerate"]
        raw_capture = PiRGBArray(camera, size=(config["camera_w"], config["camera_h"]))
        time.sleep(1)

        for frame in camera.capture_continuous(raw_capture, format="rgb"):  # bgr flag also returns rgb o_0
            image = cv.cvtColor(frame.array, cv.COLOR_RGB2BGR)
            raw_capture.truncate(0)
            # taking only single frame for now
            break
    else:
        print("Loading ", config["query_image_path"])
        image = cv.imread(config["query_image_path"])

    database = load_database()
    fragments = get_aoi_fragments(image)
    fragment, frag_path_dist = find_most_unlike_frag(fragments, database)
    px_x, px_y = get_fragment_center_coords(fragment)  # pixel coords

    # debug
    print("Pixel center coordinates:", px_x, px_y)
    debug_image(image, fragment, px_x, px_y)

    # smoothie
    if not config_ws_local["USE_SMOOTHIE_CONNECTION_SIMULATION"]:
        # calc moving x coordinate (convert px in mm for g-code)
        sm_x = int(abs(px_x - config["cork_center_x"]) / config["one_mm_in_px"])
        if px_x < config["cork_center_x"]:
            sm_x = -sm_x
        # calc moving y coordinate (convert px in mm for g-code)
        # y is always positive as cork y cur coord always under image AOI area
        sm_y = int(abs(px_y - config["cork_center_y"]) / config["one_mm_in_px"])

        # validate coordinates
        error_msg_x = validate_moving_key(sm_x, "X", config_ws_local["X_MIN"], config_ws_local["X_MAX"], "X_MIN",
                                          "X_MAX", x_current.value)
        error_msg_y = validate_moving_key(sm_y, "Y", config_ws_local["Y_MIN"], config_ws_local["Y_MAX"], "Y_MIN",
                                          "Y_MAX", y_current.value)
        error_msg_z = validate_moving_key(config["extraction_z"], "Z", config_ws_local["Z_MIN"],
                                          config_ws_local["Z_MAX"], "Z_MIN", "Z_MAX", z_current.value)
        if not error_msg_x:
            print(error_msg_x)
            return
        if not error_msg_y:
            print(error_msg_y)
            return
        if not error_msg_z:
            print(error_msg_z)
            return

        # moving xy
        print("Moving to X", sm_x, "Y", sm_y, "with F", config["smoothie_xy_f"])
        with x_current.get_lock():
            g_code = "G0 X" + str(sm_x) + " Y" + str(sm_y) + " F" + str(config["smoothie_xy_F"])
            smc.send(g_code)
            response = read_until_not(">", smc)
            print(response)
            x_current.value += sm_x
            y_current.value += sm_y

        # + или -Z для извлечения и возврата на место, значение Z (в веб серв конфиге мин макс Z = 0, 52)
        # extracting (z)
        print("Extracting, lift down cork with Z", -config["extraction_z"])
        with z_current.get_lock():
            g_code = "G0 Z" + str(-config["extraction_z"]) + " F" + str(config["smoothie_z_F"])
            smc.send(g_code)
            response = read_until_not(">", smc)
            print("Response:", response)
            z_current.value += config["extraction_z"]

        print("Extracting, lift up cork with Z", config["extraction_z"])
        with z_current.get_lock():
            g_code = "G0 Z" + str(config["extraction_z"]) + " F" + str(config["smoothie_z_F"])
            smc.send(g_code)
            response = read_until_not(">", smc)
            print("Response:", response)
            z_current.value += config["extraction_z"]


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
