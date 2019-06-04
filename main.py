from tarax.colordescriptor import ColorDescriptor
from tarax.searcher import Searcher
import cv2
import numpy as np
import json
import create_database

# load config
with open("config/config.json", "r") as config_file:
    config = json.loads(config_file.read())

# initialize the image descriptor
print("Loading database...")
cd = ColorDescriptor((8, 12, 3))
sr = Searcher(config["hist_database_path"])

# load query image
print("Loading query image...")
query_image = cv2.imread(config["query_image_path"])

query_h, query_w = query_image.shape[:2]
print('Original Dimensions : ', query_image.shape)

resized_h = 1200
resized_w = int(resized_h * query_w / query_h)

# resize image
resized_image = cv2.resize(query_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', resized_image.shape)


def get_weed_fragments():
    # if there will remain a piece of the image that is less than the shift distance flag
    additional_w_proc = resized_w % config["fragment_x_offset"] != 0
    additional_h_proc = resized_h % config["fragment_y_offset"] != 0
    offsets_x_cnt = resized_w // config["fragment_x_offset"] - 1
    offsets_y_cnt = resized_h // config["fragment_y_offset"] - 1
    cur_start_x, cur_start_y, cur_end_x, cur_end_y = 0, 0, config["fragment_w"], config["fragment_h"]

    print("Last line processing:", additional_h_proc)
    print("Last column processing:", additional_w_proc)

    # loop over image fragments
    print("Starting loop over fragments...")
    weed_fragments = []  # structure is [[start_x, start_y, end_x, end_y, key, distance]] (nested list)
    for line in range(offsets_y_cnt):  # rows
        print("Starting line:", line)

        for column in range(offsets_x_cnt):  # cols
            print("Starting column:", column)

            features = cd.describe(resized_image[cur_start_y:cur_end_y, cur_start_x:cur_end_x])
            key, distance = sr.search_best(features)
            if config["weed_keyword"] in key:
                weed_fragments.append([cur_start_x, cur_start_y, cur_end_x, cur_end_y, key, distance])

            cur_start_x += config["fragment_x_offset"]
            cur_end_x += config["fragment_x_offset"]

        # if there remains a piece of the image (width)
        if additional_w_proc:
            # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
            features = cd.describe(resized_image[cur_start_y:cur_end_y, resized_w - config["fragment_w"]:resized_w])
            key, distance = sr.search_best(features)
            if config["weed_keyword"] in key:
                weed_fragments.append([resized_w - config["fragment_w"], cur_start_y, resized_w, cur_end_y, key, distance])

        # set x coords to start (first column) and set y coords to next row
        cur_start_x = 0
        cur_end_x = config["fragment_w"]
        cur_start_y += config["fragment_y_offset"]
        cur_end_y += config["fragment_y_offset"]

    # if there remains a piece of the image (height)
    if additional_h_proc:
        # set coords to last row (size_y - frag_y_size) and first column
        cur_start_x = 0
        cur_end_x = config["fragment_w"]
        cur_start_y = resized_h - config["fragment_h"]
        cur_end_y = resized_h

        print("Starting last column processing...")
        for column in range(offsets_x_cnt):  # cols
            print("Starting column:", column)

            features = cd.describe(resized_image[cur_start_y:cur_end_y, cur_start_x:cur_end_x])
            key, distance = sr.search_best(features)
            if config["weed_keyword"] in key:
                weed_fragments.append([cur_start_x, cur_start_y, cur_end_x, cur_end_y, key, distance])

            cur_start_x += config["fragment_x_offset"]
            cur_end_x += config["fragment_x_offset"]

        # if there remains a piece of the image (width)
        if additional_w_proc:
            # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
            features = cd.describe(resized_image[cur_start_y:cur_end_y, resized_w - config["fragment_w"]:resized_w])
            key, distance = sr.search_best(features)
            if config["weed_keyword"] in key:
                weed_fragments.append([resized_w - config["fragment_w"], cur_start_y, resized_w, cur_end_y, key, distance])

    return weed_fragments


def draw_fragments_on_img(sourse_image, weed_fragments):
    # write weed image fragments on black blackground
    print("Making result image with leafs only...")
    result_image = np.zeros(sourse_image.shape)

    for record in weed_fragments:
        # record structure is [start_x, start_y, end_x, end_y, key, distance]
        start_x, start_y, end_x, end_y = record[:4]
        result_image[start_y:end_y, start_x:end_x] = sourse_image[start_y:end_y, start_x:end_x]

    return result_image


def main():
    weed_fragments = get_weed_fragments()
    result_image = draw_fragments_on_img(resized_image, weed_fragments)
    print("Writing result image to", config["result"], "file.")
    cv2.imwrite(config["output_image_dir"] +
                "result 1" +
                config["output_image_extension"], result_image)
    print("Done.")


def performance_test():
    import timeit
    execution_time = timeit.repeat(get_weed_fragments, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))


if __name__ == "__main__":
    #performance_test()
    main()
