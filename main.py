from tarax.colordescriptor import ColorDescriptor
from tarax.searcher import Searcher
import cv2
import numpy as np

settings = {}
settings["shouw_debug"] = True
settings["index"] = r"database\csv\database.csv"
settings["dataset"] = r"database\images"
settings["query"] = r"input\query 1.jpg"
settings["result"] = r"output\result 1.jpg"
settings["fragment_x_size"] = 70
settings["fragment_y_size"] = 70
settings["fragment_x_offset"] = int(settings["fragment_x_size"] / 2)
settings["fragment_y_offset"] = int(settings["fragment_y_size"] / 2)
settings["weed_keyword"] = "leaf"  # temporary, used to mark which fragments should be saved (name (key) comparing)

# initialize the image descriptor
print("Loading database...")
cd = ColorDescriptor((8, 12, 3))
sr = Searcher(settings["index"])

# load query image
print("Loading query image...")
query_image = cv2.imread(settings["query"])

query_h, query_w = query_image.shape[:2]
print('Original Dimensions : ', query_image.shape)

resized_h = 1200
resized_w = int(resized_h * query_w / query_h)

# resize image
resized_image = cv2.resize(query_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', resized_image.shape)


def get_weed_fragments():
    # if there will remain a piece of the image that is less than the shift distance flag
    additional_w_proc = resized_w % settings["fragment_x_offset"] != 0
    additional_h_proc = resized_h % settings["fragment_y_offset"] != 0
    offsets_x_cnt = resized_w // settings["fragment_x_offset"]
    offsets_y_cnt = resized_h // settings["fragment_y_offset"]
    cur_start_x, cur_start_y, cur_end_x, cur_end_y = 0, 0, settings["fragment_x_size"], settings["fragment_y_size"]

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
            if settings["weed_keyword"] in key:
                weed_fragments.append([cur_start_x, cur_start_y, cur_end_x, cur_end_y, key, distance])

            cur_start_x += settings["fragment_x_offset"]
            cur_end_x += settings["fragment_x_offset"]

        # if there remains a piece of the image (width)
        if additional_w_proc:
            # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
            features = cd.describe(resized_image[cur_start_y:cur_end_y, resized_w - settings["fragment_x_size"]:resized_w])
            key, distance = sr.search_best(features)
            if settings["weed_keyword"] in key:
                weed_fragments.append([resized_w - settings["fragment_x_size"], cur_start_y, resized_w, cur_end_y, key, distance])

        # set x coords to start (first column) and set y coords to next row
        cur_start_x = 0
        cur_end_x = settings["fragment_x_size"]
        cur_start_y += settings["fragment_y_offset"]
        cur_end_y += settings["fragment_y_offset"]

    # if there remains a piece of the image (height)
    if additional_h_proc:
        # set coords to last row (size_y - frag_y_size) and first column
        cur_start_x = 0
        cur_end_x = settings["fragment_x_size"]
        cur_start_y = resized_h - settings["fragment_y_size"]
        cur_end_y = resized_h

        print("Starting last column processing...")
        for column in range(offsets_x_cnt):  # cols
            print("Starting column:", column)

            features = cd.describe(resized_image[cur_start_y:cur_end_y, cur_start_x:cur_end_x])
            key, distance = sr.search_best(features)
            if settings["weed_keyword"] in key:
                weed_fragments.append([cur_start_x, cur_start_y, cur_end_x, cur_end_y, key, distance])

            cur_start_x += settings["fragment_x_offset"]
            cur_end_x += settings["fragment_x_offset"]

        # if there remains a piece of the image (width)
        if additional_w_proc:
            # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
            features = cd.describe(resized_image[cur_start_y:cur_end_y, resized_w - settings["fragment_x_size"]:resized_w])
            key, distance = sr.search_best(features)
            if settings["weed_keyword"] in key:
                weed_fragments.append([resized_w - settings["fragment_x_size"], cur_start_y, resized_w, cur_end_y, key, distance])

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
    print("Writing result image to", settings["result"], "file.")
    cv2.imwrite(settings["result"], result_image)
    print("Done.")


def performance_test():
    import timeit
    execution_time = timeit.repeat(get_weed_fragments, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))


if __name__ == "__main__":
    performance_test()
    #main()
