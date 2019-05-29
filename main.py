from tarax.colordescriptor import ColorDescriptor
from tarax.searcher import Searcher
import cv2
import numpy as np

args = {}
args["shouw_debug"] = True
args["index"] = r"database\csv\database.csv"
args["dataset"] = r"database\images"
args["query"] = r"input\query 1.jpg"
args["result"] = r"output\result 1.jpg"
args["fragment_x_size"] = 70
args["fragment_y_size"] = 70
args["fragment_x_offset"] = int(args["fragment_x_size"] / 2)
args["fragment_y_offset"] = int(args["fragment_y_size"] / 2)
args["weed_keyword"] = "leaf"  # temporary, used to mark which fragments should be saved (name (key) comparing)

# initialize the image descriptor
print("Loading database...")
cd = ColorDescriptor((8, 12, 3))
sr = Searcher(args["index"])

# load query image
print("Loading query image...")
query_image = cv2.imread(args["query"])

query_h, query_w = query_image.shape[:2]
print('Original Dimensions : ', query_image.shape)

resized_h = 1200
resized_w = int(resized_h * query_w / query_h)

# resize image
resized_image = cv2.resize(query_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', resized_image.shape)

# if there will remain a piece of the image that is less than the shift distance flag
additional_w_proc = resized_w % args["fragment_x_offset"] != 0
additional_h_proc = resized_h % args["fragment_y_offset"] != 0
offsets_x_cnt = resized_w // args["fragment_x_offset"]
offsets_y_cnt = resized_h // args["fragment_y_offset"]
cur_start_x, cur_start_y, cur_end_x, cur_end_y = 0, 0, args["fragment_x_size"], args["fragment_y_size"]

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
        if args["weed_keyword"] in key:
            weed_fragments.append([cur_start_x, cur_start_y, cur_end_x, cur_end_y, key, distance])

        cur_start_x += args["fragment_x_offset"]
        cur_end_x += args["fragment_x_offset"]

    # if there remains a piece of the image (width)
    if additional_w_proc:
        # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
        features = cd.describe(resized_image[cur_start_y:cur_end_y, resized_w - args["fragment_x_size"]:resized_w])
        key, distance = sr.search_best(features)
        if args["weed_keyword"] in key:
            weed_fragments.append([resized_w - args["fragment_x_size"], cur_start_y, resized_w, cur_end_y, key, distance])

    # set x coords to start (first column) and set y coords to next row
    cur_start_x = 0
    cur_end_x = args["fragment_x_size"]
    cur_start_y += args["fragment_y_offset"]
    cur_end_y += args["fragment_y_offset"]

# if there remains a piece of the image (height)
if additional_h_proc:
    # set coords to last row (size_y - frag_y_size) and first column
    cur_start_x = 0
    cur_end_x = args["fragment_x_size"]
    cur_start_y = resized_h - args["fragment_y_size"]
    cur_end_y = resized_h

    print("Starting last column processing...")
    for column in range(offsets_x_cnt):  # cols
        print("Starting column:", column)

        features = cd.describe(resized_image[cur_start_y:cur_end_y, cur_start_x:cur_end_x])
        key, distance = sr.search_best(features)
        if args["weed_keyword"] in key:
            weed_fragments.append([cur_start_x, cur_start_y, cur_end_x, cur_end_y, key, distance])

        cur_start_x += args["fragment_x_offset"]
        cur_end_x += args["fragment_x_offset"]

    # if there remains a piece of the image (width)
    if additional_w_proc:
        # start_x = w - fragment_x_size, end_x = w (taking first right fragment of the current line)
        features = cd.describe(resized_image[cur_start_y:cur_end_y, resized_w - args["fragment_x_size"]:resized_w])
        key, distance = sr.search_best(features)
        if args["weed_keyword"] in key:
            weed_fragments.append([resized_w - args["fragment_x_size"], cur_start_y, resized_w, cur_end_y, key, distance])

# write weed image fragments on black blackground
print("Making result image with leafs only...")
result_image = np.zeros(resized_image.shape)
for record in weed_fragments:
    # record structure is [start_x, start_y, end_x, end_y, key, distance]
    start_x, start_y, end_x, end_y = record[:4]
    result_image[start_y:end_y, start_x:end_x] = resized_image[start_y:end_y, start_x:end_x]

print("Writing result image to", args["result"], "file.")
cv2.imwrite(args["result"], result_image)

print("Done.")
