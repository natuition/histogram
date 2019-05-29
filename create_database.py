from tarax.colordescriptor import ColorDescriptor
import glob
import cv2

args = {}
args["index"] = r"database\csv\database.csv"
args["dataset"] = r"database\images"

# initialize the color descriptor
# For our image search engine, we’ll be utilizing a 3D color histogram in the HSV color space with 8 bins for the Hue channel, 12 bins for the saturation channel, and 3 bins for the value channel, yielding a total feature vector of dimension 8 x 12 x 3 = 288.
# This means that for every image in our dataset, no matter if the image is 36 x 36 pixels or 2000 x 1800 pixels, all images will be abstractly represented and quantified using only a list of 288 floating point numbers.
# https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
cd = ColorDescriptor((8, 12, 3))

with open(args["index"], "w") as output_file:
    # use glob to grab the image paths and loop over them
    for image_path in glob.glob(args["dataset"] + "/*.jpg"):  # or "/*.png"
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        image_unique_name = image_path[image_path.rfind("/") + 1:]
        image = cv2.imread(image_path)

        # make image description and convert to str
        features = [str(f) for f in cd.describe(image)]
        output_file.write("%s,%s\n" % (image_unique_name, ",".join(features)))
