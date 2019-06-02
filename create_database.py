from tarax.colordescriptor import ColorDescriptor
import glob
import cv2

args = {}
args["index"] = r"database\csv\database.csv"
args["dataset"] = r"database\images"
args["query_image_path"] = r"input\prise2.jpg"
args["x_view_size"] = 880
args["y_view_size"] = 750
args["output_image_path"] = r"output\prise2 - AOI.jpg"
args["output_image_path_full"] = r"output\prise2 - AOI Full.jpg"

# initialize the color descriptor
# For our image search engine, we’ll be utilizing a 3D color histogram in the HSV color space with 8 bins for the Hue channel, 12 bins for the saturation channel, and 3 bins for the value channel, yielding a total feature vector of dimension 8 x 12 x 3 = 288.
# This means that for every image in our dataset, no matter if the image is 36 x 36 pixels or 2000 x 1800 pixels, all images will be abstractly represented and quantified using only a list of 288 floating point numbers.
# https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
cd = ColorDescriptor((8, 12, 3))


def split_image():
    image = cv2.imread(args["query_image_path"])
    y_center, x_center = int(image.shape[0] / 2), int(image.shape[1] / 2)
    image = cv2.circle(image, (x_center, y_center), 5, (255, 0, 0), 2)

    # area of interest
    aoi_x_start, aoi_y_start = x_center - int(args["x_view_size"] / 2), y_center - int(args["y_view_size"] / 2)
    aoi_x_end, aoi_y_end = aoi_x_start + args["x_view_size"], aoi_y_start + args["y_view_size"]
    aoi_image = image[aoi_y_start:aoi_y_end, aoi_x_start:aoi_x_end]
    image = cv2.rectangle(image, (aoi_x_start, aoi_y_start), (aoi_x_end, aoi_y_end), (255, 0, 0), 2)

    cv2.imwrite(args["output_image_path"], aoi_image)
    cv2.imwrite(args["output_image_path_full"], image)


def create_database():
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


def main():
    split_image()


if __name__ == "__main__":
    main()
