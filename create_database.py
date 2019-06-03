from tarax.colordescriptor import ColorDescriptor
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


def split_image(image):
    # image = cv2.rectangle(image, (aoi_x_start, aoi_y_start), (aoi_x_end, aoi_y_end), (255, 0, 0), 2)
    # image = cv2.circle(image, (img_x_center, img_y_center), 5, (255, 0, 0), 2)

    img_y_center, img_x_center = int(image.shape[0] / 2), int(image.shape[1] / 2)  # y_c = 972, x_c = 1296

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
    aoi_image_1 = image[aoi_1_y_start:aoi_1_y_end, aoi_1_x_start:aoi_1_x_end]
    aoi_image_2 = image[aoi_2_y_start:aoi_2_y_end, aoi_2_x_start:aoi_2_x_end]
    aoi_image_3 = image[aoi_3_y_start:aoi_3_y_end, aoi_3_x_start:aoi_3_x_end]

    # draw AOIs on src img
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
                config["output_image_extension"], image)


def create_database():
    with open(config["index"], "w") as output_file:
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
    split_image(image)


if __name__ == "__main__":
    main()
