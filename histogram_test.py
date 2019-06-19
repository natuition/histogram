#!/usr/bin/env python

from config.local import *
import histogram
import timeit


def performance_test():
    import timeit
    execution_time = timeit.repeat(get_weed_fragments, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))


def performance_test():
    execution_time = timeit.repeat(temp_perf_test_func_to_execute, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))


def performance_big_test():
    test_results = []

    ex_time_120_60 = timeit.repeat(temp_perf_test_func_to_execute, number=1, repeat=5, globals=globals())
    test_results.append(ex_time_120_60)

    config["fragment_w"] = 150
    config["fragment_h"] = 150
    config["fragment_x_offset"] = 75
    config["fragment_y_offset"] = 75
    ex_time_150_75 = timeit.repeat(temp_perf_test_func_to_execute, number=1, repeat=5, globals=globals())
    test_results.append(ex_time_150_75)

    config["fragment_w"] = 200
    config["fragment_h"] = 200
    config["fragment_x_offset"] = 100
    config["fragment_y_offset"] = 100
    ex_time_200_100 = timeit.repeat(temp_perf_test_func_to_execute, number=1, repeat=5, globals=globals())
    test_results.append(ex_time_200_100)

    config["fragment_w"] = 250
    config["fragment_h"] = 250
    config["fragment_x_offset"] = 125
    config["fragment_y_offset"] = 125
    ex_time_250_125 = timeit.repeat(temp_perf_test_func_to_execute, number=1, repeat=5, globals=globals())
    test_results.append(ex_time_250_125)

    print(test_results)

    with open("performance_test.txt", "w") as file:
        for result in test_results:
            file.write(str(result) + ", Min: " + min(result) + "\n")


def temp_perf_test_func_to_execute():
    aoi_areas = get_aoi_areas(test_image)
    fragments = []

    for area in aoi_areas:
        fragments.extend(get_fragments(area))

    # take max of min
    result_info = ["Init value", 0]
    result_frag = None
    i = 1
    for fragment in fragments:
        print("Processing fragment ", i, "of", len(fragments))
        i += 1
        histogram = cd.calc_hist(fragment)
        key, dist = sr.search_best(histogram)
        if dist > result_info[1]:
            result_info[0], result_info[1], result_frag = key, dist, fragment

    # uncomment if you want to save results on HDD
    # CAUTION! saving WILL slow down algorythm, dont use when measuring
    """
    cv2.imwrite(config["output_image_dir"] + "result_fragment.jpg", result_frag)
    with open(config["output_image_dir"] + "most not similar fragment with patterns DB.txt", "w") as file:
        data = "Path: " + result_info[0] + "\nDist: " + result_info[1]
        file.write(data)
    """
