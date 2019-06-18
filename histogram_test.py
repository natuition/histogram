def performance_test():
    import timeit
    execution_time = timeit.repeat(get_weed_fragments, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))
