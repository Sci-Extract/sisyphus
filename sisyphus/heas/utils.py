from concurrent.futures import ThreadPoolExecutor, as_completed


def label_multi_threads(labeler, paras, args, workers):
    """label the paragraphs in parallel, paras and args should match one by one

    Args:
        labeler: the label function, which should be a dspy compatible caller
        paras: paragraphs
        args: the arguments for the labeler, e.g. [{filed_1: a, field_2: b}, (field_1: c, filed_2: d), ...]
        workers: the number of workers
    """
    with ThreadPoolExecutor(workers) as executor:
        futures = [executor.submit(labeler, **arg) for arg in args]
        future_para = dict(zip(futures, paras))
        para_result_tp = []
        for future in as_completed(futures):
            para_result_tp.append((future_para[future], future.result()))
    return para_result_tp