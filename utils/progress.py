from tqdm import tqdm


def progress_bar(iterable, desc=None, total=None):
    return tqdm(iterable, desc=desc, total=total)
