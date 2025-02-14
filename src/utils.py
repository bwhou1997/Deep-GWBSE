
import time
import functools

class H5ls:
    def __init__(self):
        # Store an empty list for dataset names
        self.names = []

    def __call__(self, name, h5obj):
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(h5obj,'dtype') and not name in self.names:
            self.names += [name]


def time_watch(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"\n{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper