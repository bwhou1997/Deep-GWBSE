
import time
import functools
import tracemalloc

eV2Ry = 0.073498618
Ry2eV = 13.6056980659

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
        print(f"{func.__name__} time used: {end_time - start_time:.6f} seconds")
        return result
    return wrapper


def memory_watch(top_n=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            result = func(*args, **kwargs)
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            # print(f"[ Top {top_n} Memory]")
            total = 0
            for stat in top_stats:
                total += stat.size / 1024 / 1024 / 1024
            print(f"\n{func.__name__} memory used: {total:.2f} GB")
            return result
        return wrapper
    return decorator