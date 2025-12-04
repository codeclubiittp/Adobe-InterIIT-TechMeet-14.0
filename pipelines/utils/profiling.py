import time

class profiler:
    def __init__(self):
        self.start_time = None

    def begin(self):
        self.start_time = time.perf_counter()

    def end(self, message: str = "elapsed time") -> float:
        if self.start_time is None:
            print("profiler.begin() was not called")
            return 0.0
            
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        print(f"{message}: {elapsed:.4f}s")
        self.start_time = None 
        return elapsed