import gc
import ctypes

def release_system_memory():
    """Forces Python garbage collection and trims the C glibc memory arena."""
    gc.collect()
    try:
        # Load standard C library on Linux
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
