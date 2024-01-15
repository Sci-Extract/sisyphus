""""In development
Attention before running this script, making sure you have downloaded chrome browser(not the playwright chrome but the user own chrome browser)
"""

from sisyphus.crawler.download_si import downlaod
import asyncio
import time


if __name__ == "__main__":
    start = time.perf_counter()
    executable_path = "/the/path/to/your/chrome/execution"
    user_data_dir = "/the/path/to/your/chrome/data/file"
    asyncio.run(downlaod(executable_path, user_data_dir, "SI_download_failed.txt"))
    end = time.perf_counter()
    print(f"cost {end - start} s")