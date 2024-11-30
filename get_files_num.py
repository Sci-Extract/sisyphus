import os
import sys

def count_files(directory):
    try:
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <location>")
        sys.exit(1)

    directory = sys.argv[1]
    file_count = count_files(directory)

    if file_count is not None:
        print(f"Number of files in '{directory}': {file_count}")