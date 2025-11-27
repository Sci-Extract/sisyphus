from script.file_name_doi_conversion import doi_to_file_name
import sys
import subprocess

arguments = sys.argv[1]

file_path = 'articles_processed/' + doi_to_file_name(arguments)
try:
    subprocess.run(["open", file_path], check=True)
except Exception as e:
    print(f"Failed to open file: {e}")