# File to auto import the data files required from initial tests
# Usage: python import-data.py

import os, sys, tarfile
import urllib.request

CURRENT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__)))

root_url = "https://www.statmt.org/europarl/v7/"
fetch_files = [
    "es-en.tgz",
    "de-en.tgz",
    # "fr-en.tgz",
]


def download_file(url, file_path, file_name):
    print("Downloading " + file_name)
    urllib.request.urlretrieve(url,
                               os.path.join(CURRENT_DIR, file_path, file_name))
    print("Done.")


def decompress_file(file_path, file_name):
    print("Expanding " + file_name)
    try:
        tar = tarfile.open(os.path.join(CURRENT_DIR, file_path, file_name),
                           'r:gz')
        for item in tar:
            tar.extract(item, os.path.join(CURRENT_DIR, file_path))
        print('Done.')
    except:
        name = os.path.basename(file_name)
        print(name[:name.rfind('.')], '<filename>')


def main():
    output_folder = "data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    urls = list(map(lambda file_name: root_url + file_name, fetch_files))

    for url in urls:
        download_file(url, output_folder, url.split("/")[-1])
        decompress_file(output_folder, url.split("/")[-1])


if (__name__ == "__main__"):
    main()
