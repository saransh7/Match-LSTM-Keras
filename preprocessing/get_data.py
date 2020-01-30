"""
Downloads Squad data
"""
import os
import sys
import json
import config as c
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_data(url, filename, num_bytes=None):
    local_filename = None
    if not os.path.exists(os.path.join(c.data_dir, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(
                    url + filename, os.path.join(c.data_dir, filename), reporthook=reporthook(t))
        except AttributeError as e:
            print(
                "An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # file already downloaded
    file_stats = os.stat(os.path.join(c.data_dir, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully loaded".format(filename))
    else:
        raise Exception(
            "Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename


def main():
    print('Downloading Data..')

    if not os.path.exists(c.data_dir):
        os.makedirs(c.data_dir)

    download_data(c.SQUAD_BASE_URL, c.train_filename)
    download_data(c.SQUAD_BASE_URL, c.dev_filename)
    download_data(c.glove_base_url, c.glove_filename)


if __name__ == "__main__":
    main()
