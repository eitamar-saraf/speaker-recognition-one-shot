import rootpath
import os

ROOT = rootpath.detect()


def join_path(path, *paths):
    return os.path.join(path, *paths)
