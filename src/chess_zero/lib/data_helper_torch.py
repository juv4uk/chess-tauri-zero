"""PyTorch-loop data helper -- direct port of lib/data_helper.py's
write_game_data_to_file/read_game_data_from_file/get_game_data_filenames.

Deliberately drops pretty_print (PGN export + pyperclip.copy): that's
for a human watching games live, not needed by the training loop, and
pyperclip fails outright in this headless environment.
"""
import json
import os
from glob import glob

PLAY_DATA_DIR = "../data/play_data_torch"  # relative to src/, matching every other torch script's ../data/... convention
PLAY_DATA_FILENAME_TMPL = "play_%s.json"


def write_game_data_to_file(path, data):
    with open(path, "wt") as f:
        json.dump(data, f)


def read_game_data_from_file(path):
    with open(path, "rt") as f:
        return json.load(f)


def get_game_data_filenames(data_dir=PLAY_DATA_DIR):
    pattern = os.path.join(data_dir, PLAY_DATA_FILENAME_TMPL % "*")
    return list(sorted(glob(pattern)))
