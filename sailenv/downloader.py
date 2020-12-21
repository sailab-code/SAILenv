#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini, Matteo Tiezzi
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# SAILenv is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages

import wget
import platform

BASE_URL = "http://eliza.diism.unisi.it/sailenv/"
BASE_EXECUTABLE_URL = BASE_URL + "bin/"
BASE_SOURCE_URL = BASE_URL + "source/"
os_names = {
    "Darwin": "OSX",
    "Windows": "Windows",
    "Linux": "Linux"
}
os_name = os_names[platform.system()]


def download_executable(version="latest"):
    url = f"{BASE_EXECUTABLE_URL}SAILenv_{os_name}-{version}.zip"
    print("Downloading from " + url)
    wget.download(url)


def download_source(version="latest"):
    url = f"{BASE_SOURCE_URL}SAILenv-{version}.zip"
    print("Downloading from " + url)
    wget.download(url)