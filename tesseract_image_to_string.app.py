# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pytesseract
import sys


pytesseract.pytesseract.tesseract_cmd = r'tesseract'


PSM_00_NAME = 'Orientation and script detection (OSD) only.'
PSM_01_NAME = 'Automatic page segmentation with OSD.'
PSM_02_NAME = 'Automatic page segmentation, but no OSD, or OCR.'
PSM_03_NAME = 'Fully automatic page segmentation, but no OSD. (Default)'
PSM_04_NAME = 'Assume a single column of text of variable sizes.'
PSM_05_NAME = 'Assume a single uniform block of vertically aligned text.'
PSM_06_NAME = 'Assume a single uniform block of text.'
PSM_07_NAME = 'Treat the image as a single text line.'
PSM_08_NAME = 'Treat the image as a single word.'
PSM_09_NAME = 'Treat the image as a single word in a circle.'
PSM_10_NAME = 'Treat the image as a single character.'
PSM_11_NAME = 'Sparse text. Find as much text as possible in no particular order.'
PSM_12_NAME = 'Sparse text with OSD.'
PSM_13_NAME = 'Raw line. Treat the image as a single text line,\nbypassing hacks that are Tesseract-specific.'

PSM_NAME_TO_NUM = {
    PSM_00_NAME: 0,
    PSM_01_NAME: 1,
    PSM_02_NAME: 2,
    PSM_03_NAME: 3,
    PSM_04_NAME: 4,
    PSM_05_NAME: 5,
    PSM_06_NAME: 6,
    PSM_07_NAME: 7,
    PSM_08_NAME: 8,
    PSM_09_NAME: 9,
    PSM_10_NAME: 10,
    PSM_11_NAME: 11,
    PSM_12_NAME: 12,
    PSM_13_NAME: 13

}
PSM_NUM_TO_NAME = {v: k for k, v in PSM_NAME_TO_NUM.items()}

OEM_00_NAME = 'Legacy engine only.'
OEM_01_NAME = 'Neural nets LSTM engine only.'
OEM_02_NAME = 'Legacy + LSTM engines.'
OEM_03_NAME = 'Default, based on what is available.'

OEM_NAME_TO_NUM = {
    OEM_00_NAME: 0,
    OEM_01_NAME: 1,
    OEM_02_NAME: 2,
    OEM_03_NAME: 3
}
OEM_NUM_TO_NAME = {v: k for k, v in OEM_NAME_TO_NUM.items()}

lang = 'eng'
psm = 3
oem = 3
config = ''

def on_set(key, val):
    if key == "lang":
        global lang
        lang = val
    elif key == "psm":
        global psm
        # sys.stdout.write(f"[image_to_string] set psm: {psm}\n")
        # sys.stdout.flush()
        psm = PSM_NAME_TO_NUM[val]
    elif key == "oem":
        global oem
        # sys.stdout.write(f"[image_to_string] set oem: {oem}\n")
        # sys.stdout.flush()
        oem = OEM_NAME_TO_NUM[val]
    elif key == "config":
        global config
        config = val


def on_get(key):
    if key == "lang":
        return lang
    elif key == "psm":
        return PSM_NUM_TO_NAME[psm]
    elif key == "oem":
        return OEM_NUM_TO_NAME[oem]
    elif key == "config":
        return config


def on_init():
    return True


def on_valid():
    return True


def on_run(image: np.ndarray):
    global lang
    global oem
    global psm
    global config
    #sys.stdout.write(f"[findMaxLocRect] array : {array}")
    #sys.stdout.write(f"[findMaxLocRect] template : {template}")
    #sys.stdout.flush()

    assert len(image.shape) == 3
    assert image.shape[0] >= 1
    assert image.shape[1] >= 1
    assert image.shape[2] >= 1

    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result_str = pytesseract.image_to_string(img_rgb, lang= lang, config=f'--psm {psm} --oem {oem} {config}')

    # sys.stdout.write(f"[image_to_string] options : lang: {lang}, psm: {psm}, oem: {oem}, config: {config}\n")
    # sys.stdout.write(f"[image_to_string] result_str : {result_str}\n")
    # sys.stdout.flush()
    
    return {
        'string': np.array(w, np.int32),
        'h': np.array(h, np.int32),
        'wh': np.array([w, h], np.int32)
        }


def on_destroy():
    return True


if __name__ == '__main__':
    pass
