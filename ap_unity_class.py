import numpy as np
import torch
import os
from os import listdir
from gzip import GzipFile
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score as jsc
import pandas as pd
import cv2
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import label
import matplotlib.patches as patches


class Cat_data:
    """
    object category class acting on all the dataset
    """

    def __init__(self, class_cat_couple, working_dir, data_dir, gt, det, cat_dict):
        self.class_cat_name = class_cat_couple[0]
        self.class_cat_id = class_cat_couple[1]
        self.working_dir = working_dir
        self.data_dir = data_dir
        self.gt = gt
        self.det = det
        self.SMOOTH = 1e-6  # to avoid 0/0
        self.DEBUG = False
        self.cat_dict = cat_dict

    def build(self):
        frame_counter = 0  # used as name file for all dataset
        for folder_name in self.cat_dict:  # explore all the directories
            class_root = join(self.working_dir, self.data_dir, folder_name)
            frame_dir = os.path.join(class_root, "frames")
            # get the frames on which compute metrics
            frame_names = [f for f in listdir(frame_dir) if isfile(join(frame_dir, f))]
            # get the basename of each frame
            file_names = [Path(f).stem for f in
                          frame_names]  # TODO generalze these to get all frames from all classes
            # list to manage batch of 5 frames
            counter = 0
            for i in file_names:
                if counter == 5:
                    break
                # load targets and predictions of the current frame
                sup_bin, pred_bin, confidence = self.__load_frame_file(i, frame_dir, class_root)
                self.__detect_and_log_bb(sup_bin, self.gt, name=f"{frame_counter}")
                self.__detect_and_log_bb(pred_bin, self.det, name=f"{frame_counter}", confidence=confidence)
                counter += 1
                frame_counter += 1

        # concatenate the list of frames sup/predictions in a single tensor

    def __load_frame_file(self, id,  frame_dir, class_root):

        img = plt.imread(join(frame_dir, id + ".png"))
        self.h, self.w, self.c = img.shape
        if self.DEBUG:
            imgplot = plt.imshow(img)
            plt.show()
        try:
            # load category supervision
            indices, targets = self.__sup_loader(class_root, id)
            target_resh = targets.reshape(self.h, self.w)
            # zeroing all pixels not beloning to current class
            # crete binary mask of the object
            target_resh_binarized = target_resh == self.class_cat_id

            if self.DEBUG:
                imgplot2 = plt.imshow(target_resh_binarized.astype(np.float))
                plt.show()
        except Exception as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly
            print(f"Exception in supervisions of category {self.class_cat_name} and file {id}")
            pass

        try:
            # now load prediction
            pred = self.__pred_loader(class_root, id)
            pred_class_binarized = pred[0, self.class_cat_id] > 0.0005  # crete binary mask of prediction
            confidence = np.max(
                pred[0, self.class_cat_id])
            if confidence > 1.0:
                confidence = 1.0 # max because the predictions are 0 or a value, two possibilities
            # elaborated maskrnn output shape: 1 x 91 x H x W => extract H x W, only the desidered class

            if self.DEBUG:
                pred_plot = plt.imshow(pred_class_binarized.astype(np.float))
                plt.show()
        except Exception as inst:
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly
            print(f"Exception in predictions of category {self.class_cat_name} and file {id}")
            pass

        return target_resh_binarized, pred_class_binarized, confidence

    def __sup_loader(self, path, file):
        with GzipFile(join(path, "sup", file + ".indices.bin")) as f:
            indices = np.load(f)
        with GzipFile(join(path, "sup", file + ".targets.bin")) as f:
            targets = np.load(f)
        return indices, targets

    def __pred_loader(self, path, file):
        with GzipFile(join(path, "predictions", file + ".bin")) as f:
            pred = np.load(f)
        return pred

    def __detect_and_log_bb(self, img_binarized, folder, name, confidence=None):  # name class_index.txt

        lbl = label(img_binarized)
        props = regionprops(lbl)  # returns bbox in (min_row, min_col, max_row, max_col)
        if self.DEBUG:
            fig, ax = plt.subplots(1)
            ax.imshow(img_binarized.astype(np.float))

        good_props = [i for i in props if i.area > 100]
        props = good_props
        for prop in props:
            with open(os.path.join(folder, name + ".txt"), "a") as f:  #
                if confidence is None:  # it is a supervision
                    f.write(
                        f"{self.class_cat_name} {prop.bbox[0]} {prop.bbox[1]} {prop.bbox[2]} {prop.bbox[3]}\n")  # check order
                else:
                    f.write(
                        f"{self.class_cat_name} {confidence} {prop.bbox[0]} {prop.bbox[1]} {prop.bbox[2]} {prop.bbox[3]}\n")


if __name__ == '__main__':
    cat_dict = {65: 'bed', 84: 'book', 62: 'chair',
                63: 'couch', 67: 'dining_table', 48: 'fork', 73: 'laptop',
                5: 'airplane', 64: 'potted_plant', 75: 'remote', 50: 'spoon',
                43: 'tennis_racket', 70: 'toilet', 72: 'tv'}
    working_dir = os.getcwd()

    cat_dict_inv = {}
    for k, v in cat_dict.items():
        cat_dict_inv[v] = k

    list_cat = [k for k, v in cat_dict_inv.items()]

    data_dir = "dataset_unity"
    gt = "groundtruths"
    det = "detections"
    # create input dirs
    if not os.path.exists(gt):
        os.makedirs(gt)
    if not os.path.exists(det):
        os.makedirs(det)

    # create list of catagory objects
    objs = [
        Cat_data(class_cat_couple=(k, v), working_dir=working_dir, data_dir=data_dir, gt=gt, det=det, cat_dict=list_cat)
        for k, v
        in cat_dict_inv.items()]
    # execute performance computation
    for i in objs:
        i.build()
