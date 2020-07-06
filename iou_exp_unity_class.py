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


class Category:
    """
    object category class
    """

    def __init__(self, class_cat_couple, global_df, working_dir, data_dir):
        self.class_cat_name = class_cat_couple[0]
        self.class_cat_id = class_cat_couple[1]
        self.global_df = global_df
        self.working_dir = working_dir
        self.data_dir = data_dir
        self.SMOOTH = 1e-6  # to avoid 0/0
        self.class_root = join(working_dir, data_dir, self.class_cat_name)
        self.DEBUG = False

    def build(self):
        self.frame_dir = os.path.join(self.class_root, "frames")
        # get the frames on which compute metrics
        self.frame_names = [f for f in listdir(self.frame_dir) if isfile(join(self.frame_dir, f))]
        # get the basename of each frame
        self.file_names = [Path(f).stem for f in self.frame_names]
        # list to manage batch of 5 frames
        supervisions = []
        predictions = []
        counter = 0

        for i in self.file_names:
            if counter == 5:
                break
            # load targets and predictions of the current frame
            sup, pred = self.__load_frame_file(i)
            supervisions.append(sup)
            predictions.append(pred)
            counter += 1

        # concatenate the list of frames sup/predictions in a single tensor
        supervisions_stacked = np.stack(supervisions, axis=0)
        predictions_stacked = np.stack(predictions, axis=0)
        # compute the bach iou
        iou, std = self.__iou(outputs=predictions_stacked, labels=supervisions_stacked)
        return iou, std

    def __load_frame_file(self, id):

        img = plt.imread(join(self.frame_dir, id + ".png"))
        self.h, self.w, self.c = img.shape
        if self.DEBUG:
            imgplot = plt.imshow(img)
            plt.show()
        try:
            # load category supervision
            indices, targets = self.__sup_loader(self.class_root, id)
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
            pred = self.__pred_loader(self.class_root, id)
            pred_class_binarized = pred[0, self.class_cat_id] > 0.5 # crete binary mask of prediction
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

        return target_resh_binarized, pred_class_binarized

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

    def __iou(self, outputs: np.array, labels: np.array):
        # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = (outputs & labels).astype(np.float).sum(
            (1, 2))  # zero if Truth=0 or Prediction=0 - sum over axis 1 and 2, the batch axes is manteined
        union = (outputs | labels).astype(np.float).sum((1, 2))  # zero if both are 0

        iou = (intersection + self.SMOOTH) / (union + self.SMOOTH)  # smooth division to avoid 0/0

        # checked with the sklearn metric => equivalent!
        # outp = outputs.reshape(-1)
        # labp = labels.reshape(-1)
        # jscsk = jsc(y_true=labp, y_pred=outp)

        return iou.mean(), iou.std()  #

    # TODO add bb_iou


if __name__ == '__main__':
    # dictionary of detected categories
    cat_dict = {65: 'bed', 84: 'book', 62: 'chair',
                63: 'couch', 67: 'dining_table', 48: 'fork', 73: 'laptop',
                5: 'airplane', 64: 'potted_plant', 75: 'remote', 50: 'spoon',
                43: 'tennis_racket', 70: 'toilet', 72: 'tv'}
    # invert the dictionary
    cat_dict_inv = {}
    for k, v in cat_dict.items():
        cat_dict_inv[v] = k

    working_dir = os.getcwd()
    data_dir = "dataset_unity"
    # create global dataframe to store results
    global_df = pd.DataFrame(columns=["Category", "mean_iou", "std_iou"])
    # create list of catagory objects
    objs = [Category(class_cat_couple=(k, v), global_df=global_df, working_dir=working_dir, data_dir=data_dir) for k, v
            in cat_dict_inv.items()]
    # execute performance computation
    for i in objs:
        iou, std = i.build()
        # append values to dataframe
        global_df = global_df.append(
            pd.DataFrame([[i.class_cat_name, iou, std]], columns=global_df.columns))

    global_df.reset_index(drop=True, inplace=True)
    global_df.to_csv("iou_results.csv", index=False)