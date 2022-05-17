import csv
import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Dict

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
import numpy as np
from math import sin, cos


def computeBox3D(label, P):
    '''
    takes an object label and a projection matrix (P) and projects the 3D
    bounding box into the image plane.

    (Adapted from devkit_object/matlab/computeBox3D.m)

    Args:
      label -  object label list or array
    '''
    w = label[0]
    h = label[1]
    l = label[2]
    x = label[3]
    y = label[4]
    z = label[5]
    ry = label[6]

    # w = label["dimensions"][0]
    # h = label["dimensions"][1]
    # l = label["dimensions"][2]
    # x = label["location"][0]
    # y = label["location"][1]
    # z = label["location"][2]
    # ry = label["rotation_y"]

    # compute rotational matrix around yaw axis
    R = np.array([[+cos(ry), 0, +sin(ry)],
                  [0, 1, 0],
                  [-sin(ry), 0, +cos(ry)]])

    # 3D bounding box corners

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # --w/2

    # x_corners += -l / 2
    # y_corners += -h
    # z_corners += -w / 2
    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    # bounding box in object co-ordinate
    corners_3D = np.array([x_corners, y_corners, z_corners])
    # print ( 'corners_3d', corners_3D.shape, corners_3D)

    # rotate
    corners_3D = R.dot(corners_3D)
    # print ( 'corners_3d', corners_3D.shape, corners_3D)

    # translate
    corners_3D += np.array([x, y, z]).reshape((3, 1))
    # print ( 'corners_3d', corners_3D)

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    # edges, lines 3d/2d bounding box in vertex index
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [0, 5], [1, 4], [2, 7], [3, 6]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [0, 5], [5, 4], [4, 1], [1, 2], [2, 7],
             [7, 6], [6, 3]]
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    bb2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]  #

    corners_2D = corners_2D[:2]
    base_indices = [0, 1, 4, 5]
    base_3Dto2D = corners_2D[:, base_indices]

    return base_3Dto2D, corners_2D, corners_3D, bb2d_lines_verts[:2]


def labelToBoundingBox(ax, labeld, calibd):
    '''
    Draw 2D and 3D bpunding boxes.

    Each label  file contains the following ( copied from devkit_object/matlab/readLabels.m)
    #  % extract label, truncation, occlusion
    #  lbl = C{1}(o);                   % for converting: cell -> string
    #  objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...
    #  objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
    #  objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
    #  objects(o).alpha      = C{4}(o); % object observation angle ([-pi..pi])
    #
    #  % extract 2D bounding box in 0-based coordinates
    #  objects(o).x1 = C{5}(o); % left   -> in pixel
    #  objects(o).y1 = C{6}(o); % top
    #  objects(o).x2 = C{7}(o); % right
    #  objects(o).y2 = C{8}(o); % bottom
    #
    #  % extract 3D bounding box information
    #  objects(o).h    = C{9} (o); % box width    -> in object coordinate
    #  objects(o).w    = C{10}(o); % box height
    #  objects(o).l    = C{11}(o); % box length
    #  objects(o).t(1) = C{12}(o); % location (x) -> in camera coordinate
    #  objects(o).t(2) = C{13}(o); % location (y)
    #  objects(o).t(3) = C{14}(o); % location (z)
    #  objects(o).ry   = C{15}(o); % yaw angle  -> rotation aroun the y/vetical axis
    '''

    # Velodyne to/from referenece camera (0) matrix
    Tr_velo_to_cam = np.zeros((4, 4))
    Tr_velo_to_cam[3, 3] = 1
    Tr_velo_to_cam[:3, :4] = calibd['Tr_velo_to_cam'].reshape(3, 4)
    # print ('Tr_velo_to_cam', Tr_velo_to_cam)

    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    # print ('Tr_cam_to_velo', Tr_cam_to_velo)

    #
    R0_rect = np.zeros((4, 4))
    R0_rect[:3, :3] = calibd['R0_rect'].reshape(3, 3)
    R0_rect[3, 3] = 1
    # print ('R0_rect', R0_rect)
    P2_rect = calibd['P2'].reshape(3, 4)
    # print('P2_rect', P2_rect)

    bb3d = []
    bb2d = []

    for key in labeld.keys():

        color = 'white'
        if key == 'Car':
            color = 'red'
        elif key == 'Pedestrian':
            color = 'pink'
        elif key == 'Cyclist':
            color = 'purple'
        elif key == 'DontCare':
            color = 'white'

        for o in range(labeld[key].shape[0]):

            # 2D
            left = labeld[key][o][3]
            bottom = labeld[key][o][4]
            width = labeld[key][o][5] - labeld[key][o][3]
            height = labeld[key][o][6] - labeld[key][o][4]

            p = patches.Rectangle(
                (left, bottom), width, height, fill=False, edgecolor=color, linewidth=1)
            ax.add_patch(p)

            xc = (labeld[key][o][5] + labeld[key][o][3]) / 2
            yc = (labeld[key][o][6] + labeld[key][o][4]) / 2
            bb2d.append([xc, yc])

            # 3D
            w3d = labeld[key][o][7]
            h3d = labeld[key][o][8]
            l3d = labeld[key][o][9]
            x3d = labeld[key][o][10]
            y3d = labeld[key][o][11]
            z3d = labeld[key][o][12]
            yaw3d = labeld[key][o][13]

            if key != 'DontCare':
                base_3Dto2D, corners_2D, corners_3D, paths_2D = computeBox3D(labeld[key][o], P2_rect)
                verts = paths_2D.T  # corners_2D.T
                codes = [Path.LINETO] * verts.shape[0]
                codes[0] = Path.MOVETO
                pth = Path(verts, codes)
                p = patches.PathPatch(pth, fill=False, color='purple', linewidth=2)
                ax.add_patch(p)

    # a sanity test point in velodyne co-ordinate to check  camera2 imaging plane projection
    testp = [11.3, -2.95, -1.0]
    bb3d.append(testp)

    xnd = np.array(testp + [1.0])
    # print ('bb3d xnd velodyne   ', xnd)
    # xpnd = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(xnd)))
    xpnd = Tr_velo_to_cam.dot(xnd)
    # print ('bb3d xpnd cam0      ', xpnd)
    xpnd = R0_rect.dot(xpnd)
    # print ('bb3d xpnd rect cam0 ', xpnd)
    xpnd = P2_rect.dot(xpnd)
    # print ('bb3d xpnd cam2 image', xpnd)
    # print ('bb3d xpnd cam2 image', xpnd/xpnd[2])

    p = patches.Circle((xpnd[0] / xpnd[2], xpnd[1] / xpnd[2]), fill=False, radius=3, color='red', linewidth=2)
    ax.add_patch(p)

    return np.array(bb2d), np.array(bb3d)


class Kitti(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── calib
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    calibs_dir_name = "calib"

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.calibs = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "testing"

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
            calibs_dir = os.path.join(self._raw_folder, self._location, self.calibs_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
                self.calibs.append(os.path.join(calibs_dir, f"{img_file.split('.')[0]}.txt"))
        self.dic = self.class_dic(root)

    def __getitem__(self, index: int):
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            (image, target, calib, corners)
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float[2][16]

            corners is a list of dictionaries with the following keys:

            - base_3Dto2D: float[2][4]
            - corners_2D: float[2][8]
            - corners_3D: float[3][8]
            - paths_2D:

        """
        image = Image.open(self.images[index])
        image = np.asarray(image)
        target = self._parse_target(index) if self.train else None
        # calib = self._parse_calib(index) if self.train else None
        # corners = self._parse_corners(index, target, calib) if self.train else None
        sample = {}
        sample['image'] = image
        sample['target'] = target
        # sample['corners'] = corners

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _parse_corners(self, index, target, calib):
        corner = []
        P2_rect = calib['P2'].reshape(3, 4)
        for single in target:
            base_3Dto2D, corners_2D, corners_3D, paths_2D = computeBox3D(single, P2_rect)
            corner.append(
                {
                    # "corners_2D": corners_2D,
                    # "corners_3D": corners_3D,
                    # "paths_2D": paths_2D,
                    # "base_3Dto2D": base_3Dto2D
                    base_3Dto2D
                }
            )
        return corner

    def _parse_target(self, index: int) -> List:
        calib = {}
        with open(self.calibs[index]) as inp:
            for line in inp.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        P2_rect = calib['P2'].reshape(3, 4)
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                base_3Dto2D, corners_2D, corners_3D, paths_2D = computeBox3D([float(x) for x in line[8:15]], P2_rect)
                target.append(
                    {
                        "type": self.dic[line[0]],
                        # "truncated": float(line[1]),
                        # "occluded": int(line[2]),
                        # "alpha": float(line[3]),
                        # "bbox": [float(x) for x in line[4:8]],
                        "base": base_3Dto2D,
                        # "dimensions": [float(x) for x in line[8:11]],
                        # "location": [float(x) for x in line[11:14]],
                        # "rotation_y": float(line[14]),
                    }
                )
        return target

    def _parse_calib(self, index: int) -> Dict:
        calib = {}
        with open(self.calibs[index]) as inp:
            for line in inp.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        return calib

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self._raw_folder, exist_ok=True)

        # download files
        for fname in self.resources:
            download_and_extract_archive(
                url=f"{self.data_url}{fname}",
                download_root=self._raw_folder,
                filename=fname,
            )

    def class_dic(self, root):
        dic = {}
        index = 0
        for dir in self.targets:
            with open(dir) as inp:
                content = csv.reader(inp, delimiter=" ")
                for line in content:
                    if not line[0] in dic:
                        dic[line[0]] = index
                        index += 1
        print("dic size: ", len(dic))
        print(dic)
        return dic

    def name_to_label(self, name):
        return self.dic[name]

    def label_to_name(self, label):
        return "not implemented"

    def num_classes(self):
        return len(self.dic)
