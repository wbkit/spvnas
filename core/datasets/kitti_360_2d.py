""" KITTI-360 dataset class. Uses the npz files """

import os
import os.path
import re

from PIL import Image
from plyfile import PlyData
from torchvision.transforms import (
    ColorJitter,
    RandomCrop,
    RandomHorizontalFlip,
    GaussianBlur,
    RandomAutocontrast,
    AugMix,
)
import numpy as np
import torch
from helpers.project import CameraPerspective
from helpers.annotation import Annotation3DPly, Annotation2D
from helpers.labels import (
    ID2TRAINID,
    KITTI360_NUM_CLASSES,
)
from helpers.labels import labels as LABELS

__all__ = ["KITTI360"]
SEQ_TRAIN_VAL = [0, 2, 4, 5, 6, 7, 9, 10]


class KITTI360(dict):
    def __init__(self, root, voxel_size, num_points, radius, **kwargs):
        submit_to_server = kwargs.get("submit", False)
        sample_stride = kwargs.get("sample_stride", 1)

        if submit_to_server:
            super().__init__(
                {
                    "train": KITTI360Internal(
                        root, voxel_size, num_points, radius, sample_stride=1, split="train"
                    ),
                    "test": KITTI360Internal(
                        root, voxel_size, num_points, radius, sample_stride=1, split="test"
                    ),
                }
            )
        else:
            super().__init__(
                {
                    "train": KITTI360Internal(
                        root,
                        voxel_size,
                        num_points,
                        radius,
                        sample_stride=1,
                        split="train",
                    ),
                    "test": KITTI360Internal(
                        root,
                        voxel_size,
                        num_points,
                        radius,
                        sample_stride=sample_stride,
                        split="val",
                    ),
                }
            )


class KITTI360Internal:
    def __init__(
        self,
        root,
        voxel_size,
        num_points,
        radius,
        split,
        modality,
        feature_extractor=None,
        sample_stride=1,
        config=None,
    ):
        self.kitti360Path = root
        self.split = split
        self.modality = modality
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        if config is not None:
            self.student_aug = config["student"]["aug_list"]
            self.cutout_size = config["student"]["cutout_size"]
        else:
            self.student_aug = ["jitter"]
            self.cutout_size = 120

        self.reverse_label_name_mapping = LABELS
        self.num_classes = KITTI360_NUM_CLASSES
        self.angle = 0.0
        self.radius = radius

        self.crop_obj = RandomCrop((376, 512))

        self.label_2d_obj = Annotation2D()
        self.feature_extractor = feature_extractor

        label_dir = os.path.join(
            self.kitti360Path,
            "data_3d_semantics",
        )

        # Get the training and validation splits
        train_file = os.path.join(
            self.kitti360Path, "data_3d_semantics/train/2013_05_28_drive_train.txt"
        )
        val_file = os.path.join(
            self.kitti360Path, "data_3d_semantics/train/2013_05_28_drive_val.txt"
        )
        with open(train_file, "r") as f:
            train_chunk_list = [line.strip() for line in f]
        with open(val_file, "r") as f:
            val_chunk_list = [line.strip() for line in f]

        # train_chunk_list = train_chunk_list[::3]
        # val_chunk_list = val_chunk_list[::3]

        camera_obj_list = []
        lidar_obj_list = []
        frame_list_train = []
        frame_list_val = []
        # Iterate over all sequences
        for seg_idx, seq_id in enumerate(SEQ_TRAIN_VAL):
            # Get the camera positions & frames
            # Create a list of the corresponding camera objects to be used later by the frame list
            sequence = "2013_05_28_drive_%04d_sync" % seq_id
            camera_obj = CameraPerspective(self.kitti360Path, sequence, cam_id=0)
            camera_obj_list.append(camera_obj)

            # Get the paths of all 3D chunks in the sequence and their corresponding frame numbers
            label_3d_obj = Annotation3DPly(label_dir, sequence)
            lidar_obj_list.append(label_3d_obj)

            lidar_frame_range_list = []
            for path_3d in label_3d_obj.pcdFileList:
                start_idx, end_idx = self.__extract_numbers_from_filename(path_3d)
                lidar_frame_range_list.append((start_idx, end_idx))

            # Match the chunks to the camera frames
            for frame_idx in range(len(camera_obj.cam2world)):
                # Check if the frame has a corresponding label file
                label_file = os.path.join(
                    self.kitti360Path,
                    "data_2d_semantics/train",
                    "2013_05_28_drive_%04d_sync" % seq_id,
                    "image_%02d" % 0,
                    "semantic",
                    "%010d.png" % camera_obj.frames[frame_idx],
                )
                if not os.path.isfile(label_file):
                    # print(f"Error: No label file found for frame {frame_idx}")
                    continue

                # Find the chunk that contains the frame
                matching_chunk = self.__find_indices_in_range(
                    camera_obj.frames[frame_idx], lidar_frame_range_list
                )
                if len(matching_chunk) == 1:
                    matching_chunk = matching_chunk[0]
                elif len(matching_chunk) == 2:
                    # If there are two chunks that contain the frame,
                    # choose the one that is further from the range boundary
                    matching_chunk = (
                        matching_chunk[0]
                        if camera_obj.frames[frame_idx]
                        - lidar_frame_range_list[matching_chunk[0]][1]
                        < lidar_frame_range_list[matching_chunk[1]][0]
                        - camera_obj.frames[frame_idx]
                        else matching_chunk[1]
                    )
                else:
                    print(f"Error: No matching chunk found for frame {frame_idx}")
                    continue

                # Create frame list item
                chunk_path = label_3d_obj.pcdFileList[matching_chunk]
                if chunk_path.replace(self.kitti360Path, "") in train_chunk_list:
                    frame_list_train.append(
                        {
                            "sequence": sequence,
                            "camera_obj_idx": seg_idx,
                            "frame_idx": frame_idx,
                            "lidar_path": label_3d_obj.pcdFileList[matching_chunk],
                        }
                    )
                elif chunk_path.replace(self.kitti360Path, "") in val_chunk_list:
                    frame_list_val.append(
                        {
                            "sequence": sequence,
                            "camera_obj_idx": seg_idx,
                            "frame_idx": frame_idx,
                            "lidar_path": label_3d_obj.pcdFileList[matching_chunk],
                        }
                    )
                else:
                    continue

        # Finalise the dataset lists
        self.camera_obj_list = camera_obj_list
        self.lidar_obj_list = lidar_obj_list
        if split == "train":
            self.frame_list = frame_list_train
        elif self.split == "val":
            self.frame_list = frame_list_val
        elif self.split == "test":
            raise NotImplementedError("Test split not implemented yet")

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        if self.modality == "rgb":
            return self.__getitem_rgb(idx)
        else:
            return self.__getitem_3d(idx)

    def __getitem_3d(self, idx):
        idx_info = self.frame_list[idx]
        point_file = idx_info["lidar_path"]
        # Load the point cloud
        pt_cloud, label, visible = self.read_kitti_360_npz(point_file)

        # Only show visible points
        pt_cloud = pt_cloud[visible == 1]
        label = label[visible == 1]

        # Get the camera positions & frames
        camera_obj = self.camera_obj_list[idx_info["camera_obj_idx"]]
        camera_positions = camera_obj.cam2world[camera_obj.frames[idx_info["frame_idx"]]]
        cam_x = camera_positions[0, 3]
        cam_y = camera_positions[1, 3]
        cam_z = camera_positions[2, 3]

        # Trasform to local KOS
        pt_cloud[:, 0] = pt_cloud[:, 0] - cam_x
        pt_cloud[:, 1] = pt_cloud[:, 1] - cam_y
        pt_cloud[:, 2] = pt_cloud[:, 2] - cam_z

        # Filter points that are within the given radius in the x, y plane
        distances = np.sqrt(pt_cloud[:, 0] ** 2 + pt_cloud[:, 1] ** 2)
        is_close_enough = distances <= self.radius
        pt_cloud = pt_cloud[is_close_enough]
        label = label[is_close_enough]

        # distances = distances[is_close_enough]  # TODO test

        # Select a random subset of points
        inds = np.linspace(0, pt_cloud.shape[0] - 1, pt_cloud.shape[0], dtype=int)
        if pt_cloud.shape[0] > self.num_points:
            inds = np.random.choice(inds, self.num_points, replace=False)
            pt_cloud = pt_cloud[inds, :]
            label = label[inds]
            # distances = distances[inds]  # TODO test

        # Convert the labels to train IDs
        label = ID2TRAINID[label.astype(np.uint8)]

        ####################
        # Prepare the data #
        ####################
        # TODO The rotation doesn't need to be 3D
        if "train" in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array(
                [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            )

            pt_cloud[:, :3] = np.dot(pt_cloud[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array(
                [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            )
            pt_cloud[:, :3] = np.dot(pt_cloud[:, :3], transform_mat)

        pc_ = np.round(pt_cloud[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = pt_cloud[:, 0:6]  # xyzRGB as features
        # feat_[:, 0] = distances  # TODO test
        labels_ = label

        # _, inds, inverse_map = sparse_quantize(pc_, return_index=True, return_inverse=True)

        # pc = pc_[inds]
        # feat = feat_[inds].astype(np.float32)
        # labels = labels_[inds]
        # lidar = SparseTensor(feat, pc)
        # labels = SparseTensor(labels, pc)
        # labels_ = SparseTensor(labels_, pc_)
        # inverse_map = SparseTensor(inverse_map, pc_)

        # return {
        #     "lidar": lidar,
        #     "targets": labels,
        #     "targets_mapped": labels_,
        #     "inverse_map": inverse_map,
        #     "file_name": self.frame_list[idx]["lidar_path"],
        # }
        return NotImplementedError

    def __getitem_rgb(self, idx):
        rgb_image, label_image = self.get_image(idx)

        # Apply data augmentation
        if self.split == "train":
            # Random crop
            seed = torch.randint(0, 2**32 - 1, ())
            torch.manual_seed(seed)
            rgb_image = self.crop_obj(rgb_image)
            torch.manual_seed(seed)
            label_image = self.crop_obj(label_image)
            # Random horizontal flip
            if np.random.rand() < 0.5:
                rgb_image = RandomHorizontalFlip(1)(rgb_image)
                label_image = RandomHorizontalFlip(1)(label_image)

        rgb_image_student = rgb_image.copy()
        if self.split == "train":
            # # Random Autocontrast
            # if np.random.rand() < 0.5:
            #     rgb_image_student = RandomAutocontrast(1)(rgb_image_student)
            # Random color jitter
            if "jitter" in self.student_aug:
                if np.random.rand() < 0.5:
                    rgb_image_student = ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.25
                    )(rgb_image_student)
            # Random Gaussian noise
            if "blur" in self.student_aug:
                if np.random.rand() < 0.2:
                    rgb_image_student = GaussianBlur(5, sigma=(0.1, 1.5))(rgb_image_student)
            # Apply AugMix for regularization
            if "augmix" in self.student_aug:
                if np.random.rand() < 0.5:
                    rgb_image_student = AugMix(severity=1)(rgb_image_student)

            # Apply CutOut for regularization
            if "cutout" in self.student_aug:
                sqare_size = 130
                x1 = np.random.randint(0, rgb_image_student.size[0]) - sqare_size // 2
                y1 = np.random.randint(0, rgb_image_student.size[1]) - sqare_size // 2
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x1 + sqare_size > rgb_image_student.size[0]:
                    x1 = rgb_image_student.size[0] - sqare_size
                if y1 + sqare_size > rgb_image_student.size[1]:
                    y1 = rgb_image_student.size[1] - sqare_size

                rgb_image_student = np.array(rgb_image_student)
                rgb_image_student[x1 : x1 + sqare_size, y1 : y1 + sqare_size, :] = (0, 0, 0)

        # Convert the labels to train IDs
        label_image = np.array(label_image)
        label_image = ID2TRAINID[label_image.astype(np.uint8)]

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs_teacher = self.feature_extractor(rgb_image, label_image, return_tensors="pt")
        for k, v in encoded_inputs_teacher.items():
            encoded_inputs_teacher[k].squeeze_()  # remove batch dimension
        encoded_inputs_student = self.feature_extractor(
            rgb_image_student, label_image, return_tensors="pt"
        )
        for k, v in encoded_inputs_student.items():
            encoded_inputs_student[k].squeeze_()  # remove batch dimension

        return {"student": encoded_inputs_student, "teacher": encoded_inputs_teacher}

    def get_image(self, idx):
        idx_info = self.frame_list[idx]
        image_nr = self.camera_obj_list[idx_info["camera_obj_idx"]].frames[idx_info["frame_idx"]]

        sequence = self.frame_list[idx]["sequence"]
        image = "%010d.png" % image_nr
        rgb_file = os.path.join(
            self.kitti360Path, "data_2d_raw", sequence, "image_00/data_rect", image
        )
        if self.split == "train":
            label_file = os.path.join(
                self.kitti360Path, "data_2d_semantics/train", sequence, "image_00/scribble", image
            )
        else:
            label_file = os.path.join(
                self.kitti360Path, "data_2d_semantics/train", sequence, "image_00/semantic", image
            )

        rgb_image = Image.open(rgb_file)
        label_image = Image.open(label_file).convert("L")
        return rgb_image, label_image

    def __extract_numbers_from_filename(self, filename):
        pattern = r"\d{10}"  # \d matches digits, {10} matches exactly 10 occurrences
        numbers = re.findall(pattern, filename)

        if len(numbers) == 2:
            try:
                first_number = int(numbers[0])
                second_number = int(numbers[1])
                return first_number, second_number
            except ValueError:
                print("Error: The extracted substrings are not valid numbers.")
                return None, None
        else:
            print("Error: Two 10-digit numbers were not found in the filename.")
            return None, None

    def __find_indices_in_range(self, value, tuple_list):
        return [index for index, t in enumerate(tuple_list) if t[0] <= value <= t[1]]

    def read_kitti_360_ply(self, filepath):
        with open(filepath, "rb") as f:
            window = PlyData.read(f)

        data = np.array(
            [
                window["vertex"][axis]
                for axis in [
                    "x",
                    "y",
                    "z",
                    "red",
                    "green",
                    "blue",
                ]
            ]
        ).T
        label = np.array(window["vertex"]["semantic"])
        visible = np.array(window["vertex"]["visible"])

        return data, label, visible

    def read_kitti_360_npz(self, filepath):
        # Replace the file extension
        filepath = filepath.replace(".ply", ".npz")
        data = np.load(filepath)
        return data["pt_cloud"], data["label"], data["visible"]

    @staticmethod
    def collate_fn(inputs):
        batch = sparse_collate_fn(inputs)
        return batch


class KITTI360ts(KITTI360Internal):
    def __init__(
        self,
        root,
        voxel_size,
        num_points,
        radius,
        split,
        modality,
        feature_extractor=None,
        sample_stride=1,
    ):
        super().__init__(
            root,
            voxel_size,
            num_points,
            radius,
            split,
            "rgb",
            feature_extractor=None,
            sample_stride=1,
        )

    def __getitem__(self, idx):
        rgb_image, label_image = self.get_image(idx)

        # # Apply data augmentation that is consistent across student and teacher
        # if self.split == "train":
        #     # Random crop
        #     seed = torch.randint(0, 2**32 - 1, ())
        #     torch.manual_seed(seed)
        #     rgb_image = self.crop_obj(rgb_image)
        #     torch.manual_seed(seed)
        #     label_image = self.crop_obj(label_image)
        #     # Random horizontal flip
        #     if np.random.rand() < 0.5:
        #         rgb_image = RandomHorizontalFlip(1)(rgb_image)
        #         label_image = RandomHorizontalFlip(1)(label_image)

        # rgb_image_student = rgb_image.copy()
        # if self.split == "train":
        #     # Random Autocontrast
        #     if np.random.rand() < 0.5:
        #         rgb_image_student = RandomAutocontrast(1)(rgb_image_student)
        #     # Random color jitter
        #     rgb_image_student = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(
        #         rgb_image_student
        #     )
        #     # Random Gaussian noise
        #     if np.random.rand() < 0.5:
        #         rgb_image_student = GaussianBlur(1)(rgb_image_student)
        #     # Random Gausssian blur
        #     if np.random.rand() < 0.5:
        #         rgb_image_student = GaussianBlur(5)(rgb_image_student)

        # # Convert the labels to train IDs
        # label_image = np.array(label_image)
        # label_image = ID2TRAINID[label_image.astype(np.uint8)]

        # try:
        #     # encoded_inputs_student = self.feature_extractor(
        #     #     rgb_image_student, label_image, return_tensors="pt"
        #     # )
        #     # for k, v in encoded_inputs_student.items():
        #     #     encoded_inputs_student[k].squeeze_()  # remove batch dimension
        #     encoded_inputs_teacher = self.feature_extractor(
        #         rgb_image, label_image, return_tensors="pt"
        #     )
        #     for k, v in encoded_inputs_teacher.items():
        #         encoded_inputs_teacher[k].squeeze_()  # remove batch dimension
        # except:
        #     print("Exeption")

        # Apply data augmentation
        if self.split == "train":
            # Random crop
            seed = torch.randint(0, 2**32 - 1, ())
            torch.manual_seed(seed)
            rgb_image = self.crop_obj(rgb_image)
            torch.manual_seed(seed)
            label_image = self.crop_obj(label_image)
            # Random horizontal flip
            if np.random.rand() < 0.5:
                rgb_image = RandomHorizontalFlip(1)(rgb_image)
                label_image = RandomHorizontalFlip(1)(label_image)
            # Random color jitter
            rgb_image = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(rgb_image)

        # Convert the labels to train IDs
        label_image = np.array(label_image)
        label_image = ID2TRAINID[label_image.astype(np.uint8)]

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs_teacher = self.feature_extractor(rgb_image, label_image, return_tensors="pt")
        for k, v in encoded_inputs_teacher.items():
            encoded_inputs_teacher[k].squeeze_()  # remove batch dimension

        return {"student": encoded_inputs_teacher, "teacher": encoded_inputs_teacher}
