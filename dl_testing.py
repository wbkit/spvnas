from core.datasets import KITTI360


def main():
    dl_obj = KITTI360(
        root="/scratch_net/biwidl303/wboet/datasets/KITTI360/", voxel_size=0.05, num_points=80000
    )


if __name__ == "__main__":
    main()
