from dataset.kitti_dataset import SemKITTI


def dataset_builder(args):
    """Build the outdoor (SemanticKITTI) train/val datasets.

    The CARLA branch this used to carry imported `dataset.carla_dataset`, which was
    never committed — the module raised ImportError on import and was therefore
    unusable. Removed rather than left broken.
    """
    if args.dataset != "kitti":
        raise ValueError(f"unknown dataset {args.dataset!r}; only 'kitti' is supported")

    dataset = SemKITTI(args, "train")
    val_dataset = SemKITTI(args, "val")
    args.num_class = 20
    args.grid_size = [128, 128, 32]
    class_names = [
        "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
        "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence",
        "vegetation", "trunk", "terrain", "pole", "traffic-sign",
    ]

    return dataset, val_dataset, args.num_class, class_names
