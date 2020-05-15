from PIL import Image

from .uavdt import UAVDT
from .visdrone import VisDrone
import torch.utils.data

def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas

dataset_list = {
    'uavdt': UAVDT,
    'visdrone': VisDrone,
}

class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)

def cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--dataset', type=str, default='uavdt',
                        choices=dataset_list.keys())
    group.add_argument('--train-annotations', default=None)
    group.add_argument('--train-image-dir', default=None)
    group.add_argument('--val-annotations', default=None)
    group.add_argument('--val-image-dir', default=None)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--pre-duplicate-data', default=None, type=int,
                       help='duplicate pre data in preprocessing')
    group.add_argument('--loader-workers', default=2, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')


def dataset_factory(args, preprocess, target_transforms, test_mode=False, collate_fn=None):
    dataset = dataset_list[args.dataset]
    if test_mode:
        data = dataset(
            root=dataset.test_path[args.dataset_split],
            annFile=None,
            preprocess=preprocess,
            n_images=args.n_images,
        )
        return torch.utils.data.DataLoader(
            data, batch_size=args.batch_size, pin_memory=args.pin_memory,
            num_workers=args.loader_workers, collate_fn=collate_fn)

    train_data = dataset(
        root=args.train_image_dir or dataset.train_image_dir,
        annFile=args.train_annotations or dataset.train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )

    val_data = dataset(
        root=args.val_image_dir or dataset.val_image_dir,
        annFile=args.val_annotations or dataset.val_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )

    pre_train_data = dataset(
        root=args.train_image_dir or dataset.train_image_dir,
        annFile=args.train_annotations or dataset.train_annotations,
        preprocess=preprocess,
        target_transforms=target_transforms,
        n_images=args.pre_n_images,
    )

    if args.duplicate_data:
        train_data = torch.utils.data.ConcatDataset(
            [train_data for _ in range(args.duplicate_data)])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle= not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)


    if args.duplicate_data:
        val_data = torch.utils.data.ConcatDataset(
            [val_data for _ in range(args.duplicate_data)])
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)


    if args.pre_duplicate_data:
        pre_train_data = torch.utils.data.ConcatDataset(
            [pre_train_data for _ in range(args.pre_duplicate_data)])
    pre_train_loader = torch.utils.data.DataLoader(
        pre_train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader, pre_train_loader
