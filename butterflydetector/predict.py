"""Predict bbox for given images."""

import argparse
import glob
import json
import logging
import os

import numpy as np
import PIL
import torch

from .network import nets
from . import data_manager, decoder, show, transforms
from .nms import non_max_suppression_fast, py_cpu_softnms
from .data import LABELS_VISDRONE, COLORS

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m butterfly.predict',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['bbox', 'json'],
                        help='what to output: bbox, json')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--line-width', default=6, type=int,
                        help='line width for skeleton')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    parser.add_argument('--snms', default=False, action='store_true',
                        help='Use Soft NMS')
    parser.add_argument('--snms-threshold', default=0.005, type=float,
                        help='Set Soft-NMS threshold to remove')
    parser.add_argument('--nms-threshold', default=0.5, type=float,
                        help='Set NMS overlap threshold to remove')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig()
    logging.getLogger('butterfly_detector').setLevel(log_level)
    LOG.setLevel(log_level)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args

def main():
    args = cli()

    # load model
    model_cpu, _ = nets.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.head_names = model_cpu.head_names
        model.head_strides = model_cpu.head_strides
    processor = decoder.factory_from_args(args, model, args.device)

    # data
    preprocess = None
    if args.long_edge:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
            transforms.EVAL_TRANSFORM,
        ])
    else:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.CenterPadTight(32),
            transforms.EVAL_TRANSFORM,
        ])
    data = data_manager.ImageList(args.images, preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    skeleton_painter = show.InstancePainter(
        show_box=False,
        markersize=args.line_width - 5,
        linewidth=args.line_width,
    )

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        fields_batch = processor.fields(image_tensors_batch)
        pred_batch = processor.annotations_batch(fields_batch, debug_images=image_tensors_batch)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            if args.output_directory is None:
                output_path = meta['file_name']
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(args.output_directory, file_name)
            LOG.info('batch %d: %s to %s', batch_i, meta['file_name'], output_path)

            # load the original image if necessary
            cpu_image = None
            if args.debug or \
               'bbox' in args.output_types:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            processor.set_cpu_image(cpu_image, None)
            if preprocess is not None:
                pred, keypoint_sets, bboxes, scores = preprocess.annotations_inverse(pred, meta)

            categories =  np.argmax(keypoint_sets[:,:,2], axis=1)
            if len(bboxes)>0:
                #_, pick = non_max_suppression_fast(np.concatenate((bboxes[range(len(bboxes)), np.argmax(bboxes[:,:, 2], axis=1)], scores[:, np.newaxis]), axis=1), categories, overlapThresh=0.5)
                bboxes_res = np.array([])
                keypoint_sets_res = np.array([])
                scores_res = np.array([])

                if args.snms:
                    for cls in set(categories):
                        pick, scores_temp = py_cpu_softnms(bboxes[range(len(bboxes)), np.argmax(bboxes[:,:, 2], axis=1)][categories==cls], scores[categories==cls], Nt=0.5, sigma=0.5, thresh=args.snms_threshold, method=1)
                        if len(bboxes_res) == 0:
                            bboxes_res = bboxes[categories==cls][pick]
                            keypoint_sets_res = keypoint_sets[categories==cls][pick]
                            scores_res = scores_temp
                        else:
                            bboxes_res = np.concatenate((bboxes_res, bboxes[categories==cls][pick]))
                            keypoint_sets_res = np.concatenate((keypoint_sets_res, keypoint_sets[categories==cls][pick]))
                            scores_res = np.concatenate((scores_res, scores_temp))
                else:
                    _, pick = non_max_suppression_fast(np.concatenate((bboxes[range(len(bboxes)), np.argmax(bboxes[:,:, 2], axis=1)], scores[:, np.newaxis]), axis=1), categories, overlapThresh=args.nms_threshold)
                    bboxes_res = bboxes[pick]
                    keypoint_sets_res = keypoint_sets[pick]
                    scores_res = scores[pick]
                    categories = categories[pick]
                bboxes = bboxes_res
                keypoint_sets = keypoint_sets_res
                scores = scores_res

            if 'json' in args.output_types:
                with open(output_path + '.pifpaf.json', 'w') as f:
                    json.dump([
                        {
                            'keypoints': np.around(ann.data, 1).reshape(-1).tolist(),
                            'bbox': np.around(bbox_from_keypoints(ann.data), 1).tolist(),
                            'score': round(ann.score(), 3),
                        }
                        for ann in pred
                    ], f)
            texts = [LABELS_VISDRONE[np.argmax(kps[:,2])] for kps in keypoint_sets]
            colors = [COLORS[np.argmax(kps[:,2])] for kps in keypoint_sets]
            if 'bbox' in args.output_types:
                with show.image_canvas(cpu_image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    skeleton_painter.annotations(ax, pred, colors=colors, scores=scores, texts=texts)


if __name__ == '__main__':
    main()
