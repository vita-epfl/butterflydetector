"""Evaluation on COCO data."""
import matplotlib
import matplotlib.pyplot as plt

import argparse
import json
import logging
import os
import time
import zipfile

import numpy as np
import torch
import itertools

from .network import nets
from . import decoder, encoder, show, transforms, data_manager

from collections import defaultdict
from .nms import non_max_suppression_fast, py_cpu_softnms

import sys
LOG = logging.getLogger(__name__)
class EvalAerial(object):
    def __init__(self, processor, annotations_inverse, headnames, args, skeleton=None):
        self.processor = processor
        self.annotations_inverse = annotations_inverse
        self.predictions = defaultdict(list)
        self.decoder_time = 0.0
        self.dict_folder = defaultdict(list)
        self.args = args
        self.image_ids = []

    def from_predictions(self, annotations, meta,
                         debug=False, image_cpu=None, verbose=False,
                         category_id=1):
        image_id = int(meta['image_id'])
        self.image_ids.append(image_id)

        if self.annotations_inverse:
            pred, instances, bboxes, scores = self.annotations_inverse(annotations, meta)

        categories =  np.argmax(instances[:,:,2], axis=1)
        if len(bboxes)>0:
            #_, pick = non_max_suppression_fast(np.concatenate((bboxes[range(len(bboxes)), np.argmax(bboxes[:,:, 2], axis=1)], scores[:, np.newaxis]), axis=1), categories, overlapThresh=0.5)
            bboxes_res = np.array([])
            keypoint_sets_res = np.array([])
            scores_res = np.array([])
            if self.args.snms:
                for cls in set(categories):
                    pick, scores_temp = py_cpu_softnms(bboxes[range(len(bboxes)), np.argmax(bboxes[:,:, 2], axis=1)][categories==cls], scores[categories==cls], Nt=0.5, sigma=0.5, thresh=self.args.snms_threshold, method=1)
                    if len(bboxes_res) == 0:
                        bboxes_res = bboxes[categories==cls][pick]
                        keypoint_sets_res = instances[categories==cls][pick]
                        scores_res = scores_temp
                    else:
                        bboxes_res = np.concatenate((bboxes_res, bboxes[categories==cls][pick]))
                        keypoint_sets_res = np.concatenate((keypoint_sets_res, instances[categories==cls][pick]))
                        scores_res = np.concatenate((scores_res, scores_temp))
            elif self.args.nms:
                _, pick = non_max_suppression_fast(np.concatenate((bboxes[range(len(bboxes)), np.argmax(bboxes[:,:, 2], axis=1)], scores[:, np.newaxis]), axis=1), categories, overlapThresh=self.args.nms_threshold)
                bboxes_res = bboxes[pick]
                keypoint_sets_res = instances[pick]
                scores_res = scores[pick]
            if self.args.nms or self.args.snms:
                bboxes = bboxes_res
                instances = keypoint_sets_res
                scores = scores_res
        list_instances = zip(instances, bboxes, scores)
        image_annotations = []


        for tuple_instance in list_instances:
            instance, bbox, score = tuple_instance
            # avoid visible keypoints becoming invisible due to rounding
            v_mask = instance[:, 2] > 0.0
            instance[v_mask, 2] = np.maximum(0.01, instance[v_mask, 2])

            keypoints = np.around(instance, 2)
            category_id = np.argmax(keypoints[:,2])
            # elif self.cif_used:
            #     category_id = int(np.argmax(instance_class[:,2]))
            image_annotations.append({
                'image_id': image_id,
                'category_id': category_id,
                'keypoints': keypoints,
                'bbox': bbox[category_id, :4],
                'score': max(0.01, score),
            })

            if bbox[category_id, 2] == 0 or bbox[category_id, 3] == 0:
                continue
            self.predictions[category_id+1].append({
                'image_id': image_id,
                'category_id': category_id,
                'keypoints': keypoints,
                'bbox': bbox[category_id, :4],
                'score': max(0.01, score),
            })
        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            image_annotations.append({
                'image_id': image_id,
                'category_id': -1,
                'keypoints': np.zeros((6,3)),
                'bbox': np.zeros((4)),
                'score': 0.0,
            })
        bboxes = defaultdict(list)

        fileName = meta['file_name']
        split_fileName = fileName.split("/")
        if self.args.dataset == "visdrone":
            folder = os.path.splitext(split_fileName[-1])[0]
        elif self.args.dataset == "uavdt":
            folder = split_fileName[-2]
            image_numb = int(split_fileName[-1][3:9])
        else:
            folder = fileName
            mode = meta['mode']
            time = meta['time']
        for ann in image_annotations:
            #x, y, w, h = self.extract_bbox(ann)
            x, y, w, h = ann['bbox']
            if w == 0 or h == 0:
                continue
            s = ann['score']
            bboxes[ann['category_id']].append([x, y, w, h, s])
            #self.dict_folder[folder].append(",".join(list(map(str,[image_numb, -1, x, y, w, h, s, 1, ann['category_id']]))))
        for categ in bboxes.keys():
            for bbox in np.array(bboxes[categ]):
                x, y, w, h, s = bbox
                if self.args.dataset == "visdrone":
                    self.dict_folder[folder].append(",".join(list(map(str,[x, y, w, h, s, categ+1, -1, -1]))))
                elif self.args.dataset == "uavdt":
                    self.dict_folder[folder].append(",".join(list(map(str,[image_numb, -1, x, y, w, h, s, 1, categ]))))
                elif self.args.dataset == "eurocity":
                    self.dict_folder[folder].append(",".join(list(map(str,[x, y, w, h, s, categ, mode, time]))))
        if len(self.dict_folder[folder])==0:
            self.dict_folder[folder].append(",".join(list(map(str,[0, 0, 0, 0, 0, 0, mode, time]))))

    def write_predictions(self, path):
        for folder in self.dict_folder.keys():
            with open(os.path.join(path,folder+".txt"), "w") as file:
                file.write("\n".join(self.dict_folder[folder]))

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser)
    encoder.cli(parser)
    data_manager.dataset_cli(parser)
    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('-n', default=0, type=int,
                        help='number of batches')
    parser.add_argument('--skip-n', default=0, type=int,
                        help='skip n batches')
    parser.add_argument('--dataset-split', choices=('val', 'test', 'test-dev', 'test-challenge', 'custom'), default='val',
                        help='dataset to evaluate')
    parser.add_argument('--min-ann', default=0, type=int,
                        help='minimum number of truth annotations')
    parser.add_argument('--long-edge', default=641, type=int,
                        help='long edge of input images')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--all-images', default=False, action='store_true',
                        help='run over all images irrespective of catIds')
    parser.add_argument('--snms', default=False, action='store_true',
                        help='Use Soft NMS')
    parser.add_argument('--nms', default=False, action='store_true',
                        help='Use NMS')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot results to output')
    parser.add_argument('--snms-threshold', default=0.005, type=float,
                        help='Set Soft-NMS threshold to remove')
    parser.add_argument('--nms-threshold', default=0.5, type=float,
                        help='Set NMS overlap threshold to remove')
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # if args.dataset in ('test', 'test-dev') and (not args.write_predictions and not args.write_results):
    #     raise Exception('have to use --write-predictions for this dataset')
    #if args.dataset in ('test', 'test-dev') and not args.all_images:
    #    raise Exception('have to use --all-images for this dataset')

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    # generate a default output filename
    if args.output is None:
        args.output = '{}.evalcoco-{}edge{}-samples{}{}{}'.format(
            args.checkpoint,
            '{}-'.format(args.dataset_split) if args.dataset_split != 'val' else '',
            args.long_edge,
            args.n,
            '-noforcecompletepose' if not args.force_complete_pose else '',
            '-twoscale' if args.two_scale else '',
        )

    return args

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def write_evaluations(eval_aerial, path, args, total_time):
    mkdir_if_missing(path)
    eval_aerial.write_predictions(path)

    n_images = len(eval_aerial.image_ids)

    print('n images = {}'.format(n_images))
    print('decoder time = {:.1f}s ({:.0f}ms / image)'
          ''.format(eval_aerial.decoder_time, 1000 * eval_aerial.decoder_time / n_images))
    print('total time = {:.1f}s ({:.0f}ms / image)'
          ''.format(total_time, 1000 * total_time / n_images))

def preprocess_factory_from_args(args):
    preprocess = None
    collate_fn = data_manager.collate_images_anns_meta
    if args.batch_size == 1:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.CenterPadTight(32),
            transforms.EVAL_TRANSFORM,
        ])
    else:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
            transforms.EVAL_TRANSFORM,
        ])

    return preprocess, collate_fn


def main():
    args = cli()

    # skip existing?
    if args.skip_existing:
        if os.path.exists(args.output + '.stats.json'):
            print('Output file {} exists already. Exiting.'
                  ''.format(args.output + '.stats.json'))
            return
        print('Processing: {}'.format(args.checkpoint))
    preprocess, collate_fn = preprocess_factory_from_args(args)
    data_loader = data_manager.dataset_factory(
        args, preprocess, None, test_mode=True, collate_fn=collate_fn)

    model_cpu, _ = nets.factory_from_args(args)
    model = model_cpu.to(args.device)
    headnames = tuple(h.shortname for h in model.head_nets)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.head_names = model_cpu.head_names
        model.head_strides = model_cpu.head_strides
    processor = decoder.factory_from_args(args, model, args.device)


    eval = EvalAerial(processor, preprocess.annotations_inverse if preprocess else None, headnames=headnames, args=args)
    total_start = time.time()
    loop_start = time.time()
    for batch_i, (image_tensors_cpu, _, meta_batch) in enumerate(data_loader):
        logging.info('batch %d, last loop: %.3fs, batches per second=%.1f',
                     batch_i, time.time() - loop_start,
                     batch_i / max(1, (time.time() - total_start)))
        if batch_i < args.skip_n:
            continue
        if args.n and batch_i >= args.n:
            break

        loop_start = time.time()

        fields_batch = processor.fields(image_tensors_cpu)

        decoder_start = time.perf_counter()
        pred_batch = processor.annotations_batch(
            fields_batch, debug_images=image_tensors_cpu)
        eval.decoder_time += time.perf_counter() - decoder_start

        # loop over batch
        assert len(image_tensors_cpu) == len(fields_batch)
        for image_tensor_cpu, pred, meta in zip(
                image_tensors_cpu, pred_batch, meta_batch):
            eval.from_predictions(pred, meta,
                                       debug=args.debug,
                                       image_cpu=image_tensor_cpu)
    total_time = time.time() - total_start

    # processor.instance_scorer.write_data('instance_score_data.json')$
    data_loader.dataset.write_evaluations(eval, args.output, total_time)

if __name__ == '__main__':
    main()
