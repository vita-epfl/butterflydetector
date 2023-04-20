import logging
import re

from .butterfly import Butterfly
from .butterfly_laplacewh import ButterflyLaplaceWH
from .skeleton import Skeleton

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('Butterfly encoder')
    group.add_argument('--butterfly-side-length', default=Butterfly.side_length, type=int,
                       help='side length of the PIF field')


def factory(args, strides):
    # configure Butterfly
    Butterfly.side_length = args.butterfly_side_length
    ButterflyLaplaceWH.side_length = args.butterfly_side_length
    return factory_heads(args.headnets, strides)


def factory_heads(headnames, strides):
    if isinstance(headnames[0], (list, tuple)):
        return [factory_heads(task_headnames, task_strides)
                for task_headnames, task_strides in zip(headnames, strides)]

    encoders = [factory_head(head_name, stride)
                for head_name, stride in zip(headnames, strides)]


    if headnames[-1] == 'skeleton' and len(headnames) == len(strides) + 1 and 'butterfly' in headnames[0]:
        m = re.match('butterfly([0-9]+)$', headnames[0])
        if m is not None:
            n_keypoints = int(m.group(1))
        else:
            n_keypoints = 1
        encoders.append(Skeleton(n_keypoints=n_keypoints*5))
    if headnames[-1] == 'skeleton' and len(headnames) == len(strides) + 1:
        encoders.append(Skeleton())

    return encoders


def factory_head(head_name, stride):

    if head_name in (
        'butterfly', 'nsbutterfly',
    ) or re.match('(?:ns|o)?butterfly([0-9]+)$', head_name) is not None:
        m = re.match('(?:ns|o)?butterfly([0-9]+)$', head_name)
        if m is not None:
            n_keypoints = int(m.group(1))
            LOG.debug('using %d keypoints for pif', n_keypoints)
        else:
            n_keypoints = 17
        scale_wh = 1
        obutterfly = False
        if 'nsbutterfly' in head_name:
            scale_wh = stride
        if 'obutterfly' in head_name:
            obutterfly = True
        return Butterfly(stride, scale_wh, n_keypoints=n_keypoints, obutterfly=obutterfly)

    if head_name in (
        'butterfly_laplacewh', 'nsbutterfly_laplacewh',
    ) or re.match('(?:ns|o)?butterfly_laplacewh([0-9]+)$', head_name) is not None:
        m = re.match('(?:ns|o)?butterfly_laplacewh([0-9]+)$', head_name)
        if m is not None:
            n_keypoints = int(m.group(1))
            LOG.debug('using %d keypoints for pif', n_keypoints)
        else:
            n_keypoints = 17
        scale_wh = 1
        obutterfly = False
        if 'nsbutterfly_laplacewh' in head_name:
            scale_wh = stride
        if 'obutterfly_laplacewh' in head_name:
            obutterfly = True
        return ButterflyLaplaceWH(stride, scale_wh, n_keypoints=n_keypoints, obutterfly=obutterfly)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
