import logging
import re

from .butterfly_subpixel import Butterfly as Butterfly_subpixel
from .butterfly import Butterfly
from .processor import Processor
from .visualizer import Visualizer

LOG = logging.getLogger(__name__)


def cli(parser, *,
        seed_threshold=0.2,
        instance_threshold=0.0,
        keypoint_threshold=None,
        workers=None):
    group = parser.add_argument_group('decoder configuration')
    group.add_argument('--seed-threshold', default=seed_threshold, type=float,
                       help='minimum threshold for seeds')
    group.add_argument('--instance-threshold', type=float,
                       default=instance_threshold,
                       help='filter instances by score')
    group.add_argument('--keypoint-threshold', type=float,
                       default=keypoint_threshold,
                       help='filter keypoints by score')
    group.add_argument('--decoder-workers', default=workers, type=int,
                       help='number of workers for pose decoding')
    group.add_argument('--experimental-decoder', default=False, action='store_true',
                       help='use an experimental decoder')
    group.add_argument('--extra-coupling', default=0.0, type=float,
                       help='extra coupling')

    group.add_argument('--debug-fields-indices', default=[], nargs='+',
                       help=('indices of fields to create debug plots for '
                             '(group with comma, e.g. "0,1 2" to create one plot '
                             'with field 0 and 1 and another plot with field 2)'))
    group.add_argument('--debug-file-prefix', default=None,
                       help='save debug plots with this prefix')
    group.add_argument('--profile-decoder', default=None, action='store_true',
                       help='profile decoder')

    group = parser.add_argument_group('Butterfly decoder')
    group.add_argument('--scale-div', default=Butterfly.scale_div, type=int,
                       help='overwrite scale-div with fixed value, e.g. 0.5')


def factory_from_args(args, model, device=None):
    # configure Butterfly
    Butterfly.scale_div = args.scale_div

    debug_visualizer = None
    if args.debug_fields_indices:
        debug_visualizer = Visualizer(
            args.debug_fields_indices,
            file_prefix=args.debug_file_prefix,
            show = args.show
        )

    # default value for keypoint filter depends on whether complete pose is forced
    if args.keypoint_threshold is None:
        args.keypoint_threshold = 0.001 if not args.force_complete_pose else 0.0

    # decoder workers
    if args.decoder_workers is None and \
       getattr(args, 'batch_size', 1) > 1 and \
       debug_visualizer is None:
        args.decoder_workers = args.batch_size

    decode = factory_decode(model,
                            experimental=args.experimental_decoder,
                            seed_threshold=args.seed_threshold,
                            extra_coupling=args.extra_coupling,
                            multi_scale=args.multi_scale,
                            multi_scale_hflip=args.multi_scale_hflip,
                            debug_visualizer=debug_visualizer)

    return Processor(model, decode,
                     instance_threshold=args.instance_threshold,
                     keypoint_threshold=args.keypoint_threshold,
                     debug_visualizer=debug_visualizer,
                     profile=args.profile_decoder,
                     worker_pool=args.decoder_workers,
                     device=device)


def factory_decode(model, *,
                   extra_coupling=0.0,
                   experimental=False,
                   multi_scale=False,
                   multi_scale_hflip=True,
                   **kwargs):
    """Instantiate a decoder."""

    head_names = (
        tuple(model.head_names)
        if hasattr(model, 'head_names')
        else tuple(h.shortname for h in model.head_nets)
    )
    LOG.debug('head names = %s', head_names)
    if head_names in (
        ('butterfly',), ('nsbutterfly',), ('obutterfly',),)\
        or (len(head_names) == 1 and re.match('(?:ns|o)?butterfly([0-9]+)$', head_names[0]) is not None):
        return Butterfly(model.head_strides[-1],
                      head_names=head_names,
                      head_index=0,
                      **kwargs)
        # return Butterfly(model.head_strides[-1],
        #               head_names=head_names,
        #               **kwargs)

    raise Exception('decoder unknown for head names: {}'.format(head_names))
