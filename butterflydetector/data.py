import numpy as np

BBOX_SKELETON_COMPLETE = [
    [1, 2], [2, 3], [3, 4], [4, 1], [1,5], [2,5], [3,5], [4,5]
]

BBOX_POINTS = [
    'top_left',
    'top_right',
    'bottom_right',
    'bottom_left',
    'center',
]

HFLIP_BBOX = {
    'top_left': 'top_right',
    'top_right': 'top_left',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
}

LABELS_UAVDT = {0:"car", 1:"truck", 2:"bus", 3:"van", 4:"cyclist", 5:"pedestrian"}
LABELS_VISDRONE = {0:"pedestrian", 1:"people", 2:"bicycle", 3:"car", 4:"van", 5:"truck", 6:"tricycle", 7:"awning-tricycle", 8:"bus", 9:"motor", 10:"others"}
COLORS = {-1:"Black", 0:"Red", 1:"Blue", 2:"Yellow", 3:"Lime", 4:"Aqua", 5:"DeepPink", 6:"Indigo", 7:"SaddleBrown", 8:"Magenta", 9:"Green", 10:"Gray"}
