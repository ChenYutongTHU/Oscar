import anytree
import base64
import json
import numpy as np
import os.path as op
import torch
from typing import Callable, Dict, List, Optional, Tuple

from oscar.modeling.modeling_utils import BeamHypotheses
class ConstraintFilter(object):
    r"""
    A helper class to perform constraint filtering for providing sensible set of constraint words
    while decoding.
    Extended Summary
    ----------------
    The original work proposing `Constrained Beam Search <https://arxiv.org/abs/1612.00576>`_
    selects constraints randomly.
    We remove certain categories from a fixed set of "blacklisted" categories, which are either
    too rare, not commonly uttered by humans, or well covered in COCO. We resolve overlapping
    detections (IoU >= 0.85) by removing the higher-order of the two objects (e.g. , a "dog" would
    suppress a ‘mammal’) based on the Open Images class hierarchy (keeping both if equal).
    Finally, we take the top-k objects based on detection confidence as constraints.
    Parameters
    ----------
    hierarchy_jsonpath: str
        Path to a JSON file containing a hierarchy of Open Images object classes.
    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    """

    # fmt: off
    BLACKLIST: List[str] = [
        "auto part", "bathroom accessory", "bicycle wheel", "boy", "building", "clothing",
        "door handle", "fashion accessory", "footwear", "girl", "hiking equipment", "human arm",
        "human beard", "human body", "human ear", "human eye", "human face", "human foot",
        "human hair", "human hand", "human head", "human leg", "human mouth", "human nose",
        "land vehicle", "mammal", "man", "person", "personal care", "plant", "plumbing fixture",
        "seat belt", "skull", "sports equipment", "tire", "tree", "vehicle registration plate",
        "wheel", "woman", "__background__",
    ]
    # fmt: on

    REPLACEMENTS: Dict[str, str] = {
        "band-aid": "bandaid",
        "wood-burning stove": "wood burning stove",
        "kitchen & dining room table": "table",
        "salt and pepper shakers": "salt and pepper",
        "power plugs and sockets": "power plugs",
        "luggage and bags": "luggage",
    }

    def __init__(
        self, hierarchy_jsonpath, nms_threshold, max_given_constraints
    ):
        def __read_hierarchy(node, parent=None):
            # Cast an ``anytree.AnyNode`` (after first level of recursion) to dict.
            attributes = dict(node)
            children = attributes.pop("Subcategory", [])

            node = anytree.AnyNode(parent=parent, **attributes)
            for child in children:
                __read_hierarchy(child, parent=node)
            return node

        # Read the object class hierarchy as a tree, to make searching easier.
        self._hierarchy = __read_hierarchy(json.load(open(hierarchy_jsonpath)))

        self._nms_threshold = nms_threshold
        self._max_given_constraints = max_given_constraints

    def __call__(self, boxes: np.ndarray, class_names: List[str], scores: np.ndarray) -> List[str]:

        # Remove padding boxes (which have prediction confidence score = 0), and remove boxes
        # corresponding to all blacklisted classes. These will never become CBS constraints.
        keep_indices = []
        print('Initialize')
        print([(i,c) for i,c in enumerate(class_names)])
        for i in range(len(class_names)):
            if scores[i] > 0 and class_names[i] not in self.BLACKLIST:
                keep_indices.append(i)
        print('After removing blackbox')
        print([(i,c) for i,c in enumerate(class_names) if i in keep_indices])
        boxes = boxes[keep_indices]
        class_names = [class_names[i] for i in keep_indices]
        scores = scores[keep_indices]

        # Perform non-maximum suppression according to category hierarchy. For example, for highly
        # overlapping boxes on a dog, "dog" suppresses "animal".
        keep_indices = self._nms(boxes, class_names)
        print('After nms')
        print([(i,c) for i,c in enumerate(class_names) if i in keep_indices])
        boxes = boxes[keep_indices]
        class_names = [class_names[i] for i in keep_indices]
        scores = scores[keep_indices]

        # Retain top-k constraints based on prediction confidence score.
        class_names_and_scores = sorted(list(zip(class_names, scores)), key=lambda t: -t[1])
        class_names_and_scores = class_names_and_scores[: self._max_given_constraints]

        # Replace class name according to ``self.REPLACEMENTS``.
        class_names = [self.REPLACEMENTS.get(t[0], t[0]) for t in class_names_and_scores]

        # Drop duplicates.
        class_names = list(set(class_names))
        return class_names

    def _nms(self, boxes: np.ndarray, class_names: List[str]):
        if len(class_names) == 0:
            return []

        # For object class, get the height of its corresponding node in the hierarchy tree.
        # Less height => finer-grained class name => higher score.
        heights = np.array(
            [
                anytree.search.findall(self._hierarchy, lambda node: node.LabelName.lower() in c)[0].height
                for c in class_names
            ]
        )
        # Get a sorting of the heights in ascending order, i.e. higher scores first.
        score_order = heights.argsort()

        # Compute areas for calculating intersection over union. Add 1 to avoid division by zero
        # for zero area (padding/dummy) boxes.
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Fill "keep_boxes" with indices of boxes to keep, move from left to right in
        # ``score_order``, keep current box index (score_order[0]) and suppress (discard) other
        # indices of boxes having lower IoU threshold with current box from ``score_order``.
        # list. Note the order is a sorting of indices according to scores.
        keep_box_indices = []

        while score_order.size > 0:
            # Keep the index of box under consideration.
            current_index = score_order[0]
            keep_box_indices.append(current_index)

            # For the box we just decided to keep (score_order[0]), compute its IoU with other
            # boxes (score_order[1:]).
            xx1 = np.maximum(x1[score_order[0]], x1[score_order[1:]])
            yy1 = np.maximum(y1[score_order[0]], y1[score_order[1:]])
            xx2 = np.minimum(x2[score_order[0]], x2[score_order[1:]])
            yy2 = np.minimum(y2[score_order[0]], y2[score_order[1:]])

            intersection = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            union = areas[score_order[0]] + areas[score_order[1:]] - intersection

            # Perform NMS for IoU >= 0.85. Check score, boxes corresponding to object
            # classes with smaller/equal height in hierarchy cannot be suppressed.
            keep_condition = np.logical_or(
                heights[score_order[1:]] >= heights[score_order[0]],
                intersection / union <= self._nms_threshold,
            )

            # Only keep the boxes under consideration for next iteration.
            score_order = score_order[1:]
            score_order = score_order[np.where(keep_condition)[0]]

        return keep_box_indices

class ConstraintBoxesReader(object):
    r"""
    A reader for annotation files containing detected bounding boxes.
    For our use cases, the detections are from an object detector trained using Open Images.
    """
    def __init__(self, boxes_tsvpath):
        self._image_key_to_boxes = {}
        with open(boxes_tsvpath, 'r') as fp:
            for line in fp:
                parts = line.strip().split('\t')
                img_key = parts[0]
                labels = json.loads(parts[1])
                boxes, class_names, scores = [], [], []
                for box in labels:
                    boxes.append(box['rect'])
                    class_names.append(box['class'].lower())
                    scores.append(box['conf'])
                boxes = np.array(boxes)
                scores = np.array(scores)
                self._image_key_to_boxes[img_key] = {"boxes": boxes, "class_names": class_names, "scores": scores}

    def __len__(self):
        return len(self._image_key_to_boxes)

    def __getitem__(self, image_key):
        # Some images may not have any boxes, handle that case too.
        if image_key not in self._image_key_to_boxes:
            return {"boxes": np.array([]), "class_names": [], "scores":
                    np.array([])}
        else:
            return self._image_key_to_boxes[image_key]


hierarchy_jsonpath='/data/private/NocapsData/Datasets/nocaps/class_hierarchy.json'
nms_threshold=0.85
max_given_constraints=3
boxes_tsvpath = '/data/private/NocapsData/Datasets/nocaps/val_oi_detector.label.tsv'
filter_ = ConstraintFilter(hierarchy_jsonpath, nms_threshold, max_given_constraints=3)
reader = ConstraintBoxesReader(boxes_tsvpath)
test_id = ['0049a724f5dc20e4','04c304cf2d55093c']

for img_id in test_id:
    constraint_boxes = reader[img_id]
    candidates = filter_(
        constraint_boxes["boxes"], constraint_boxes["class_names"], constraint_boxes["scores"]
    )
    print(img_id, candidates)