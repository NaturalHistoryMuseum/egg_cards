###########
# Imports
###########


import numpy as np
import skimage.io as io
from copy import deepcopy
from source_functions import (
    find_all_contours,
    get_max_bounds,
    filter_contours,
    refine_boxes,
)
from itertools import combinations
import itertools
import cv2
from post_processing_functions import tuplewize


##################
# Helper Functions
##################


def approximate_box(box):
    # Input: original box contours.
    # Output: polygon approximation of contour.

    tst = np.zeros((len(box[0]), 1, 2), dtype="int32")
    tst[:, 0, 0] = np.int_(box[1])
    tst[:, 0, 1] = np.int_(box[0])

    peri = cv2.arcLength(tst, True)
    approx = cv2.approxPolyDP(tst, 0.02 * peri, True)

    return approx


def filter_for_possible_box_corners(eggcardboxes):
    # Input: egg card box contour (supposedly consisting of more than one box)
    # Output: vertices of possible boxes contained within one box contour.
    approx_corners = approximate_box(eggcardboxes)
    new_approx_corners = approx_corners.reshape(len(approx_corners), 2)
    X_tupled, Y_tupled, X_values, Y_values = get_new_corners(new_approx_corners)
    return [X_tupled, Y_tupled], [X_values, Y_values], new_approx_corners


def get_new_corners(new_approx):
    # Input: polygon vertices from box contour.
    # Output: x and y coordinates of vertices of different boxes within box contour.
    horiz_groups = {}
    vert_groups = {}
    kx = 0
    ky = 0
    for ind, corners in enumerate(new_approx):
        y, x = np.round_(corners, -1)
        if ind == 0:
            horiz_groups.update({kx: {"index": [ind], "vals": [x]}})
            vert_groups.update({ky: {"index": [ind], "vals": [y]}})
        else:
            # Horizontal points:
            for k in range(kx + 1):
                found_group = False
                current_x = horiz_groups[k]["vals"]
                current_x_ind = horiz_groups[k]["index"]
                min_diff = min([abs(x_ - x) for x_ in current_x])
                if min_diff <= 20:
                    current_x.append(x)
                    current_x_ind.append(ind)
                    found_group = True
                if found_group is True:
                    horiz_groups.update(
                        {k: {"index": current_x_ind, "vals": current_x}}
                    )
                    break

            if found_group is False:
                kx += 1
                horiz_groups.update({kx: {"index": [ind], "vals": [x]}})

            # Vertical points:
            for k in range(ky + 1):
                found_group = False
                current_y = vert_groups[k]["vals"]
                current_y_ind = vert_groups[k]["index"]
                min_diff = min([abs(y_ - y) for y_ in current_y])
                if min_diff <= 20:
                    current_y.append(y)
                    current_y_ind.append(ind)
                    found_group = True
                if found_group is True:
                    vert_groups.update({k: {"index": current_y_ind, "vals": current_y}})
                    break

            if found_group is False:
                ky += 1
                vert_groups.update({ky: {"index": [ind], "vals": [y]}})

    x_points = []
    for i in range(0, len(horiz_groups)):
        x_points.append(int(np.average(horiz_groups[i]["vals"])))

    y_points = []
    for i in range(0, len(vert_groups)):
        y_points.append(int(np.average(vert_groups[i]["vals"])))

    filtered_corners_x = tuplewize(np.sort(x_points), 2)
    filtered_corners_y = tuplewize(np.sort(y_points), 2)

    return filtered_corners_x, filtered_corners_y, x_points, y_points


def check_boxes_to_ignore(X_points, Y_points, approx_corners):
    # Input: vertices of possible box contours.
    # Output: index of unsuitable box contours to filter out.

    box_corners_to_ignore = []

    for i, x in enumerate(np.round(X_points, -1)):
        for j, y in enumerate(np.round(Y_points, -1)):
            found_corner = False
            for c in np.round_(approx_corners, -1):
                if (abs(c[0] - y) <= 20) and (abs(c[1] - x) <= 20):
                    found_corner = True
                    break
            if found_corner is False:
                box_corners_to_ignore.append([i, j])

    return box_corners_to_ignore


def check_all_corners(xs, ys, new_approx):
    # Input: vertices of box and vertices of smaller box.
    # Output: (binary) check if box is within larger box.
    corners = [[xs[0], ys[0]], [xs[0], ys[1]], [xs[1], ys[0]], [xs[1], ys[1]]]
    all_corners_in_contours = True
    for c in corners:
        f = is_corner_in_contours(c[0], c[1], new_approx)
        if f is False:
            all_corners_in_contours = False
            break
    return all_corners_in_contours


def is_corner_in_contours(x, y, approx_boxes):
    # Input: x and y coordiantes of vertex, box vertices.
    # Output: (binary) check if vertex is within box.
    found_corner = False
    for c in np.round_(approx_boxes, -1):
        if (abs(c[0] - y) <= 20) and (abs(c[1] - x) <= 20):
            found_corner = True
            break
    return found_corner


def is_box_valid(corner_tuples):
    # Input: combination of vertices.
    # Output: combinations that create boxes.
    xs = [corner_tuples[i][0] for i in range(4)]
    ys = [corner_tuples[i][1] for i in range(4)]
    if (len(np.unique(xs)) == 2) and (len(np.unique(ys)) == 2):
        return True
    else:
        return False


################
# Main Functions
################


def split_eggcard_boxes(eggcard_boxes):
    # Input: all egg card box contours.
    # Output: box contours, split into multiple box contours if neccesary.
    refined_boxes = []
    for box in eggcard_boxes:
        a = approximate_box(box)
        if len(a) == 4:
            refined_boxes.append(box)
        else:
            split_boxes = get_new_boxes(box)
            if len(split_boxes) > 0:
                refined_boxes.extend(split_boxes)
    return refined_boxes


def get_new_boxes(eggcardboxes):
    # Input: egg card box contour (supposedly consisting of more than one box)
    # Output: split egg card box contours.
    XY_tuples, XY, approx_corners = filter_for_possible_box_corners(eggcardboxes)
    wrong_corners = check_boxes_to_ignore(XY[0], XY[1], approx_corners)
    refined_boxes = []

    if len(wrong_corners) > 0:
        all_possible_corners = list(
            itertools.product(*[np.sort(XY[0]), np.sort(XY[1])])
        )
        combinations_4 = list(combinations(all_possible_corners, 4))
        valid_corners = [corners for corners in combinations_4 if is_box_valid(corners)]
        for corner_tuples in valid_corners:
            xs = [corner_tuples[i][0] for i in range(4)]
            ys = [corner_tuples[i][1] for i in range(4)]
            xs = [min(xs), max(xs)]
            ys = [min(ys), max(ys)]
            all_corners_in_contours = check_all_corners(xs, ys, approx_corners)
            if all_corners_in_contours is True:
                refined_boxes.append(
                    [
                        [xs[0], xs[0], xs[1], xs[1], xs[0]],
                        [ys[0], ys[1], ys[1], ys[0], ys[0]],
                    ]
                )
    else:
        for xs in XY_tuples[0]:
            for ys in XY_tuples[1]:
                refined_boxes.append(
                    [
                        [xs[0], xs[0], xs[1], xs[1], xs[0]],
                        [ys[0], ys[1], ys[1], ys[0], ys[0]],
                    ]
                )
    return refined_boxes
