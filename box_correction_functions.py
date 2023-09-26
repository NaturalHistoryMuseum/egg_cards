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
    get_box_details,
    detect_orientation,
    get_box_index_for_textboxes,
)
from itertools import combinations
import itertools
import cv2
from post_processing_functions import tuplewize


#####################################
# Helper Functions for Box Correction
#####################################


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


####################################
# Main Functions for Box Corrections
####################################


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


################################################################################

# Box Referencing Functions
###########################
###########################


"""
In this section we provide the code needed to label the boxes and textboxes with its associated category,
The labels are: 'reg','species','locality','collector','date','setMark','noOfEggs','other'.
"""

##########################
# General Helper Functions
##########################


def find_area(box):
    box_minx, box_maxx, box_miny, box_maxy = get_box_details(box)
    area = (box_maxx - box_minx) * (box_maxy - box_miny)
    return area


def group_boxes(egg_boxes, reg_box_ind, bound=50, area_lower_bound=500):
    group_index = {}
    k = 0
    for i, box in enumerate(egg_boxes):
        if (i != reg_box_ind) and (find_area(box) > area_lower_bound):
            avg_y = np.average(box[1])
            if len(group_index) == 0:
                group_index.update({k: {"index": [i], "vals": [avg_y]}})
            else:
                group = 0
                global_min_diff = np.inf
                for j in range(k + 1):
                    averages = group_index[j]["vals"]
                    min_diff = min([abs(y - avg_y) for y in averages])
                    if (min_diff < bound) and (min_diff < global_min_diff):
                        global_min_diff = deepcopy(min_diff)
                        group = deepcopy(j)
                if global_min_diff == np.inf:
                    k = k + 1
                    group_index.update({k: {"index": [i], "vals": [avg_y]}})
                else:
                    averages = group_index[group]["vals"]
                    index_ = group_index[group]["index"]
                    averages.append(avg_y)
                    index_.append(i)
                    group_index.update({group: {"index": index_, "vals": averages}})

    group_index_ = {}
    for i in range(k + 1):
        group_index_[i] = group_index[i]["index"]

    group_index_[k + 1] = [reg_box_ind]

    return group_index_


def get_boundary_for_group_of_boxes(egg_boxes, inds):
    min_y = []
    max_y = []
    min_x = []
    max_x = []
    for i in inds:
        box_minx, box_maxx, box_miny, box_maxy = get_box_details(egg_boxes[i])
        min_y.append(box_miny)
        max_y.append(box_maxy)
        min_x.append(box_minx)
        max_x.append(box_maxx)
    return min_x, max_x, min_y, max_y


def species_check_index_correction(egg_boxes, index, limit=100):
    new_index = deepcopy(index)
    if len(index[0]) > 1:
        mins = [min(egg_boxes[i][1]) for i in index[0]]
        min_y = min(mins)
        reg_min = min(egg_boxes[list(index.values())[-1][0]][1])
        if abs(reg_min - min_y) > limit:
            new_index = {}
            new_index[0] = ["N/A"]
            for ky in list(index.keys()):
                new_index[ky + 1] = index[ky]
    return new_index


######################
# 1) Registration Box
######################

""" 
Here, we assume that the registration number is written in the vertical box to the left of the egg card.
"""


def find_reg_box(egg_boxes):
    vert_boxes = [box for box in egg_boxes if detect_orientation(box[0], box[1]) == "v"]
    vert_boxes_inds = [
        i
        for i, box in enumerate(egg_boxes)
        if detect_orientation(box[0], box[1]) == "v"
    ]

    if len(vert_boxes) == 1:
        return vert_boxes[0], vert_boxes_inds[0]
    elif len(vert_boxes) == 0:
        return "N/A", "N/A"
    else:
        k = np.argmax([find_area(box) for box in vert_boxes])
        return vert_boxes[k], vert_boxes_inds[k]


################
# 2) Species Box
################

""" 
Here, we assume that the top horizontal box on the egg card contains the specie information.
"""


def refine_group0_box(egg_boxes, index, bound=20):
    # Group 0 category should only contain one box, and that will be the species box.
    # If more than one box is included we will check whether they are in the same range, and merge if needed.

    inds = index[0]
    min_x, max_x, min_y, max_y = get_boundary_for_group_of_boxes(egg_boxes, inds)

    exclude = []
    for j, m in enumerate(min_y):
        diff1 = [abs(m - y) for j_, y in enumerate(min_y) if j_ != j]
        diff2 = [abs(max_y[j] - y) for j_, y in enumerate(max_y) if j_ != j]
        if (min(diff1) > bound) and (min(diff2) > bound):
            exclude.append(j)
    # Remove anomalies
    if len(exclude) == len(inds):
        areas = [find_area(egg_boxes[i]) for i in inds]
        new_box = egg_boxes[inds[np.argmax(areas)]]
    else:
        for indx in sorted(exclude, reverse=True):
            min_y.pop(indx)
            max_y.pop(indx)
            min_x.pop(indx)
            max_x.pop(indx)

        new_box_x = [min(min_x), min(min_x), max(max_x), max(max_x), min(min_x)]
        new_box_y = [min(min_y), max(max_y), max(max_y), min(min_y), min(min_y)]
        new_box = [new_box_x, new_box_y]

    return new_box


def backup_species_box_method(egg_boxes, index):
    reg_box = egg_boxes[list(index.values())[-1][0]]
    x_min = max(reg_box[0])
    maxs = [max(b[0]) for b in egg_boxes]
    x_max = max(maxs)
    y_min = min(reg_box[1])
    mins = [min(egg_boxes[i][1]) for i in index[1]]
    y_max = min(mins)
    box_x = [x_min, x_min, x_max, x_max, x_min]
    box_y = [y_min, y_max, y_max, y_min, y_min]
    return [box_x, box_y]


###############################
# 3) Locality & Collector Boxes
###############################

""" 
Here, we assume that the second highest horizontal boxes are the locality and collector boxes.
"""


def refine_group1_box(egg_boxes, index, verification_bound=100, split=0.6):

    inds = index[1]
    min_x, max_x, min_y, max_y = get_boundary_for_group_of_boxes(egg_boxes, inds)

    # We check whether the x coordinates of the boundary of this group are roughly correct.
    correct = True
    if (len(index[0]) == 1) or (len(index[2]) == 3):
        if len(index[0]) == 1:
            box = egg_boxes[index[0][0]]
            box_minx, box_maxx, _, _ = get_box_details(box)
        else:
            min_x_, max_x_, _, _ = get_boundary_for_group_of_boxes(egg_boxes, index[2])
            box_minx = min(min_x_)
            box_maxx = max(max_x_)

        mn = min([abs(box_minx - m) for m in min_x])
        mx = min([abs(box_maxx - m) for m in max_x])
        if (mn > verification_bound) or (mx > verification_bound):
            correct = False

    if correct is True:
        xa = min(min_x)
        xb = max(max_x)
    else:
        xa = box_minx
        xb = box_maxx

    # We estimate that the second row boxes are split 60:40.
    x_split = xa + (split * (xb - xa))

    # Create new boxes based on new split.
    ya = min(min_y)
    yb = max(max_y)
    x1 = [xa, xa, x_split, x_split, xa]
    x2 = [x_split, x_split, xb, xb, x_split]
    y = [ya, yb, yb, ya, ya]

    return [x1, y], [x2, y]


###################################
# 4) Date, Set Mark, No. Eggs Boxes
###################################

""" 
Here, we assume that the third row of horizontal boxes are the date, set mark, and no. eggs boxes.
"""


def refine_group2_box(
    egg_boxes, index, verification_bound=100, split1=0.42, split2=0.74
):

    inds = index[2]
    min_x, max_x, min_y, max_y = get_boundary_for_group_of_boxes(egg_boxes, inds)

    # We check whether the x coordinates of the boundary of this group are roughly correct.
    correct = True
    if (len(index[0]) == 1) or (len(index[1]) == 2):
        if len(index[0]) == 1:
            box = egg_boxes[index[0][0]]
            box_minx, box_maxx, _, _ = get_box_details(box)
        else:
            min_x_, max_x_, _, _ = get_boundary_for_group_of_boxes(egg_boxes, index[1])
            box_minx = min(min_x_)
            box_maxx = max(max_x_)

        mn = min([abs(box_minx - m) for m in min_x])
        mx = min([abs(box_maxx - m) for m in max_x])
        if (mn > verification_bound) or (mx > verification_bound):
            correct = False

    if correct is True:
        xa = min(min_x)
        xb = max(max_x)
    else:
        xa = box_minx
        xb = box_maxx

    # We estimate that the third row boxes are split at the 42nd and 74th percentile.
    x_split1 = xa + (split1 * (xb - xa))
    x_split2 = xa + (split2 * (xb - xa))

    # Create new boxes based on new splits.
    ya = min(min_y)
    yb = max(max_y)
    x1 = [xa, xa, x_split1, x_split1, xa]
    x2 = [x_split1, x_split1, x_split2, x_split2, x_split1]
    x3 = [x_split2, x_split2, xb, xb, x_split2]
    y = [ya, yb, yb, ya, ya]

    return [x1, y], [x2, y], [x3, y]


################
# Main Functions
################


def verify_and_filter_boxes(egg_boxes, index):

    # 1) Regitration number box (vertical):
    reg_box_ind = index[list(index.keys())[-1]][0]
    reg_box = egg_boxes[reg_box_ind]

    # 2) Species box (highest horizontal box):
    species_ind = index[list(index.keys())[0]]
    if len(species_ind) == 1:
        if species_ind[0] != "N/A":
            species_ind = species_ind[0]
            species_box = egg_boxes[species_ind]
        else:
            species_box = backup_species_box_method(egg_boxes, index)
    else:
        species_box = refine_group0_box(egg_boxes, index)

    # 3) Locality / collector boxes (second highest horizontal boxes):
    loccol_ind = index[list(index.keys())[1]]
    if len(loccol_ind) == 2:
        box1 = egg_boxes[loccol_ind[0]]
        box2 = egg_boxes[loccol_ind[1]]
        box1_minx, _, _, _ = get_box_details(box1)
        box2_minx, _, _, _ = get_box_details(box2)
        if box1_minx < box2_minx:
            loc_box = deepcopy(box1)
            col_box = deepcopy(box2)
        else:
            loc_box = deepcopy(box2)
            col_box = deepcopy(box1)
    else:
        loc_box, col_box = refine_group1_box(egg_boxes, index)

    # 4) Date / set mark / no. eggs boxes (third highest horizontal boxes):
    datsetegg_ind = index[list(index.keys())[2]]
    if len(datsetegg_ind) == 3:
        box1 = egg_boxes[datsetegg_ind[0]]
        box2 = egg_boxes[datsetegg_ind[1]]
        box3 = egg_boxes[datsetegg_ind[2]]

        box1_minx, _, _, _ = get_box_details(box1)
        box2_minx, _, _, _ = get_box_details(box2)
        box3_minx, _, _, _ = get_box_details(box3)

        boxes_ = [box1, box2, box3]
        sorted_inds = np.argsort([box1_minx, box2_minx, box3_minx])

        date_box = boxes_[sorted_inds[0]]
        sm_box = boxes_[sorted_inds[1]]
        noegg_box = boxes_[sorted_inds[2]]

    else:
        date_box, sm_box, noegg_box = refine_group2_box(egg_boxes, index)

    # 5) Remaining text box:
    x_min = max(reg_box[0])
    x_max = max([max(species_box[0]), max(col_box[0]), max(noegg_box[0])])
    y_min = max([max(date_box[1]), max(sm_box[1]), max(noegg_box[1])])
    y_max = max(reg_box[1])
    rem_box_x = [x_min, x_min, x_max, x_max, x_min]
    rem_box_y = [y_min, y_max, y_max, y_min, y_min]
    rem_box = [rem_box_x, rem_box_y]

    # 6) Place boxes into dictionary
    all_boxes = {}
    all_boxes["reg"] = reg_box
    all_boxes["species"] = species_box
    all_boxes["locality"] = loc_box
    all_boxes["collector"] = col_box
    all_boxes["date"] = date_box
    all_boxes["setMark"] = sm_box
    all_boxes["noOfEggs"] = noegg_box
    all_boxes["other"] = rem_box

    return all_boxes


def get_boxes_and_labels(image_path):
    img_sk = io.imread(image_path)
    contours = find_all_contours(cv2.cvtColor(img_sk, cv2.COLOR_BGR2GRAY))
    # Get boundary thresholds for box definition:
    xbound, ybound = get_max_bounds(contours)
    # Filter for box contours:
    eggcard_boxes_sk = filter_contours(
        contours, xbound, ybound, additional_filters=False
    )
    egg_boxes = split_eggcard_boxes(eggcard_boxes_sk)
    reg_box, reg_box_ind = find_reg_box(egg_boxes)
    index = group_boxes(egg_boxes, reg_box_ind, bound=50)
    index = species_check_index_correction(egg_boxes, index)
    boxes_ref = verify_and_filter_boxes(egg_boxes, index)

    return boxes_ref, img_sk
