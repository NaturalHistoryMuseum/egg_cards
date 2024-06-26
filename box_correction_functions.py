##########
# Imports
##########


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
    get_textbox_details,
    load_image,
)
from itertools import combinations
import itertools
import cv2
from post_processing_functions import tuplewize


#####################################
# Helper Functions for Box Correction
#####################################


def approximate_box(box, epsilon=0.02):
    # Input: original box contours.
    # Output: polygon approximation of contour.

    tst = np.zeros((len(box[0]), 1, 2), dtype="int32")
    tst[:, 0, 0] = np.int_(box[1])
    tst[:, 0, 1] = np.int_(box[0])

    peri = cv2.arcLength(tst, True)
    approx = cv2.approxPolyDP(tst, epsilon * peri, True)

    return approx


def filter_for_possible_box_corners(eggcardboxes, epsilon=0.02):
    # Input: egg card box contour (supposedly consisting of more than one box)
    # Output: vertices of possible boxes contained within one box contour.
    approx_corners = approximate_box(eggcardboxes, epsilon=epsilon)
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


def is_corner_in_contours(x, y, approx_boxes, leeway=30):
    # Input: x and y coordiantes of vertex, box vertices.
    # Output: (binary) check if vertex is within box.
    found_corner = False
    for c in np.round_(approx_boxes, -1):
        if (abs(c[0] - y) <= leeway) and (abs(c[1] - x) <= leeway):
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


def split_eggcard_boxes(eggcard_boxes, epsilon=0.02):
    # Input: all egg card box contours.
    # Output: box contours, split into multiple box contours if neccesary.
    refined_boxes = []
    for box in eggcard_boxes:
        a = approximate_box(box)
        if len(a) == 4:
            refined_boxes.append(box)
        else:
            split_boxes = get_new_boxes(box, epsilon=epsilon)
            if len(split_boxes) > 0:
                refined_boxes.extend(split_boxes)
    return refined_boxes


def get_new_boxes(eggcardboxes, epsilon=0.02):
    # Input: egg card box contour (supposedly consisting of more than one box)
    # Output: split egg card box contours.
    XY_tuples, XY, approx_corners = filter_for_possible_box_corners(
        eggcardboxes, epsilon=epsilon
    )
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


def find_area(box, textbox=False):
    if textbox is False:
        box_minx, box_maxx, box_miny, box_maxy = get_box_details(box)
    else:
        box_minx, box_maxx, box_miny, box_maxy = get_textbox_details(box)

    area = (box_maxx - box_minx) * (box_maxy - box_miny)
    return area


def group_boxes(egg_boxes, reg_box_ind, bound=50, area_lower_bound=900):
    group_index = {}
    k = 0
    for i, box in enumerate(egg_boxes):
        h_v = detect_orientation(box[0], box[1])
        if (i != reg_box_ind) and (find_area(box) > area_lower_bound) and (h_v != "v"):
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


def get_boundary_for_group_of_boxes(egg_boxes, inds, textbox=False):
    min_y = []
    max_y = []
    min_x = []
    max_x = []
    for i in inds:
        if textbox is True:
            box_minx, box_maxx, box_miny, box_maxy = get_textbox_details(egg_boxes[i])
        else:
            box_minx, box_maxx, box_miny, box_maxy = get_box_details(egg_boxes[i])
        min_y.append(box_miny)
        max_y.append(box_maxy)
        min_x.append(box_minx)
        max_x.append(box_maxx)
    return min_x, max_x, min_y, max_y


def species_check_index_correction(egg_boxes, index, limit=120):
    new_index = deepcopy(index)
    mins = [min(egg_boxes[i][1]) for i in index[0]]
    min_y = min(mins)
    reg_min = min(egg_boxes[list(index.values())[-1][0]][1])
    if abs(reg_min - min_y) > limit:
        new_index = {}
        new_index[0] = ["N/A"]
        for ky in list(index.keys()):
            new_index[ky + 1] = index[ky]
    return new_index


def remove_spare_vertical_boxes_from_eggboxes(egg_boxes, reg_box_ind):
    all_boxes = []
    new_reg_box_ind = deepcopy(reg_box_ind)
    k = -1
    for i, box in enumerate(egg_boxes):
        h_v = detect_orientation(box[0], box[1])
        if (h_v == "h") or ((h_v == "v") and (i == reg_box_ind)):
            k = k + 1
            all_boxes.append(box)
        if i == reg_box_ind:
            new_reg_box_ind = deepcopy(k)
    return all_boxes, new_reg_box_ind


def sort_boxes_in_y_order(egg_boxes):
    _, _, ymin, _ = get_boundary_for_group_of_boxes(
        egg_boxes, list(range(len(egg_boxes)))
    )
    egg_boxes_sorted = []
    for ind in np.argsort(ymin):
        egg_boxes_sorted.append(egg_boxes[ind])
    return egg_boxes_sorted


def does_box_contain_other_box(box, other_box, leeway=0):
    box_minx, box_maxx, box_miny, box_maxy = get_box_details(box)
    minx, maxx, miny, maxy = get_box_details(other_box)

    if all(
        (
            (miny >= box_miny - leeway),
            (maxy <= box_maxy + leeway),
            (minx >= box_minx - leeway),
            (maxx <= box_maxx + leeway),
        )
    ):
        return True
    else:
        return False


def remove_boxes_containing_other_boxes(egg_boxes, area_limit=5000):
    egg_boxes_new = []

    for i, box in enumerate(egg_boxes):
        contains_box = False
        for j, other_box in enumerate(egg_boxes):
            if j != i:
                c = does_box_contain_other_box(box, other_box, leeway=0)
                if c is True:
                    a = find_area(other_box)
                    if a > area_limit:
                        contains_box = True
                        break
        if contains_box is False:
            egg_boxes_new.append(box)
    return egg_boxes_new


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
    if ((len(index[0]) == 1) and (index[0][0] != "N/A")) or (len(index[2]) == 3):
        if (len(index[0]) == 1) and (index[0][0] != "N/A"):
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
    if ((len(index[0]) == 1) and (index[0][0] != "N/A")) or (len(index[1]) == 2):
        if (len(index[0]) == 1) and (index[0][0] != "N/A"):
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


def get_boxes_and_labels(
    image_path,
    lowerbound=6,
    minimum_lower_bound=3,
    filter_boxes=True,
    library="sk",
    additional_filters=False,
    additional_threshing=True,
):
    img_sk, I = load_image(image_path, library=library)
    contours = find_all_contours(I, additional_threshing=additional_threshing)
    # Get boundary thresholds for box definition:
    xbound, ybound = get_max_bounds(contours)
    # Filter for box contours:
    eggcard_boxes_sk = filter_contours(
        contours, xbound, ybound, additional_filters=additional_filters
    )
    egg_boxes = split_eggcard_boxes(eggcard_boxes_sk)
    if len(egg_boxes) <= minimum_lower_bound:
        # If the amount of boxes found is too low, we estimate all 8 boxes with backup function.
        egg_boxes = redo_boxes(image_path, indexing=False)
        eggcard_boxes_sk = deepcopy(egg_boxes)

    if len(egg_boxes) <= lowerbound:
        egg_boxes_ = split_eggcard_boxes(eggcard_boxes_sk, epsilon=0.01)
        if len(egg_boxes_) > len(egg_boxes):
            egg_boxes = deepcopy(egg_boxes_)
    if filter_boxes:
        egg_boxes = remove_boxes_containing_other_boxes(egg_boxes)
    egg_boxes_sorted = sort_boxes_in_y_order(egg_boxes)
    _, reg_box_ind = find_reg_box(egg_boxes_sorted)
    if reg_box_ind == "N/A":
        new_reg_box, new_reg_box_ind = find_reg_box(eggcard_boxes_sk)
        if new_reg_box_ind != "N/A":
            egg_boxes_sorted.insert(0, new_reg_box)
            reg_box_ind = 0
    egg_boxes_refined, reg_box_ind = remove_spare_vertical_boxes_from_eggboxes(
        egg_boxes_sorted, reg_box_ind
    )
    index = group_boxes(egg_boxes_refined, reg_box_ind, bound=50)
    index = species_check_index_correction(egg_boxes_refined, index)
    boxes_ref = verify_and_filter_boxes(egg_boxes_refined, index)

    return boxes_ref, img_sk


##########################
# Backup Outline Functions
##########################


def estimate_boxes(
    xbounds, ybounds, average_ratios=[0.115, 0.14, 0.30, 0.40, 0.61, 0.47, 0.75]
):
    # Approximates the outline contours of boxes within an egg card based on average measurements.

    avg_a, avg_b, avg_c, avg_d, avg_e, avg_f, avg_g = average_ratios

    # 1) Reg no. box:
    reg_r = xbounds[0] + (abs(xbounds[1] - xbounds[0]) * avg_a)
    reg_box = [
        [xbounds[0], xbounds[0], reg_r, reg_r, xbounds[0]],
        [ybounds[0], ybounds[1], ybounds[1], ybounds[0], ybounds[0]],
    ]

    # 2) Species box:
    species_r = ybounds[0] + (abs(ybounds[1] - ybounds[0]) * avg_b)

    species_box = [
        [reg_r, reg_r, xbounds[1], xbounds[1], reg_r],
        [ybounds[0], species_r, species_r, ybounds[0], ybounds[0]],
    ]

    # 3) Locality box:
    locality_rx = xbounds[0] + (abs(xbounds[1] - xbounds[0]) * avg_e)
    locality_ry = ybounds[0] + (abs(ybounds[1] - ybounds[0]) * avg_c)

    locality_box = [
        [reg_r, reg_r, locality_rx, locality_rx, reg_r],
        [species_r, locality_ry, locality_ry, species_r, species_r],
    ]

    # 4) Collector box:
    collector_box = [
        [locality_rx, locality_rx, xbounds[1], xbounds[1], locality_rx],
        [species_r, locality_ry, locality_ry, species_r, species_r],
    ]

    # 5) Date box:
    date_ry = ybounds[0] + (abs(ybounds[1] - ybounds[0]) * avg_d)
    date_rx = xbounds[0] + (abs(xbounds[1] - xbounds[0]) * avg_f)
    date_box = [
        [reg_r, reg_r, date_rx, date_rx, reg_r],
        [locality_ry, date_ry, date_ry, locality_ry, locality_ry],
    ]

    # 6) Set mark box:
    set_rx = xbounds[0] + (abs(xbounds[1] - xbounds[0]) * avg_g)
    set_box = [
        [date_rx, date_rx, set_rx, set_rx, date_rx],
        [locality_ry, date_ry, date_ry, locality_ry, locality_ry],
    ]

    # 7) Number of eggs box:
    egg_box = [
        [set_rx, set_rx, xbounds[1], xbounds[1], set_rx],
        [locality_ry, date_ry, date_ry, locality_ry, locality_ry],
    ]

    # 8) Other text box:
    other_box = [
        [reg_r, reg_r, xbounds[1], xbounds[1], reg_r],
        [date_ry, ybounds[1], ybounds[1], date_ry, date_ry],
    ]

    # Sort to box:
    egg_boxes_new = []

    egg_boxes_new.append([np.array(reg_box[0]), np.array(reg_box[1])])
    egg_boxes_new.append([np.array(species_box[0]), np.array(species_box[1])])
    egg_boxes_new.append([np.array(locality_box[0]), np.array(locality_box[1])])
    egg_boxes_new.append([np.array(collector_box[0]), np.array(collector_box[1])])
    egg_boxes_new.append([np.array(date_box[0]), np.array(date_box[1])])
    egg_boxes_new.append([np.array(set_box[0]), np.array(set_box[1])])
    egg_boxes_new.append([np.array(egg_box[0]), np.array(egg_box[1])])
    egg_boxes_new.append([np.array(other_box[0]), np.array(other_box[1])])

    return egg_boxes_new


def redo_boxes(
    image_path,
    average_ratios=[0.115, 0.14, 0.30, 0.40, 0.61, 0.47, 0.75],
    indexing=True,
):
    # Input: image path
    # Output: image, 8 box outlines.
    # This function is a backup version of get_boxes_and_labels.

    # 1) Load image
    img = io.imread(image_path)

    # 2) Estimate main box corners:
    contours = find_all_contours(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), binarize=True)
    xbounds, ybounds = get_max_bounds(contours, return_minmax=True)

    if (ybounds[0] < 15) and (ybounds[1] > np.shape(img)[0] - 15):
        ybounds = [50, np.shape(img)[0] - 50]

    # 3) Estimate box contours:
    egg_boxes_new = estimate_boxes(xbounds, ybounds, average_ratios=average_ratios)

    if indexing:
        # 4) Sort boxes:
        egg_boxes_sorted = sort_boxes_in_y_order(egg_boxes_new)

        # 5) Index boxes:
        _, reg_box_ind = find_reg_box(egg_boxes_sorted)
        index = group_boxes(egg_boxes_sorted, reg_box_ind, bound=50)
        index = species_check_index_correction(egg_boxes_sorted, index)
        boxes_ref = verify_and_filter_boxes(egg_boxes_sorted, index)

        return boxes_ref, img
    else:
        return egg_boxes_new
