"""

EGG CARDS
----------

Main functions to find boxes within egg cards and extract text information.
Steps:
1. Load image.
2. Find all contours.
3. Filter for contours we assume are the main boxes within the egg card.
4. Use CRAFT detect textboxes within boxes.
5. Classify boxes / textboxes as "vertical" or "horizontal".
6. Combine text boxes that are close to each other.
7. Extract text with Tesseract.
8. Save into Pandas dataframe.
"""

###########
# Imports
###########

import numpy as np
import skimage.io as io
import pandas as pd
from skimage import measure
from skimage.filters import threshold_otsu
from copy import deepcopy
import cv2
import imutils
from skimage import feature
import pytesseract
import re
import matplotlib.pyplot as plt


#########
# CRAFT
#########

from craft_text_detector import Craft

output_dir = "outputs/"

craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# # unload models from ram/gpu
# craft.unload_craftnet_model()
# craft.unload_refinenet_model()


############
# Functions
############


def load_image(image_path, library="sk"):
    # Input: path to egg card image.
    # Output: original image; grey-scaled image.
    if library == "sk":
        image = io.imread(image_path)
    else:
        image = cv2.imread(image_path)
    if np.shape(image)[-1] == 4:
        if len(np.unique(image[:, :, -1].flatten())):
            image = image[:, :, :-1]
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, image_grey


def find_all_contours(
    image_grey_original,
    binarize=False,
    thresh=0.8,
    additional_threshing=False,
    percentile_bound=20,
):
    image_grey = deepcopy(image_grey_original)
    # Input: grey-scaled image.
    # Output: contours in image, found with Marching Squares method.
    if additional_threshing:
        I = deepcopy(image_grey)
        image_grey[np.where(I < np.percentile(I.flatten(), percentile_bound))] = 0

    if binarize is False:
        image = image_grey / 255
    else:
        thresh = threshold_otsu(image_grey)
        image = image_grey > thresh
        image = 255 * image.astype("uint8")

    try:
        contours = measure.find_contours(image, thresh)
    except:
        image = 255 * image.astype("uint8")
        contours = measure.find_contours(image, thresh)
    return contours


def get_max_bounds(contours, xbound=10, ybound=12, return_minmax=False):
    # Input: contours
    # Output: x,y bounds to filter for box contours.
    all_x_coords = []
    for contour in contours:
        all_x_coords.extend(contour[:, 1])

    all_y_coords = []
    for contour in contours:
        all_y_coords.extend(contour[:, 0])

    min_x, max_x = [min(all_x_coords), max(all_x_coords)]
    min_y, max_y = [min(all_y_coords), max(all_y_coords)]

    xb = (max_x - min_x) / xbound
    yb = (max_y - min_y) / ybound

    if return_minmax:
        return [min_x, max_x], [min_y, max_y]
    else:
        return xb, yb


def round_down(number):
    n_floor = np.floor(number)
    return int(str(n_floor)[0]) * (10 ** (len(str(int(n_floor))) - 1))


def filter_contours(
    contours,
    xbound,
    ybound,
    min_contour_length=50,
    additional_filters=True,
    additional_factor_bound=1,
):
    # Input: contours, xbound, ybound
    # Output: contours around boxes within egg card.
    area_lower_bound = round_down((xbound * ybound) / additional_factor_bound)
    final_contours = []
    for contour in contours:
        a = cv2.contourArea(contour.astype(int))
        if (len(contour[:, 0]) > min_contour_length) and (a > area_lower_bound):
            range_x = max(contour[:, 1]) - min(contour[:, 1])
            range_y = max(contour[:, 0]) - min(contour[:, 0])
            if additional_filters is True:
                if (range_x > xbound) or (range_y > ybound):
                    X, Y = contour[:, 1], contour[:, 0]
                    if X[0] == X[-1]:
                        final_contours.append([X, Y])
            else:
                if (range_x > xbound) or (range_y > ybound):
                    X, Y = contour[:, 1], contour[:, 0]
                    final_contours.append([X, Y])
    return final_contours


def get_craft_textboxes(image_path):
    # Input: image path.
    # Output: textboxes within image.
    prediction_result = craft.detect_text(image_path)
    return prediction_result["boxes"]


def detect_orientation(x, y):
    # Classify whether a box/textbox is horizontal or vertical
    # Input: x coordinates of box contour, y coordinates of box contour.
    # Ouput: "h" for "horizontal" classification, "v" for "vertical".
    rx = max(x) - min(x)
    ry = max(y) - min(y)
    if rx > ry:
        orientation = "h"
    else:
        orientation = "v"
    return orientation


def crop_image(image, box_x, box_y, leeway=0):
    # Input: image, x coordinates of box contours, y coordiantes of box contours, additional leeway (border).
    # Output: cropped image.
    mY, mX = np.shape(image)[:2]
    minx = max([int(min(box_x)) - leeway, 0])
    maxx = min([int(max(box_x)) + leeway, mX])
    miny = max([int(min(box_y)) - leeway, 0])
    maxy = min([int(max(box_y)) + leeway, mY])

    img_cropped = image[miny:maxy, minx:maxx]
    return img_cropped


def get_text(
    image,
    box_x,
    box_y,
    leeway=0,
    binarize=False,
    check_orientation=True,
    compare_numbers_count=False,
):
    # Input: image, x coordinates of box contour, y coordinates of box contour, leeway for cropping.
    # Output: text, cropped image.

    # 1) Classify box as horizontal or vertical:
    horizontal_or_vertical = detect_orientation(box_x, box_y)
    # 2) Crop image:
    image_cropped = crop_image(image, box_x, box_y, leeway=leeway)
    if binarize is True:
        thresh = threshold_otsu(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY))
        image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY) > thresh

    # 3) Find text:
    # If box is horizontal, we assume text is written the right way up.
    # For vertical boxes, we take the longest text found when rotated.

    if (horizontal_or_vertical == "h") or (check_orientation is False):
        ocr_results = tesseract_ocr(image_cropped)
    else:
        # Rotate 90 degrees:
        rotated_image = rotate_image(image_cropped, 90)
        ocr_results = tesseract_ocr(rotated_image)
        # Rotate 270 degrees:
        rotated_image_270 = rotate_image(image_cropped, 270)
        ocr_results_270 = tesseract_ocr(rotated_image_270)
        if compare_numbers_count is False:
            if len(ocr_results_270) > len(ocr_results):
                ocr_results = deepcopy(ocr_results_270)
                rotated_image = deepcopy(rotated_image_270)
        else:
            c1 = re.findall("\d", ocr_results)
            c2 = re.findall("\d", ocr_results_270)
            if len(c2) > len(c1):
                ocr_results = deepcopy(ocr_results_270)
                rotated_image = deepcopy(rotated_image_270)

        image_cropped = deepcopy(rotated_image)

    return ocr_results, image_cropped


def rotate_image(image, angle):
    # Input: image, angle to rotate image by.
    # Output: rotated image.
    try:
        rotated_image = imutils.rotate_bound(image, angle)
    except:
        rotated_image = imutils.rotate_bound(np.float_(image), angle)
    return rotated_image


def tesseract_ocr(image):
    # Input: image to perform OCR on.
    # Output: extracted text.
    try:
        ocr = pytesseract.image_to_string(image, config="--psm 13 script=Latin")
    except:
        ocr = pytesseract.image_to_string(
            np.uint8(image), config="--psm 13 script=Latin"
        )
    return ocr


def get_textbox_details(box):
    # Input: textbox coordinates (CRAFT output)
    # Output: reformatted box corners.
    X, Y = box[:, 0], box[:, 1]
    minx = min(X)
    maxx = max(X)
    miny = min(Y)
    maxy = max(Y)
    return minx, maxx, miny, maxy


def get_box_details(c):
    # Input: box coordinates (contour extraction output)
    # Output: reformatted box corners.
    minx = min(c[0])
    maxx = max(c[0])
    miny = min(c[1])
    maxy = max(c[1])
    return minx, maxx, miny, maxy


def find_neighbouring_boxes(
    all_boxes, pixel_proximity_bound=110, pixel_proximity_bound_x=700
):
    # Input: all textboxes, proximity bound (to define closeness between boxes)
    # Output: Maximum y coordinates of textboxes, and index of textboxes (grouped based on proximity to one another).
    all_y = []
    all_x = []
    all_inds = []

    for u, box in enumerate(all_boxes):
        new_join = False
        _, maxx, _, maxy = get_textbox_details(box)
        for v, ys in enumerate(all_y):
            xs = all_x[v]
            join = False
            for i, y in enumerate(ys):
                x = xs[i]
                if (abs(y - maxy) <= pixel_proximity_bound) and (
                    abs(x - maxx) <= pixel_proximity_bound_x
                ):
                    join = True
                    break
            if join is True:
                ys.append(maxy)
                xs.append(maxx)
                inds = all_inds[v]
                inds.append(u)
                all_y[v] = ys
                all_x[v] = xs
                all_inds[v] = inds
                new_join = True
                break
        if new_join is False:
            all_y.append([maxy])
            all_x.append([maxx])
            all_inds.append([u])
    return all_y, all_inds


def check_orientation_of_neighbouring_boxes(all_boxes, all_inds):
    new_all_inds = []
    for inds in all_inds:
        orientations = []
        for i in inds:
            box = all_boxes[i]
            orientations.append(detect_orientation(box[:, 0], box[:, 1]))
        if len(np.unique(orientations)) == 1:
            new_all_inds.append(inds)
        else:
            vertical_boxes_inds = np.where(np.array(orientations) == "v")[0]
            vertical_boxes_inds = list(np.array(inds)[vertical_boxes_inds])
            horizontal_boxes_inds = np.where(np.array(orientations) == "h")[0]
            horizontal_boxes_inds = list(np.array(inds)[horizontal_boxes_inds])
            new_all_inds.append(vertical_boxes_inds)
            new_all_inds.append(horizontal_boxes_inds)
    return new_all_inds


def combine_boxes(all_boxes, all_inds):
    # Input: Textboxes, Index of grouped textboxes (based on proximity)
    # Output: Possibly reformatted textboxes (merged if close to one another).
    new_all_boxes = []
    # Note: we check if the grouped textboxes have the same orientation:
    all_inds = check_orientation_of_neighbouring_boxes(all_boxes, all_inds)

    for u, inds in enumerate(all_inds):
        if len(inds) == 1:
            box = all_boxes[inds[0]]
            new_all_boxes.append(box)
        else:
            minx, maxx, miny, maxy = get_textbox_details(all_boxes[inds[0]])
            for i in inds[1:]:
                box = all_boxes[i]
                box_minx, box_maxx, box_miny, box_maxy = get_textbox_details(box)
                if box_minx < minx:
                    minx = deepcopy(box_minx)
                if box_maxx > maxx:
                    maxx = deepcopy(box_maxx)
                if box_miny < miny:
                    miny = deepcopy(box_miny)
                if box_maxy > maxy:
                    maxy = deepcopy(box_maxy)
            new_box = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
            new_all_boxes.append(new_box)
    return new_all_boxes


def refine_boxes(all_boxes, pixel_proximity_bound=110):
    # Check whether any boxes need merging and if so, combine those.
    # Input: textboxes, proximity bound (to define closeness between boxes).
    # Output: Possibly reformatted textboxes (merged if close to one another).
    all_y, all_y_inds = find_neighbouring_boxes(
        all_boxes, pixel_proximity_bound=pixel_proximity_bound
    )
    if len(all_y_inds) == len(all_boxes):
        return all_boxes
    else:
        new_boxes = combine_boxes(all_boxes, all_y_inds)
        return new_boxes


def get_box_index_for_textboxes(
    craft_textboxes, eggcard_boxes, textbox_leeway=10, show_NA=False
):
    # Input: boxe contours, textboxes (from CRAFT), leeway (in pixels) for textbox.
    # Output: dictionary of box index for each textbox.
    textbox_box_index = {}
    all_box_details = [get_box_details(box_contour) for box_contour in eggcard_boxes]
    areas = [(b[1] - b[0]) * (b[3] - b[2]) for b in all_box_details]
    biggest_box_ind = np.argmax(areas)

    for u, box in enumerate(craft_textboxes):
        textbox_minx, textbox_maxx, textbox_miny, textbox_maxy = get_textbox_details(
            box
        )
        box_size = np.inf
        if show_NA is False:
            box_index = deepcopy(biggest_box_ind)
        else:
            box_index = "N/A"
        for index, box_contour in enumerate(eggcard_boxes):
            box_minx, box_maxx, box_miny, box_maxy = all_box_details[index]
            area = areas[index]
            if all(
                (
                    (textbox_minx >= box_minx - textbox_leeway),
                    (textbox_maxx <= box_maxx + textbox_leeway),
                    (textbox_miny >= box_miny - textbox_leeway),
                    (textbox_maxy <= box_maxy + textbox_leeway),
                )
            ) and (area < box_size):
                box_index = deepcopy(index)
                box_size = deepcopy(area)
        textbox_box_index[u] = box_index

    return textbox_box_index


def get_text_from_box(image, textboxes, cropping_leeway=10, binarize=False):
    # Input: image, box contour, filtered textboxes (from CRAFT), textbox leeway, cropping leeway.
    # Output: extracted text from box.

    final_text = []

    # Refine textboxes (combine neighbouring ones):
    textboxes = refine_boxes(textboxes)

    # Loop through textboxes:
    all_h_or_v = []

    for box in textboxes:
        # Get text box details:
        horizontal_or_vertical = detect_orientation(box[:, 0], box[:, 1])
        all_h_or_v.append(horizontal_or_vertical)

    if "v" not in all_h_or_v:
        if len(textboxes) == 1:
            # If there's only one horizontal text box within a box, use textbox for ocr.
            box = textboxes[0]
            ocr, I = get_text(
                image, box[:, 0], box[:, 1], leeway=cropping_leeway, binarize=binarize
            )
            ocr = ocr.replace("\n", " ").replace("\x0c", "")
            final_text.append(ocr)
        else:
            all_text = []
            for box in textboxes:
                ocr, I = get_text(
                    image,
                    box[:, 0],
                    box[:, 1],
                    leeway=cropping_leeway,
                    binarize=binarize,
                )
                ocr = ocr.replace("\n", " ").replace("\x0c", "")
                all_text.append(ocr)
            final_text.append(all_text)
    else:
        all_text = []
        for box in textboxes:
            ocr, I = get_text(
                image, box[:, 0], box[:, 1], leeway=cropping_leeway, binarize=binarize
            )
            ocr = ocr.replace("\n", " ").replace("\x0c", "").replace("|", "")
            all_text.append(ocr)
        final_text.append(all_text)

    return final_text


#######################################################

cols = [
    "k",
    "g",
    "y",
    "r",
    "b",
    "m",
    "c",
    "brown",
    "salmon",
    "gray",
    "darkgreen",
    "indigo",
    "olive",
]


def plot_boxes_and_textboxes(
    image, contours, textboxes, textbox_box_index, path_to_save
):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image)
    try:
        for i, c in enumerate(contours):
            # Get all textboxes within box:
            indexes = np.where(np.array(list(textbox_box_index.values())) == i)[0]
            if len(indexes) > 0:
                filtered_textboxes = textboxes[indexes]
                # Plot box:
                horizontal_or_vertical = detect_orientation(c[0], c[1])
                if horizontal_or_vertical == "h":
                    pattern = "-"
                else:
                    pattern = "--"
                ax.plot(c[0], c[1], linestyle=pattern, linewidth=2, color=cols[i])
                # Plot textboxes:
                for j, box in enumerate(filtered_textboxes):
                    horizontal_or_vertical = detect_orientation(box[:, 0], box[:, 1])
                    if horizontal_or_vertical == "h":
                        pattern = "-"
                    else:
                        pattern = "--"
                    ax.plot(
                        box[:, 0],
                        box[:, 1],
                        linestyle=pattern,
                        color=cols[i],
                        linewidth=1,
                    )
    except:
        pass

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.savefig(path_to_save, dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close("all")


###############
# ONE FUNCTION
###############

box_plot_outdir = "/home/arias1/Documents/GitHub/egg_cards/Images/Drawer_41/plots"


def extract_text_from_eggcard(image_path, plot_boxes=False, binarize=False):
    # Input: image path
    # Output: text from image

    # 1) Load image:
    image, image_grey = load_image(image_path)
    # 2) Find all contours:
    contours = find_all_contours(image_grey)
    # 3) Get boundary thresholds for box definition:
    xbound, ybound = get_max_bounds(contours)
    # 4) Filter for box contours:
    eggcard_boxes = filter_contours(contours, xbound, ybound)
    # 5) Find textboxes with CRAFT:
    craft_textboxes = get_craft_textboxes(image_path)
    all_text = []
    # 6) Get textbox/box index:
    textbox_box_index = get_box_index_for_textboxes(
        craft_textboxes, eggcard_boxes, textbox_leeway=10
    )
    # 7) Get text from boxes:
    for box_index, box_contour in enumerate(eggcard_boxes):
        indexes = np.where(np.array(list(textbox_box_index.values())) == box_index)[0]
        if len(indexes) > 0:
            filtered_textboxes = craft_textboxes[indexes]
            try:
                text = get_text_from_box(image, filtered_textboxes, binarize=binarize)
                all_text.append(text)
            except:
                all_text.append("N/A")
    # 8) Plot boxes:
    if plot_boxes is True:
        image_id = re.findall("\w+.jpg", image_path)[0]
        plot_path = box_plot_outdir + "/" + image_id
        plot_boxes_and_textboxes(
            image, eggcard_boxes, craft_textboxes, textbox_box_index, plot_path
        )

    return all_text
