"""
The following functions are used for the pipeline that incorporates the functions detailed in
the file box_correction_functions.py. This is to be used for "standard"-style cards only.

First, we detect the boxes and add a label onto each of them (using box_correction_functions.py).
Then, we go through each label, and extract category information from that label's box only.
This is different to previous code where we looked at all boxes, in case information was stored elsewhere.
"""

#########
# Imports
#########

from box_correction_functions import *
import os
import matplotlib.pyplot as plt
from craft_text_detector import Craft
import pytesseract
import re
from source_functions import get_text
from fuzzywuzzy import fuzz
from copy import deepcopy
from post_processing_functions import find_species_from_text, get_taxon_info

#######
# Craft
#######


output_dir = "outputs/"


def get_craft_textboxes(image_path, output_dir):
    # Textboxes with Craft
    ######################
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)
    # apply craft text detection and export detected regions to output directory
    prediction_result = craft.detect_text(image_path)

    # unload models from ram/gpu
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    return prediction_result["boxes"]


###################
# General Functions
###################


def get_boxes_and_textboxes(image_path, output_dir):
    try:
        boxes_ref, img = get_boxes_and_labels(image_path)
    except:
        boxes_ref, img = redo_boxes(image_path)

    textboxes = get_craft_textboxes(image_path, output_dir)
    inds_dict = get_box_index_for_textboxes(
        textboxes, list(boxes_ref.values()), textbox_leeway=20
    )
    return img, boxes_ref, textboxes, inds_dict


def text_from_multiple_textboxes(
    textboxes, image, inds, check_orientation=False, leeway=1, limit=3000
):
    all_text = []
    for j in inds:
        a = find_area(textboxes[j], textbox=True)
        if a < limit:
            leeway = 2
        t, _ = get_text(
            image,
            textboxes[j][:, 0],
            textboxes[j][:, 1],
            leeway=leeway,
            check_orientation=check_orientation,
        )
        all_text.append(t)
    text = " ".join(map(str, all_text))
    return text


def remove_category_from_text(keyword, text):
    n = len(re.findall("\w+", keyword))
    text = text.replace("no.", "no ").replace("No.", "No ")

    if n > 1:
        txt_ = re.findall("\w+", text)[:n]
        txt = " ".join(map(str, txt_))
    else:
        txt_ = re.findall("\w+", text)[:1]
        txt = txt_[0]

    r = fuzz.ratio(keyword, txt.lower())
    if r < 70:
        return text
    else:
        max_ = 0
        for t in txt_:
            s = re.search(t, text).span()[1]
            if s > max_:
                max_ = deepcopy(s)
        text1 = text[:max_]
        text2 = text[max_:]
        for t in txt_:
            text1 = text1.replace(t, "")

        text_ = text1 + text2

        text_ = re.findall("\S+", text_)
        final_text = " ".join(map(str, text_))

        return final_text


def remove_all_categories_from_text(
    text,
    keywords=["reg no", "locality", "collector", "date", "set mark", "no of eggs"],
    main_word="",
):
    final_text = deepcopy(text)
    for keyword in keywords:
        if keyword == main_word:
            final_text = remove_category_from_text(keyword, final_text)
        else:
            final_text = final_text.replace(keyword, "")
            final_text = final_text.replace(keyword.title(), "")

    return final_text


#####################
# Registration Number
#####################


def get_reg_number(textboxes, inds_dict, image):
    all_text = []

    for j in list(inds_dict.keys()):
        if inds_dict[j] == 0:
            b = textboxes[j]
            txt, _ = get_text(image, b[:, 0], b[:, 1], compare_numbers_count=True)
            if len(re.findall("\d+", txt)) > 0:
                all_text.append(txt)

    try:
        final_reg = ".".join(map(str, re.findall("\w+", " ".join(map(str, all_text)))))
    except:
        final_reg = deepcopy(all_text)

    return final_reg


#########
# Species
#########


def get_species_text(textboxes, boxes_ref, inds_dict, image):
    inds = [j for j in list(inds_dict.keys()) if inds_dict[j] == 1]
    if len(inds) == 1:
        b = textboxes[inds[0]]
        species_txt, _ = get_text(
            image, b[:, 0], b[:, 1], leeway=10, check_orientation=False
        )
        species = " ".join(map(str, re.findall("\w+", species_txt)))
        reg = "N/A"
    else:
        possible_species = []
        possible_reg = []
        for j in inds:
            b = textboxes[j]
            txt, _ = get_text(
                image, b[:, 0], b[:, 1], leeway=10, check_orientation=False
            )
            len_txt = len(re.findall("[a-zA-Z]", txt))
            len_num = len(re.findall("\d+", txt))
            if len_num == 0:
                possible_species.append(txt)
            else:
                possible_reg.append(txt)
        if len(possible_species) == 0:
            species_txt, _ = get_text(
                image,
                boxes_ref["species"][0],
                boxes_ref["species"][1],
                leeway=0,
                check_orientation=False,
            )
        else:
            species_txt = " ".join(map(str, possible_species))
        species = " ".join(map(str, re.findall("\w+", species_txt)))

        if len(possible_reg) == 0:
            reg = "N/A"
        else:
            reg = " ".join(
                map(str, re.findall("\w+", " ".join(map(str, possible_reg))))
            )

    return species, reg


###############################################
# Locality, Collector, Date, Set Mark, No. Eggs
###############################################


def get_text_from_category_box(
    image,
    boxes_ref,
    textboxes,
    inds_dict,
    labels,
    label_index,
    limit=25,
    min_char=5,
    leeway=1,
):
    _, box_maxx, _, _ = get_box_details(boxes_ref[labels[label_index]])
    inds = [j for j in list(inds_dict.keys()) if inds_dict[j] == label_index]
    _, xmax, _, _ = get_boundary_for_group_of_boxes(textboxes, inds, textbox=True)
    box_method = True
    if ((max(xmax) - box_maxx) > limit) or (len(inds) > 1):
        box_method = False
        text = text_from_multiple_textboxes(textboxes, image, inds, leeway=leeway)
    else:
        text, _ = get_text(
            image, boxes_ref[labels[label_index]][0], boxes_ref[labels[label_index]][1]
        )

    text = text.replace("\n", " ").replace("\x0c", "")

    if len(text) < min_char:
        if box_method is True:
            text = text_from_multiple_textboxes(textboxes, image, inds, leeway=leeway)
        else:
            text, _ = get_text(
                image,
                boxes_ref[labels[label_index]][0],
                boxes_ref[labels[label_index]][1],
            )
        text = text.replace("\n", " ").replace("\x0c", "")

    return text


################
# Other Text Box
################


def get_text_from_other_box(textboxes, inds_dict, image, combine=True):
    inds = [j for j in list(inds_dict.keys()) if inds_dict[j] == 7]
    if combine:
        boxes = refine_boxes(textboxes[inds], pixel_proximity_bound=100)
    else:
        boxes = textboxes[inds]

    text = text_from_multiple_textboxes(boxes, image, list(range(len(boxes))))

    text = text.replace("\n", " ").replace("\x0c", "")

    return text


###############
# Main Function
###############

labels = [
    "reg",
    "species",
    "locality",
    "collector",
    "date",
    "setMark",
    "noOfEggs",
    "other",
]


def get_boxes_and_textboxes_and_index(
    image_path, output_dir, leeway=30, library="sk", additional_threshing=True
):
    try:
        boxes_ref, img_sk = get_boxes_and_labels(
            image_path, library=library, additional_threshing=additional_threshing
        )
    except:
        boxes_ref, img_sk = get_boxes_and_labels(
            image_path,
            filter_boxes=False,
            library=library,
            additional_threshing=additional_threshing,
        )
    textboxes = get_craft_textboxes(image_path, output_dir)
    inds_dict = get_box_index_for_textboxes(
        textboxes, list(boxes_ref.values()), textbox_leeway=leeway
    )
    return boxes_ref, textboxes, inds_dict, img_sk


def get_all_category_text(boxes_ref, textboxes, inds_dict, image, combine_other=True):
    all_info = {}
    # 1) Registration number and Species:
    # we combine because sometimes these are in the same box.
    try:
        reg = get_reg_number(textboxes, inds_dict, image)
    except:
        reg = "N/A"
    try:
        cardSpecies, reg_backup = get_species_text(
            textboxes, boxes_ref, inds_dict, image
        )
    except:
        cardSpecies = "N/A"
        reg_backup = "N/A"
    # Registration Number
    if reg_backup != "N/A":
        c1 = re.findall("\d", reg)
        c2 = re.findall("\d", reg_backup)
        if len(c2) > len(c1):
            reg = deepcopy(reg_backup)
    reg = remove_all_categories_from_text(reg, main_word="reg no")
    all_info["registrationNumber"] = reg
    # Species
    cardSpecies = remove_all_categories_from_text(
        cardSpecies,
        keywords=["reg no", "locality", "collector", "set mark", "no of eggs"],
    )
    species_name, species_results = find_species_from_text([cardSpecies], [])
    taxon = get_taxon_info(species_name, species_results)
    all_info["cardSpecies"] = cardSpecies
    for b in taxon.keys():
        all_info[b] = taxon[b]

    # 2) Locality:
    try:
        locality = get_text_from_category_box(
            image, boxes_ref, textboxes, inds_dict, labels, 2
        )
    except:
        locality = "N/A"
    locality = remove_all_categories_from_text(locality, main_word="locality")
    all_info["locality"] = locality

    # 3) Collector:
    try:
        collector = get_text_from_category_box(
            image, boxes_ref, textboxes, inds_dict, labels, 3
        )
    except:
        collector = "N/A"
    collector = remove_all_categories_from_text(
        collector,
        keywords=["reg no", "locality", "collector", "set mark", "no of eggs"],
        main_word="collector",
    )
    all_info["collector"] = collector

    # 4) Date:
    try:
        date = get_text_from_category_box(
            image, boxes_ref, textboxes, inds_dict, labels, 4, leeway=0
        )
    except:
        date = "N/A"
    date = remove_all_categories_from_text(date, main_word="date")
    all_info["date"] = date

    # 5) Set Mark:
    try:
        setMark = get_text_from_category_box(
            image, boxes_ref, textboxes, inds_dict, labels, 5
        )
    except:
        setMark = "N/A"
    setMark = remove_all_categories_from_text(setMark, main_word="set mark")
    all_info["setMark"] = setMark

    # 6) No. Eggs:
    try:
        noOfEggs = get_text_from_category_box(
            image, boxes_ref, textboxes, inds_dict, labels, 6
        )
    except:
        noOfEggs = "N/A"
    noOfEggs = remove_all_categories_from_text(noOfEggs, main_word="no of eggs")

    all_info["noOfEggs"] = noOfEggs

    # 7) Other:
    try:
        other = get_text_from_other_box(
            textboxes, inds_dict, image, combine=combine_other
        )
    except:
        other = "N/A"
    all_info["remainingText"] = other

    return all_info
