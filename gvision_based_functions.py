from text_extraction_functions import *
from copy import deepcopy
from post_processing_functions import (
    updated_find_species_results,
    count_blanks_from_taxon,
    backup_species_extraction_by_combining_words,
)
from skimage import measure
import json
from fuzzywuzzy import fuzz

##################
# Basic Functions
##################


def get_block_vertices(vertices):
    # Input: textbox vertices dictionary from Google Vision.
    # Output: vertices in list format.
    X = [v["x"] for v in vertices]
    Y = [v["y"] for v in vertices]
    return [X, Y]


def load_vision_json(path):
    # Input: path to Google Vision json output.
    # Output: dictionary of responses.
    f = open(path)
    data = json.load(f)
    vision_response = data["responses"][0]
    return vision_response


#########################
# Google Vision Textboxes
#########################


def get_boxes_of_main_categories(
    text_annotations,
    key_terms=["Date", "Locality", "Set", "Collector", "No", "No.", "No.of", "Eggs"],
):
    # Input: Google Vision text responses.
    # Output: textboxes around box titles e.g. "Collector", "Date" etc.
    vertices_all = {}
    k = 0
    found_terms = []
    for annotation in text_annotations:
        description = annotation.get("description", "")
        bounding_poly = annotation.get("boundingPoly", {}).get("vertices", [])

        if bounding_poly:
            if description in key_terms:
                # Process bounding box coordinates
                # Draw bounding box using these coordinates
                x, y = get_block_vertices(bounding_poly)
                if description not in found_terms:
                    vertices_all.update({k: {"x": x, "y": y, "text": description}})
                    k = k + 1
                    found_terms.append(description)
                else:
                    t = [
                        p
                        for p in vertices_all.keys()
                        if vertices_all[p]["text"] == description
                    ][0]
                    min_y_orig = min(vertices_all[t]["y"])
                    if min(y) < min_y_orig:
                        vertices_all[t]["x"] = x
                        vertices_all[t]["y"] = y

    return vertices_all


def check_gvision_vertices(vertices_all, text_annotations, fuzzy_bound=90):
    # Input: Output of get_boxes_of_main_categories, Google Vision text responses, minimum fuzzy bound.
    # Output: textboxes around box titles e.g. "Collector", "Date" etc (refined to find missing categories).
    terms = ["Date", "Locality", "Set", "Collector", "Eggs"]
    found_text = [vertices_all[k]["text"] for k in vertices_all.keys()]
    missing_terms = [term for term in terms if term not in found_text]

    k = len(vertices_all)
    found_terms = []
    vertices_all_new = deepcopy(vertices_all)

    for annotation in text_annotations[1:]:
        description = annotation.get("description", "")
        bounding_poly = annotation.get("boundingPoly", {}).get("vertices", [])

        if bounding_poly:
            for term in missing_terms:
                if (fuzz.ratio(term, description) > fuzzy_bound) or (
                    term in description
                ):
                    x, y = get_block_vertices(bounding_poly)
                    description = deepcopy(term)
                    if description not in found_terms:
                        vertices_all_new.update(
                            {k: {"x": x, "y": y, "text": description}}
                        )
                        k = k + 1
                        found_terms.append(description)
                    else:
                        t = [
                            p
                            for p in vertices_all_new.keys()
                            if vertices_all_new[p]["text"] == description
                        ][0]
                        min_y_orig = min(vertices_all_new[t]["y"])
                        if min(y) < min_y_orig:
                            vertices_all_new[t]["x"] = x
                            vertices_all_new[t]["y"] = y

    all_found = True
    if len(missing_terms) != len(found_terms):
        all_found = False

    return vertices_all_new, all_found


def get_centre_categories(vertices_all, pixel_bound=50):
    # Input: textboxes from main category titles.
    # Output: dictionary of textbox vertices from main category titles.
    vertices_main = {}

    # Find coordinate for "Eggs" in "No. of Eggs" box:
    egg_box = [p for p in vertices_all.keys() if vertices_all[p]["text"] == "Eggs"]
    egg_box = vertices_all[egg_box[0]]
    y_egg_avg = np.average(egg_box["y"])

    k = len(vertices_all)

    no = "No"

    for j in range(k):
        text = vertices_all[j]["text"]
        x = vertices_all[j]["x"]
        y = vertices_all[j]["y"]
        if text not in ["No", "No.", "No.of", "Eggs"]:
            vertices_main.update({text: {"x": x, "y": y}})
        elif text in ["No.", "No", "No.of"]:
            y_avg = np.average(y)
            if abs(y_avg - y_egg_avg) < pixel_bound:
                no = deepcopy(text)
                vertices_main.update({text: {"x": x, "y": y}})

    vertices_main_ = deepcopy(vertices_main)
    try:
        vertices_main_["Egg"] = vertices_main[no]
    except:
        # Approxiamte box around the start of "No. of Eggs"
        x = egg_box["x"]
        y = egg_box["y"]
        x = [max(p, 0) for p in list(np.array(x) - (max(x) - min(x)))]
        vertices_main_.update({"Egg": {"x": x, "y": y}})

    return vertices_main_


########################
# Box Contour Functions
########################

# For mid-line #


def reformat_image(
    image, vertices_main, text_annotations, boxx, boxy, bound=75, percentile_bound=40
):
    # Input: original image, vertices of main textboxes, textboxes around all text, contour around main card.
    # Output: modified greyscale image, with top half completely white, and all textboxes turned white.
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I = deepcopy(img_grey)
    img_grey[np.where(I > np.percentile(I.flatten(), percentile_bound))] = 255

    try:
        y_lim = max(
            [
                max(vertices_main["Date"]["y"]),
                max(vertices_main["Set"]["y"]),
                max(vertices_main["Egg"]["y"]),
            ]
        )
    except:
        y_lim = max(vertices_main["Date"]["y"])

    img_grey[: int(y_lim - 1), :] = 255
    img_grey[:, : int(min(vertices_main["Date"]["x"]) - 1)] = 255
    img_grey[int(max(boxy) - bound) :, :] = 255
    img_grey[:, int(max(boxx) - bound) :] = 255

    img_grey_new = deepcopy(img_grey)
    for i, annotation in enumerate(text_annotations):
        if i > 0:
            bounding_poly = annotation.get("boundingPoly", {}).get("vertices", [])
            x, y = get_block_vertices(bounding_poly)
            img_grey_new[int(min(y)) : int(max(y)), int(min(x)) : int(max(x))] = 255

    return img_grey_new


def find_midline(grey_image, max_y_diff=100, thresh=0.8, format=True):
    # Input: modified greyscale image from reformat_image.
    # Output: coordinate of the middle line separating the main catergory boxes with the "other text" part of card.
    if format:
        grey_image = grey_image / 255

    contours = measure.find_contours(grey_image, thresh)

    c_ = [max(c[:, 1]) - min(c[:, 1]) for c in contours]
    c = contours[np.argmax(c_)]

    x = c[:, 1]
    y = c[:, 0]

    x_ = np.sort(x)
    y_ = np.array(y)[np.argsort(x)]

    x_mid = [x_[0], x_[-1]]
    y_mid = [y_[0], y_[-1]]

    if abs(y_mid[0] - y_mid[1]) < max_y_diff:
        mx = max(y_mid)
        y_mid = [mx, mx]

    return x_mid, y_mid


# For main box #


def get_main_box_contour(vision_response, boundx=25, boundy=25):
    # Input: Google Vision responses.
    # Output: Coordinates of box around card.
    X, Y = get_block_vertices(
        vision_response["textAnnotations"][0]["boundingPoly"]["vertices"]
    )
    width = vision_response["fullTextAnnotation"]["pages"][0]["width"]
    height = vision_response["fullTextAnnotation"]["pages"][0]["height"]

    minx = max([min(X) - boundx, 0])
    maxx = min([max(X) + boundx, width])
    miny = max([min(Y) - boundy, 0])
    maxy = min([max(Y) + boundy, height])

    return [minx, minx, maxx, maxx, minx], [miny, maxy, maxy, miny, miny]


# For reg box #


def v_get_reg_box(vertices_main, boxx, boxy, bound=30):
    # Input: vertices of textboxes around main category titles; main box contour.
    # Output: contour around registration number box (vertical, left-side box)
    x1 = min(vertices_main["Locality"]["x"]) - bound
    x2 = min(vertices_main["Date"]["x"]) - bound
    return [min(boxx), min(boxx), x2, x1, min(boxx)], [
        min(boxy),
        max(boxy),
        max(boxy),
        min(boxy),
        min(boxy),
    ]


# For species box #


def v_get_species_box(vertices_main, boxx, boxy, bound=25):
    # Input: vertices of textboxes around main category titles; main box contour.
    # Output: contour around species box.
    x = min(vertices_main["Locality"]["x"]) - bound
    y1 = min(vertices_main["Locality"]["y"]) - bound
    y2 = min(vertices_main["Collector"]["y"]) - bound
    return [x, x, max(boxx), max(boxx), x], [min(boxy), y1, y2, min(boxy), min(boxy)]


# For locality box #


def v_get_locality_box(vertices_main, bound=25):
    # Input: vertices of textboxes around main category titles.
    # Output: contour around locality box.
    x1 = min(vertices_main["Locality"]["x"]) - bound
    x2 = min(vertices_main["Date"]["x"]) - bound
    x3 = min(vertices_main["Collector"]["x"]) - bound
    y1 = min(vertices_main["Locality"]["y"]) - bound
    y2 = min(vertices_main["Date"]["y"]) - bound
    y3 = min(vertices_main["Set"]["y"]) - bound
    y4 = min(vertices_main["Collector"]["y"]) - bound

    return [x1, x2, x3, x3, x1], [y1, y2, y3, y4, y1]


# For collector box #


def v_get_collector_box(vertices_main, boxx, bound=25):
    # Input: vertices of textboxes around main category titles; x coordinates of main box contour.
    # Output: contour around collector box.
    x = min(vertices_main["Collector"]["x"]) - bound
    y1 = min(vertices_main["Collector"]["y"]) - bound
    y2 = min(vertices_main["Set"]["y"]) - bound
    y3 = min(vertices_main["Egg"]["y"]) - bound
    return [x, x, max(boxx), max(boxx), x], [y1, y2, y3, y1, y1]


# For date box #


def v_get_date_box(
    vertices_main, midline_x, midline_y, bound=25, boundy=75, midline_method=True
):
    # Input: vertices of textboxes around main category titles; coordinates of "middle line".
    # Output: contour around date box.
    x1 = min(vertices_main["Date"]["x"]) - bound
    x2 = min(vertices_main["Set"]["x"]) - bound
    y1 = min(vertices_main["Date"]["y"]) - bound
    y4 = min(vertices_main["Set"]["y"]) - bound

    if midline_method:
        y2 = np.interp(x1, midline_x, midline_y)
        y3 = np.interp(x2, midline_x, midline_y)
    else:
        y2 = max(vertices_main["Date"]["y"]) + boundy
        y3 = max(vertices_main["Set"]["y"]) + boundy

    return [x1, x1, x2, x2, x1], [y1, y2, y3, y4, y1]


# For set mark box #


def v_get_setmark_box(
    vertices_main, midline_x, midline_y, bound=25, boundy=75, midline_method=True
):
    # Input: vertices of textboxes around main category titles; coordinates of "middle line".
    # Output: contour around set mark box.
    x1 = min(vertices_main["Set"]["x"]) - bound
    x2 = min(vertices_main["Egg"]["x"]) - bound

    y1 = min(vertices_main["Set"]["y"]) - bound
    if midline_method:
        y2 = np.interp(x1, midline_x, midline_y)
        y3 = np.interp(x2, midline_x, midline_y)
    else:
        y2 = max(vertices_main["Set"]["y"]) + boundy
        y3 = max(vertices_main["Egg"]["y"]) + boundy

    y4 = min(vertices_main["Egg"]["y"]) - bound

    return [x1, x1, x2, x2, x1], [y1, y2, y3, y4, y1]


# For no. eggs box #


def v_get_noeggs_box(
    vertices_main, boxx, midline_x, midline_y, bound=25, boundy=75, midline_method=True
):
    # Input: vertices of textboxes around main category titles; x coordinates of main box contour; coordinates of "middle line".
    # Output: contour around no. eggs box.
    x = min(vertices_main["Egg"]["x"]) - bound

    y1 = min(vertices_main["Egg"]["y"]) - bound

    if midline_method:
        y2 = np.interp(x, midline_x, midline_y)
        y3 = np.interp(max(boxx), midline_x, midline_y)
    else:
        y2 = max(vertices_main["Egg"]["y"]) + boundy
        y3 = deepcopy(y2)

    return [x, x, max(boxx), max(boxx), x], [y1, y2, y3, y1, y1]


# For other box #


def v_get_other_box(vertices_main, boxx, boxy, midline_y, bound=30):
    # Input: vertices of textboxes around main category titles; main box contour, y coordinates of "middle line".
    # Output: contour around "other text" box.
    x_ = min(vertices_main["Date"]["x"]) - bound
    x = [x_, x_, max(boxx), max(boxx), x_]
    y = [midline_y[0], max(boxy), max(boxy), midline_y[-1], midline_y[0]]
    return x, y


###########################
# Text Extraction Functions
###########################


def v_get_reg_number(
    inds_dict,
    all_words,
    final_textboxes,
    ignore_horizontal_in_reg_box=False,
    order_text=True,
):
    # Input: dictionary of category indexes per textbox; all texts.
    # Output: registration number.
    all_text = []
    all_inds = []

    for j in list(inds_dict.keys()):
        if inds_dict[j] == 0:
            txt = all_words[j]
            if len(re.findall("\d+", txt)) > 0:
                if ignore_horizontal_in_reg_box == True:
                    box = final_textboxes[j]
                    horizontal_or_vertical = detect_orientation(box[:, 0], box[:, 1])
                    if horizontal_or_vertical == "v":
                        all_text.append(txt)
                        all_inds.append(j)
                else:
                    all_text.append(txt)
                    all_inds.append(j)
    if order_text:
        try:
            max_y = [max(final_textboxes[k][:, 1]) for k in all_inds]
            text_order = np.argsort(max_y)[::-1]
            # Order all_text by y position of textbox.
            all_text = np.array(all_text)[text_order]
            # # Order all_text by length of string.
            # all_text = np.array(all_text)[np.argsort([len(a) for a in all_text])[::-1]]
        except:
            pass

    try:
        final_reg = ".".join(map(str, re.findall("\w+", " ".join(map(str, all_text)))))
    except:
        final_reg = deepcopy(all_text)

    return final_reg


def v_check_for_possible_reg_number(texts):
    # Input: text possibly containing registration number.
    # Output: binary classification of whether registration number was found.
    reg_ = False
    digits_ = False
    for w in texts:
        if "Reg" in w:
            reg_ = True
        if len(re.findall("\d+", w)) > 0:
            digits_ = True
    if digits_ and reg_:
        reg_ = True
    else:
        reg_ = False

    return reg_


def v_get_species_text(inds_dict, all_words, final_textboxes, order_reg=True):
    # Input: dictionary of category indexes per textbox; all texts.
    # Output: text contained in species box, and possible registration number.
    inds = [j for j in list(inds_dict.keys()) if inds_dict[j] == 1]

    texts = np.array(all_words)[inds]

    reg_ = v_check_for_possible_reg_number(texts)

    if reg_ is False:
        species = " ".join(map(str, re.findall("\w+", " ".join(map(str, texts)))))
        reg = "N/A"
    else:
        species_text = []
        reg_text = []
        reg_inds = []
        for ind, w in enumerate(texts):
            if len(re.findall("\d+", w)) == 0:
                if (fuzz.ratio("Reg.No", w) < 80) and (fuzz.ratio("Reg", w) < 80):
                    species_text.append(w)
            else:
                reg_text.append(w)
                reg_inds.append(inds[ind])

        if order_reg:
            max_x = [max(final_textboxes[k][:, 0]) for k in reg_inds]
            text_order = np.argsort(max_x)
            reg_text = np.array(reg_text)[text_order]

        species = " ".join(
            map(str, re.findall("\w+", " ".join(map(str, species_text))))
        )
        reg = ".".join(map(str, re.findall("\w+", " ".join(map(str, reg_text)))))

    return species, reg


def v_get_text_from_category_box(inds_dict, all_words, label_index):
    # Input: dictionary of category indexes per textbox; all texts, category index of interest.
    # Output: text from specific category box.
    inds = [j for j in list(inds_dict.keys()) if inds_dict[j] == label_index]

    texts = np.array(all_words)[inds]

    final_text = " ".join(map(str, texts))

    return final_text


#################
# Main Functions
#################


def v_get_boxes_ref(img_sk, vertices_main, text_annotations, vision_response):
    # Input: original image, vertices of main category title textboxes, all texts, Google Vision response.
    # Output: dictionary of contours around all main category boxes.
    boxes_ref_new = {}

    # Main box
    boxx, boxy = get_main_box_contour(vision_response)

    Ig = reformat_image(img_sk, vertices_main, text_annotations, boxx, boxy)
    # Line below main boxes
    x_, y_ = find_midline(Ig)

    # 1) Reg number box:
    X, Y = v_get_reg_box(vertices_main, boxx, boxy)
    boxes_ref_new["reg"] = [X, Y]

    # 2) Species box:
    X, Y = v_get_species_box(vertices_main, boxx, boxy)
    boxes_ref_new["species"] = [X, Y]

    # 3) Locality box
    X, Y = v_get_locality_box(vertices_main)
    boxes_ref_new["locality"] = [X, Y]

    # 4) Collector box:
    X, Y = v_get_collector_box(vertices_main, boxx)
    boxes_ref_new["collector"] = [X, Y]

    # 5) Date Box:
    X, Y = v_get_date_box(vertices_main, x_, y_)
    boxes_ref_new["date"] = [X, Y]

    # 6) Set Mark box:
    X, Y = v_get_setmark_box(vertices_main, x_, y_)
    boxes_ref_new["setMark"] = [X, Y]

    # 7) No. Eggs box:
    X, Y = v_get_noeggs_box(vertices_main, boxx, x_, y_)
    boxes_ref_new["noOfEggs"] = [X, Y]

    # 8) Other box:
    X, Y = v_get_other_box(vertices_main, boxx, boxy, y_)
    boxes_ref_new["other"] = [X, Y]

    return boxes_ref_new


def get_all_texts_and_box_index(
    vision_response, boxes_ref, textbox_leeway=30, leeway_backup=15
):
    # Input: Google Vision response, dictionary of contours of cateogory boxes.
    # Output: Index of all texts based on the category box they appear in; list of all texts.
    new_textboxes = []
    all_words = []

    for j, p in enumerate(vision_response["textAnnotations"]):
        if j > 0:
            X = []
            Y = []
            for i in p["boundingPoly"]["vertices"]:
                x = i["x"]
                y = i["y"]
                X.append(x)
                Y.append(y)
            new_textboxes.append(np.array([X, Y], dtype=np.float32).T)
            all_words.append(p["description"])
    try:
        new_inds_dict = get_box_index_for_textboxes(
            np.array(new_textboxes),
            list(boxes_ref.values()),
            textbox_leeway=textbox_leeway,
        )
    except:
        new_inds_dict = get_box_index_for_textboxes(
            np.array(new_textboxes),
            list(boxes_ref.values()),
            textbox_leeway=textbox_leeway + leeway_backup,
        )

    return new_inds_dict, all_words, new_textboxes


def v_get_all_category_text(
    inds_dict,
    all_words,
    final_textboxes,
    species_method="new",
    ignore_horizontal_in_reg_box=False,
):
    # Input: all words from Vision response and category index.
    # Output: responses per category.
    all_info = {}
    # 1) Registration number and Species:
    # we combine because sometimes these are in the same box.
    try:
        reg = v_get_reg_number(
            inds_dict,
            all_words,
            final_textboxes,
            ignore_horizontal_in_reg_box=ignore_horizontal_in_reg_box,
        )
    except:
        reg = "N/A"
    try:
        cardSpecies, reg_backup = v_get_species_text(
            inds_dict, all_words, final_textboxes
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
    reg = remove_word_from_string(reg, "reg no")
    reg = ".".join(re.findall("\d+", reg))

    all_info["registrationNumber"] = reg
    # Species
    cardSpecies = remove_all_categories_from_text(
        cardSpecies,
        keywords=["reg no", "locality", "collector", "set mark", "no of eggs"],
    )
    if species_method == "new":
        species_name, species_results = updated_find_species_results(cardSpecies)
    else:
        species_name, species_results = find_species_from_text([cardSpecies], [])
    taxon = get_taxon_info(species_name, species_results)

    taxon_blanks = count_blanks_from_taxon(taxon, list(taxon.keys())[:4])
    if (taxon_blanks >= 3) and (len(re.findall("\w+", cardSpecies)) > 3):
        (
            species_name_new,
            species_results_new,
            combined_name,
        ) = backup_species_extraction_by_combining_words(cardSpecies)
        taxon_new = get_taxon_info(species_name_new, species_results_new)
        taxon_new_blanks = count_blanks_from_taxon(
            taxon_new, list(taxon_new.keys())[:4]
        )
        if (taxon_new_blanks < taxon_blanks) and (len(species_name_new) > 0):
            taxon = deepcopy(taxon_new)
            species_name = deepcopy(combined_name)

    if (len(species_name) == 0) or (len(re.findall("\w+", species_name)) == 1):
        all_info["cardSpecies"] = " ".join(re.findall("\w+", cardSpecies))
    else:
        all_info["cardSpecies"] = species_name
    for b in taxon.keys():
        all_info[b] = taxon[b]

    # 2) Locality:
    try:
        locality = v_get_text_from_category_box(inds_dict, all_words, 2)
    except:
        locality = "N/A"
    locality = remove_all_categories_from_text(locality, main_word="locality")
    all_info["locality"] = locality

    # 3) Collector:
    try:
        collector = v_get_text_from_category_box(inds_dict, all_words, 3)
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
        date = v_get_text_from_category_box(inds_dict, all_words, 4)
    except:
        date = "N/A"
    date = remove_all_categories_from_text(date, main_word="date")
    date = remove_word_from_string(date, "date")
    all_info["date"] = date

    # 5) Set Mark:
    try:
        setMark = v_get_text_from_category_box(inds_dict, all_words, 5)
    except:
        setMark = "N/A"
    setMark = remove_all_categories_from_text(setMark, main_word="set mark")
    all_info["setMark"] = setMark

    # 6) No. Eggs:
    try:
        noOfEggs = v_get_text_from_category_box(inds_dict, all_words, 6)
    except:
        noOfEggs = "N/A"
    noOfEggs = remove_all_categories_from_text(noOfEggs, main_word="no of eggs")

    all_info["noOfEggs"] = noOfEggs

    # 7) Other:
    try:
        other = v_get_text_from_category_box(inds_dict, all_words, 7)
    except:
        other = "N/A"
    all_info["remainingText"] = other

    return all_info


##############
# One Function
##############


def v_get_all_card_info(
    path_to_json, path_to_image, pixel_bound=50, textbox_leeway=30, min_fuzzy_bound=90
):
    # Input: path to Google Vision json, path to image.
    # Output: image, and category responses.
    # 1) Load image:
    image = io.imread(path_to_image)
    # 2) Load Google Vision output json:
    vision_response = load_vision_json(path_to_json)
    text_annotations = vision_response.get("textAnnotations", [])
    # 3) Get textboxes from Vision output:
    vertices_all = get_boxes_of_main_categories(text_annotations)
    vertices_all_refined, found_missing = check_gvision_vertices(
        vertices_all, text_annotations, fuzzy_bound=min_fuzzy_bound
    )
    if found_missing is False:
        vertices_all_refined, _ = check_gvision_vertices(
            vertices_all_refined, text_annotations, fuzzy_bound=min_fuzzy_bound - 10
        )
    vertices_main = get_centre_categories(vertices_all_refined, pixel_bound=pixel_bound)
    # 4) Get contours around category boxes:
    boxes_ref = v_get_boxes_ref(image, vertices_main, text_annotations, vision_response)
    # 5) Group textboxes with category boxes:
    inds_dict, all_words, final_textboxes = get_all_texts_and_box_index(
        vision_response, boxes_ref, textbox_leeway=textbox_leeway
    )
    # 6) Sort out texts by category:
    all_info = v_get_all_category_text(inds_dict, all_words, final_textboxes)

    return image, all_info
