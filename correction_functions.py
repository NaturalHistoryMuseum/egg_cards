#########
# Imports
#########


from pygbif import species
import numpy as np
import re
import pandas as pd
from copy import deepcopy
from fuzzywuzzy import fuzz
from ast import literal_eval
from dateparser.search import search_dates
from post_processing_functions import find_species_from_text, get_taxon_info


########
# Terms
########

# Key terms excluding taxonomic terms.
key_terms = ["reg no", "locality", "collector", "date", "set mark", "no of eggs"]
key_terms_extended = [
    "reg no",
    "reg. no",
    "locality",
    "collector",
    "date",
    "set mark",
    "no of eggs" "no. of eggs",
]
key_terms_p_ordered = [
    "reg no",
    "locality",
    "collector",
    "date",
    "no of eggs",
    "set mark",
]
taxon_terms = ["order", "family", "genus", "species", "canonicalName"]

months = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

months_shortened = [
    "jan",
    "feb",
    "march",
    "apr",
    "may",
    "june",
    "july",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


###################
# GENERAL FUNCTIONS
###################


def is_nan_(w):
    try:
        return np.isnan(w)
    except:
        return False


def find_weird_results(list_to_check, min_length_boundary, max_length_boundary):
    weird = []
    weird_inds = []

    for i, l in enumerate(list_to_check):
        if (
            (is_nan_(l) == True)
            or (len(l) < min_length_boundary)
            or (len(l) > max_length_boundary)
            or (l == "N/A")
        ):
            weird.append(l)
            weird_inds.append(i)
    return weird, weird_inds


def check_for_word_in_text(original_text, words, use_text=True):
    if use_text:
        text = re.findall("[a-zA-Z]+", original_text)
        text = " ".join(map(str, original_text))
    else:
        text = deepcopy(original_text)
    in_word = False
    for word in words:
        if word in text.lower():
            in_word = True
    return in_word, text, original_text


def find_words_(texts_to_check, keywords, word_limit_min=0, word_limit_max=200):
    new_text = ""
    for text in texts_to_check:
        try:
            in_word, _, _ = check_for_word_in_text(text, keywords)
        except:
            in_word = False
        if (
            (in_word is True)
            and (len(text) > word_limit_min)
            and (len(text) < word_limit_max)
        ):
            new_text = deepcopy(text)
            break
    return in_word, new_text


#################
# DATE FUNCTIONS
#################


def is_date_present(text_segment, check_weekdays=True):
    date_present = False
    try:
        text = str(text_segment).lower()
        contains_month = False
        contains_day = False

        for m in months:
            if m in text:
                contains_month = True

        for m in months_shortened:
            if m in text:
                contains_month = True

        if check_weekdays:
            for m in days:
                if m in text:
                    contains_day = True

            if (contains_month is True) or (contains_day is True):
                date_present = True
        else:
            if contains_month is True:
                date_present = True

    except:
        date_present = False
    return date_present


def is_date_weird(dates):
    dates_weird = []
    weird_ids = []
    for i, d in enumerate(dates):
        try:
            contains_date = False
            numbers = re.findall("\d+", d)
            if is_date_present(d) or (len(numbers) > 0):
                contains_date = True
            if contains_date is True:
                pass
            else:
                dates_weird.append(d)
                weird_ids.append(i)
        except:
            dates_weird.append(d)
            weird_ids.append(i)
    return dates_weird, weird_ids


def update_dates(weird_ids, set_mark, noeggs, dates):

    set_mark_copy = list(set_mark)
    noeggs_copy = list(noeggs)
    dates_copy = list(dates)
    inds_to_update = []
    k = 0
    v = 0

    for ind in weird_ids:
        sm = set_mark[ind]
        eg = noeggs[ind]
        new_date = ""
        found_date = False

        if is_date_present(sm):
            new_date = deepcopy(sm)
            found_date = True
        if is_date_present(eg) and (len(str(eg)) > len(str(sm))):
            new_date = deepcopy(eg)
            found_date = True

        if found_date is True:
            dates_copy[ind] = new_date
            if new_date == sm:
                set_mark_copy[ind] = "N/A"
            if new_date == eg:
                noeggs_copy[ind] = "N/A"
            k = k + 1

        if found_date is False:
            inds_to_update.append(ind)

    return set_mark_copy, noeggs_copy, dates_copy, inds_to_update


def does_text_contain_date(txt):
    contains = False
    try:
        b = search_dates(txt)
        b = b[0][0]
        if len(b) > 4:
            contains = True
    except:
        pass
    return contains


def find_date_backup(text_segments):
    new_date = ""
    for txt in text_segments:
        try:
            b = search_dates(txt)
            b = b[0][0]
            if len(b) > len(new_date):
                new_date = deepcopy(b)
        except:
            pass
    return new_date


def double_check_dates(new_dates, dates, remtext, inds_to_update, weird_ids):
    remtext_copy = deepcopy(remtext)
    newdates_copy = deepcopy(new_dates)
    for ind in weird_ids:
        if ind in inds_to_update:
            rem_text = remtext[ind]
            found_date = False
            new_date = ""
            chosen_index = 0
            for i, txt in enumerate(rem_text):
                r = re.findall("\d+", txt)
                if len(r) > 0:
                    if is_date_present(txt, check_weekdays=False):
                        if (len(txt) > len(new_date)) and (len(txt) < 50):
                            found_date = True
                            new_date = deepcopy(txt)
                            chosen_index = deepcopy(i)
            if found_date:
                newdates_copy[ind] = new_date
                del rem_text[chosen_index]
                if is_nan_(dates[ind]) is False:
                    rem_text.append(dates[ind])
                remtext_copy[ind] = rem_text
            else:
                newdate = find_date_backup(rem_text)
                if newdate == "":
                    newdates_copy[ind] = ""
                    if is_nan_(dates[ind]) is False:
                        rem_text.append(dates[ind])
                    remtext_copy[ind] = rem_text
                else:
                    newdates_copy[ind] = newdate
                    if is_nan_(dates[ind]) is False:
                        rem_text.append(dates[ind])
                    remtext_copy[ind] = rem_text
        else:
            rem_text = remtext[ind]
            if is_nan_(dates[ind]) is False:
                rem_text.append(dates[ind])
            remtext_copy[ind] = rem_text

    return remtext_copy, newdates_copy


###################
# REG NO FUNCTIONS
###################


def backup_reg(weird_ids_reg, remtext, reg_no):

    reg_no_copy = deepcopy(reg_no)

    for ind in weird_ids_reg:
        text = remtext[ind]
        for txt in text:
            try:
                r = re.findall("\d+[.,]\d+[.,]\d+", txt)
                if len(r) > 0:
                    if len(txt) < 15:
                        final = deepcopy(txt)
                    else:

                        try:
                            r2 = re.findall("\d+\.\d+\.\d+\.\d+", txt)
                            if len(r2) > 0:
                                r = deepcopy(r2)
                        except:
                            pass
                        final = deepcopy(r)
                    reg_no_copy[ind] = final
            except:
                pass

    return reg_no_copy


###################
# SPECIES FUNCTIONS
###################


def find_weird_species(species, results):
    weird_sp = []
    weird_sp_inds = []

    for i, sp in enumerate(species):
        try:
            r = re.findall("\d", sp)
            if len(r) > 3:
                weird_sp.append(sp)
                weird_sp_inds.append(i)
            else:
                max_ = 0
                in_term = False
                for c in taxon_terms:
                    try:
                        keyword = results[c].iloc[i].lower()
                        if (keyword in sp.lower()) or (sp.lower() in keyword):
                            in_term = True
                    except:
                        pass

                    try:
                        keyword = results[c].iloc[i].lower()
                        ratio = fuzz.ratio(keyword, sp.lower())
                        if ratio > max_:
                            max_ = deepcopy(ratio)
                    except:
                        pass

                if in_term:
                    pass
                else:
                    if max_ < 70:
                        weird_sp.append(sp)
                        weird_sp_inds.append(i)
        except:
            weird_sp.append(sp)
            weird_sp_inds.append(i)

    return weird_sp, weird_sp_inds


def redo_weird_species(remtext, weird_sp_inds, weird_sp, results):
    results_copy = deepcopy(results)
    for i, w in enumerate(weird_sp):
        try:
            species_name, species_results = find_species_from_text(
                remtext[weird_sp_inds[i]], [], backup=True
            )
            taxon = get_taxon_info(species_name, species_results)
            for t in list(taxon.keys()):
                results_copy[t].iloc[weird_sp_inds[i]] = taxon[t]
            results_copy["cardSpecies"].iloc[weird_sp_inds[i]] = species_name
            # Remove from rem text
            k = ""
            for u, p in enumerate(remtext[weird_sp_inds[i]]):
                if p == species_name:
                    k = deepcopy(u)
            txt_ = deepcopy(remtext[weird_sp_inds[i]])
            del txt_[k]
            txt_.append(w)
            results_copy["remainingText"].iloc[weird_sp_inds[i]] = txt_

        except:
            pass

    return results_copy


###################
# NO EGGS FUNCTIONS
###################


def check_no_eggs(noeggs, setmark, localities, remText):

    # remText = list(results["remainingText"])
    # localities = list(results["locality"])
    # setmark = list(results["setMark"])
    # noeggs = list(results["noOfEggs"])

    _, weird_noeggs_inds = find_weird_results(noeggs, 0, 150)

    possible_noeggs = deepcopy(noeggs)

    # No of Eggs:
    for ind in weird_noeggs_inds:
        texts_to_check = [setmark[ind], localities[ind]]
        texts_to_check.extend(remText[ind])
        _, new_text = find_words_(
            texts_to_check,
            ["no of eggs", "of eggs"],
            word_limit_min=0,
            word_limit_max=50,
        )
        possible_noeggs[ind] = new_text

    return possible_noeggs, weird_noeggs_inds


####################
# SET MARK FUNCTIONS
####################


def check_setmark(noeggs, setmark, localities, remText):

    _, weird_setmark_inds = find_weird_results(setmark, 0, 100)

    possible_setmark = deepcopy(setmark)

    # Set Mark:
    for ind in weird_setmark_inds:
        texts_to_check = [noeggs[ind], localities[ind]]
        texts_to_check.extend(remText[ind])
        _, new_text = find_words_(
            texts_to_check, ["set mark", "setmark"], word_limit_min=0, word_limit_max=50
        )
        possible_setmark[ind] = new_text

    return possible_setmark, weird_setmark_inds


#####################
# COLLECTOR FUNCTIONS
#####################


def check_collector(collections, localities, remText):

    possible_collections = deepcopy(collections)

    _, weird_collection_inds = find_weird_results(collections, 3, 200)

    for ind in weird_collection_inds:
        texts = remText[ind]
        collector_ = ""
        for text in texts:
            if ("collection" in text.lower()) or ("collector" in text.lower()):
                if (len(text) > len(collector_)) and (len(text) < 100):
                    collector_ = deepcopy(text)

        if collector_ == "":
            text = localities[ind]
            if is_nan_(text) is False:
                if ("collection" in text.lower()) or ("collector" in text.lower()):
                    collector_ = deepcopy(text)

        possible_collections[ind] = collector_

    return possible_collections


#####################
# LOCALITY FUNCTIONS
#####################

"""
Note that for the locality functions we are using spacy.
If this can't be installed locally, then use colab.
"""

try:
    import spacy
    import spacy.cli

    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")
except:
    pass


def get_locations_from_text(text, entity_labels=["GPE", "LOC", "FAC"]):
    locations = []

    text = text.replace("-", "")

    doc = nlp(text)
    for entity in doc.ents:
        if (
            (entity.label_ in entity_labels)
            and (entity not in locations)
            and (len(entity.text) > 2)
        ):
            txt = entity.text
            numbers = re.findall("\d", txt)
            letters = re.findall("[a-zA-Z]", txt)
            if (len(numbers) < 3) and (len(letters) > 2):
                locations.append(entity)
    if len(locations) > 0:
        loc_ = ", ".join(map(str, locations))
    else:
        loc_ = "N/A"

    return loc_


def check_localities(locality, remText):
    weird_locality, weird_locality_inds = find_weird_results(locality, 3, 150)
    possible_localities = deepcopy(locality)

    for ind in weird_locality_inds:
        location = ""

        # Check for locality data in the super long original text:
        original = weird_locality[ind]
        try:
            orig_len = len(original)
        except:
            orig_len = 0

        if orig_len > 150:
            loc = get_locations_from_text(
                text, entity_labels=["GPE", "LOC", "FAC", "ORG"]
            )
            if loc != "N/A":
                location = deepcopy(loc)

        # Check through remaining texts:
        texts = remText[ind]
        for text in texts:
            loc = get_locations_from_text(text, entity_labels=["GPE", "LOC"])
            if (loc != "N/A") and (len(loc) > len(location)):
                location = deepcopy(loc)

        # Check again but with added entity types:
        if location == "":
            for text in texts:
                loc = get_locations_from_text(text)
                if (loc != "N/A") and (len(loc) > len(location)):
                    location = deepcopy(loc)

        possible_localities[ind] = location

    return possible_localities


#####################
# ALL-IN-ONE FUNCTION
#####################


def make_corrections_to_data(path_to_data, correct_all=True):
    # 1) Load CSV containing egg card info per id:
    results = pd.read_csv(path_to_data)

    # 2) Define columns:
    dates = results["date"]
    set_mark = results["setMark"]
    noeggs = results["noOfEggs"]
    remtext = results["remainingText"].apply(literal_eval)
    reg_no = results["registrationNumber"]
    species = results["cardSpecies"]

    # 3) Correct dates:
    _, weird_ids = is_date_weird(dates)
    set_mark_copy, noeggs_copy, dates_copy, inds_to_update = update_dates(
        weird_ids, set_mark, noeggs, dates
    )
    remtext_copy, newdates_copy = double_check_dates(
        dates_copy, dates, remtext, inds_to_update, weird_ids
    )

    # 4) Find missing registration numbers:
    _, weird_ids_reg = find_weird_results(reg_no, 3, 100)
    reg_no_copy = backup_reg(weird_ids_reg, remtext_copy, reg_no)

    # 5) Put results so far into new dataframe:
    results_v2 = deepcopy(results)
    results_v2["setMark"] = set_mark_copy
    results_v2["date"] = newdates_copy
    results_v2["noOfEggs"] = noeggs_copy
    results_v2["remainingText"] = remtext_copy
    results_v2["registrationNumber"] = reg_no_copy

    # 6) Correct species details:
    weird_sp, weird_sp_inds = find_weird_species(species, results_v2)
    results_v3 = redo_weird_species(remtext_copy, weird_sp_inds, weird_sp, results_v2)

    # 7) Define variables from latest dataframe:
    remText = results_v3["remainingText"]
    set_mark = results_v3["setMark"]
    noeggs = results_v3["noOfEggs"]
    locality = results_v3["locality"]
    collector = results_v3["collector"]

    # 7) Verify number of eggs:
    new_noeggs, _ = check_no_eggs(noeggs, set_mark, locality, remText)

    # 8) Verify set mark:
    new_setmark, _ = check_setmark(noeggs, set_mark, locality, remText)

    # 9) Verify collector and find collection:
    new_collections = check_collector(collector, locality, remText)

    # 10) Correct locality:
    if correct_all:
        new_localities = check_localities(locality, remText)
    else:
        new_localities = deepcopy(locality)

    # 11) Update dataframe:
    results_final = deepcopy(results_v3)
    results_final["setMark"] = new_setmark
    results_final["noOfEggs"] = new_noeggs
    results_final["collector"] = new_collections
    results_final["locality"] = new_localities

    return results_final
