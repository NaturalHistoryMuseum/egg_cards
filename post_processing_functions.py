"""

EGG CARDS
----------

Main functions that use the OCR results to extract textual information.
We extract information regarding the following data:

1. Taxonomic information. - use for matching
2. Registration number. - use for matching
3. Locality.
4. Collector. - use for matching
5. Collection date. - use for matching
6. Set mark. - use for matching
7. Number of eggs - use for mathcing
8. Other text

Taxonomic information includes the taxonomic terms that appear on the index card
as well as additional information from GBIF API (if found) regarding the:

1. Order
2. Family
3. Genus
4. Species
5. Canonical Name

"""

##########
# Imports
##########

from pygbif import species
import numpy as np
import re
from copy import deepcopy
from fuzzywuzzy import fuzz


########
# Terms
########
# Key terms excluding taxonomic terms.
key_terms = ["reg no", "locality", "collector", "date", "set mark", "no of eggs"]
taxon_terms = ["order", "family", "genus", "species", "canonicalName"]

###################
# GENERAL FUNCTIONS
###################


def process_ocr_text(text):
    # Input: OCR results.
    # Output: Separated text.
    all_text_segments = []
    for t in text:
        typ = type(t[0])
        if typ == list:
            for p in t[0]:
                all_text_segments.append(p)
        else:
            all_text_segments.append(t[0])
    return all_text_segments


def get_text_index_for_keyterms(all_text_segments):
    # Input: Processed text from OCR.
    # Output: Dictionary of key terms per text index; dictionary of text index per keyterm.
    text_terms = {}
    text_keys = {}
    for u, t in enumerate(all_text_segments):
        txt = " ".join(map(str, re.findall("\w+", t))).lower()
        results = []
        for kt in key_terms:
            if kt in txt:
                results.append(kt)
                text_keys[kt] = u
        if len(results) == 0:
            results = "None"
        text_terms[u] = results
    return text_keys, text_terms


def get_term_related_text(all_text_segments, text_terms, term):
    # Input: all processed text segments; dictionary of key terms per text index.
    # Output: locality-related text segments.
    n = len(text_terms)
    ocr_terms = []
    for i in range(n):
        if type(text_terms[i]) != str:
            for t in text_terms[i]:
                if t == term:
                    ocr_terms.append(all_text_segments[i])
    return ocr_terms


def get_specific_terms(related_text, term, fuzzy_threshold):
    # Input: list of related text; interested term (e.g. "locality"), threshold for fuzzy score related to key term.
    # Output: Key term related final text.
    words = []
    for w in related_text:
        words.extend(re.findall("\w+", w))

    filtered_words = []
    for w in words:
        l = fuzz.ratio(term, w.lower())
        if (l < fuzzy_threshold) and (w not in filtered_words):
            filtered_words.append(w)

    final_text = " ".join(map(str, filtered_words))

    return final_text


# 1 -- TAXONOMIC FUNCTIONS
###########################


def get_index_of_nontaxon_text(text_terms):
    # Input: Dictionary of key terms per text index.
    # Output: List of text index for non-taxonomic text segments.
    n = len(text_terms)
    index_list = []
    for i in range(n):
        if text_terms[i] != "None":
            index_list.append(i)
    return index_list


def find_species_results(all_text_segments, non_taxon_list):
    # Input: all text segments from card (OCR output); text index for non-taxonomic text.
    # Output: species name, index in text segments, GBIF API results.
    species_name = ""
    species_index = "N/A"
    species_results = ""

    for u, t in enumerate(all_text_segments):
        text_ = re.findall("[a-zA-Z]+", t)
        text_ = " ".join(map(str, text_))
        if (u not in non_taxon_list) and (len(text_) > 3):
            r = species.name_lookup(text_)
            if len(r["results"]) > 0:
                in_name = False
                for j in taxon_terms:
                    try:
                        res = r["results"][0][j]
                        if res.lower() in t.lower():
                            in_name = True
                    except:
                        pass
                if in_name is True:
                    species_name = deepcopy(t)
                    species_index = deepcopy(u)
                    species_results = r["results"]
            else:
                t_ = re.findall("[a-zA-Z]+", t)
                if len(t_) == 3:
                    sp = " ".join(map(str, t_[:2]))
                    r = species.name_lookup(sp)
                    if len(r["results"]) > 0:
                        species_name = deepcopy(t)
                        species_index = deepcopy(u)
                        species_results = r["results"]

    return species_name, species_index, species_results


def find_species_results_backup(all_text_segments):
    # Input: all text segments.
    # Output: list of possible specie names, list of GBIF API results.
    name = []
    results = []
    for u, t in enumerate(all_text_segments):
        text_ = re.findall("[a-zA-Z]+", t)
        for txt in text_:
            r = species.name_lookup(txt)
            if len(r["results"]) > 0:
                try:
                    cl = r["results"][0]["class"]
                    if cl == "Aves":
                        name.append(txt)
                        results.append(r["results"])
                except:
                    pass
    return name, results


def filter_species_results(names, results):
    # Function to be used to filter results from the backup taxon extraction function.
    # Input: results from find_species_results_backup.
    # Output: filtered list of possible species names, and GBIF API results.
    filtered_words = []
    filtered_results = []
    for ind, i in enumerate(names):
        for j in taxon_terms:
            try:
                res = results[ind][0][j]
                if i.lower() in res.lower():
                    if i not in filtered_words:
                        filtered_words.append(i)
                        filtered_results.append(results[ind])
            except:
                pass

    return filtered_words, filtered_results


def find_species_from_text(all_text_segments, non_taxon_list):
    # Main function to extract taxonomic information / GBIF API from text segments.
    # Input: all text segments (OCR results), text index for non-taxonomic text.
    # Output: species name (from card), GBIF API results.
    species_name, _, species_results = find_species_results(
        all_text_segments, non_taxon_list
    )
    if species_name == "":
        # If empty, use backup method:
        names, results = find_species_results_backup(all_text_segments)
        # Filter out potential incorrect terms:
        filtered_words, filtered_results = filter_species_results(names, results)
        if len(filtered_words) > 0:
            if len(filtered_words) == 1:
                return filtered_words[0], filtered_results[0]
            else:
                return filtered_words, filtered_results
    else:
        return species_name, species_results


def get_taxon_info(species, results):
    # Input: results from find_species_from_text
    # Output: Taxonomic information from GBIF API.
    taxon = {}

    if type(species) != str:
        # If there are multiple results, pick the first one.
        results = results[0]

    for term in taxon_terms:
        try:
            t = results[0][term]
        except:
            t = ""
        taxon[term] = t

    return taxon


# 2 -- REG NUMBER FUNCTIONS
############################


def find_reg_no_from_text(all_text_segments, text_keys):
    # Input: all processed text segments, dictionary of text index per key term.
    # Output: registration number.
    try:
        index = text_keys["reg no"]
        r = all_text_segments[index]
        reg_no = ".".join(map(str, re.findall("\d+", r)))
    except:
        reg_no = "N/A"
    return reg_no


# 3 -- LOCALITY FUNCTIONS
##########################


def get_locality(all_text_segments, text_terms):
    # Input: all processed text segments, dictionary of key terms per text segment.
    # Output: locality.
    locality_text = get_term_related_text(all_text_segments, text_terms, "locality")
    try:
        locality = get_specific_terms(locality_text, "locality", 85)
    except:
        locality = "N/A"
    return locality


# 3 -- COLLECTOR FUNCTIONS
###########################


def get_collector(all_text_segments, text_terms):
    # Input: all processed text segments, dictionary of key terms per text segment.
    # Output: locality.
    collector_text = get_term_related_text(all_text_segments, text_terms, "collector")
    try:
        collector = get_specific_terms(collector_text, "collector", 91)
    except:
        collector = "N/A"
    return collector


# 5 -- COLLECTION DATE
########################


def get_date(all_text_segments, text_terms):
    # Input: all processed text segments, dictionary of key terms per text segment.
    # Output: collection date.
    try:
        related_text = get_term_related_text(all_text_segments, text_terms, "date")
        k = np.argmax([len(re.findall("\d+", t)) for t in related_text])
        related_text = related_text[k]
        dates = re.findall("\w+", related_text)
        dates_ = []
        for d in dates:
            l = fuzz.ratio("date", d.lower())
            if (l < 75) and (d not in dates_):
                dates_.append(d)
        date = " ".join(map(str, dates_))
    except:
        date = "N/A"
    return date


# 6 -- SET MARK FUNCTIONS
##########################


def get_setmark(all_text_segments, text_terms):
    # Input: all processed text segments, dictionary of key terms per text segment.
    # Output: set mark.
    try:
        related_text = get_term_related_text(all_text_segments, text_terms, "set mark")
        terms_filtered = []
        for txt in related_text:
            terms = re.findall("\w+", txt)
            for w in terms:
                r1 = fuzz.ratio("set", w.lower())
                r2 = fuzz.ratio("mark", w.lower())
                if (r1 < 60) and (r2 < 70) and (w not in terms_filtered):
                    terms_filtered.append(w)
        setmark = " ".join(map(str, terms_filtered))
    except:
        setmark = "N/A"
    return setmark


# 7 -- NO. EGGS FUNCTION
#########################


def get_no_of_eggs(all_text_segments, text_terms):
    try:
        # Input: all processed text segments, dictionary of key terms per text segment.
        # Output: number of eggs.
        related_text = get_term_related_text(
            all_text_segments, text_terms, "no of eggs"
        )
        related_text = related_text[
            0
        ]  # just get the first isntance for now if there's more than one.
        words = re.findall("\S+", related_text)
        filtered_words = []
        for w in words:
            if (fuzz.ratio("no of eggs", w) < 15) and (w not in filtered_words):
                filtered_words.append(w)
        noeggs = " ".join(map(str, filtered_words))
    except:
        noeggs = "N/A"
    return noeggs


# 8 -- REMAINING TEXT
######################


def get_remaining_text(all_text_segments, text_terms, species_name):
    # Input: all processed text segments, dictionary of key terms per text segment; species text.
    # Output: number of eggs.
    remaining_text = []
    try:
        for i, txt in enumerate(all_text_segments):
            if (type(text_terms[i]) == str) and (txt != species_name):
                words = re.findall("\S+", txt)
                remaining_text.append(" ".join(map(str, words)))
    except:
        pass
    return remaining_text


#######################
# ALL-IN-ONE-FUNCTION #
########################


def get_all_egg_card_results(folder_path, filename):
    # Load ocr results
    path = folder_path + "/" + filename
    id_ = filename[:-4]
    text = np.load(path, allow_pickle=True)
    # Process results
    all_text_segments = process_ocr_text(text)
    # Get indexes related to key terms:
    text_keys, text_terms = get_text_index_for_keyterms(all_text_segments)

    # Extract data:
    all_results = {}
    all_results["id"] = id_
    # 1. Taxonomic information
    non_taxon_list = get_index_of_nontaxon_text(text_terms)
    species_name, species_results = find_species_from_text(
        all_text_segments, non_taxon_list
    )
    taxon = get_taxon_info(species_name, species_results)
    species_name_ = " ".join(map(str, re.findall("\S+", species_name)))
    # Add taxon info to dictionary
    all_results["cardSpecies"] = species_name_
    for b in taxon.keys():
        all_results[b] = taxon[b]

    # 2. Registration number
    reg_no = find_reg_no_from_text(all_text_segments, text_keys)
    all_results["registrationNumber"] = reg_no

    # 3. Locality.
    locality = get_locality(all_text_segments, text_terms)
    all_results["locality"] = locality

    # 4. Collector
    collector = get_collector(all_text_segments, text_terms)
    all_results["collector"] = collector

    # 5. Collection date
    date = get_date(all_text_segments, text_terms)
    all_results["date"] = date

    # 6. Set mark
    set_mark = get_setmark(all_text_segments, text_terms)
    all_results["setMark"] = set_mark

    # 7. Number of eggs
    no_eggs = get_no_of_eggs(all_text_segments, text_terms)
    all_results["noOfEggs"] = no_eggs

    # 8. Other text
    remaining_text = get_remaining_text(all_text_segments, text_terms, species_name)
    all_results["remainingText"] = remaining_text

    return all_results
