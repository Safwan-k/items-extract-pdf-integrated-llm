import spacy

loaded_nlp = spacy.load("items_custom_model/items-pdf-model")
# loaded_nlp = spacy.load("items-pdf-model")


def extract_info(text):
    doc = loaded_nlp(text)
    extracted_info = {
        "NAME": [],
        "SIZE": [],
        "PRICE": [],
        "ITEM_ID": []
    }

    for ent in doc.ents:
        if ent.label_ in extracted_info:
            extracted_info[ent.label_].append(ent.text)

    return extracted_info


def process_multiple_texts(texts):
    results = []
    for text in texts:
        result = extract_info(text)
        if any(result.values()):  # Check if any value in the dictionary is non-empty
            results.append(result)
    return results


# __texts = [
#     'JJ - Vintage Black White Photo Tee S: $45 939530113 M: $45 939530114 L: $45 939530115 XL: $45 939530116 2XL: $45 939530117',
#     'JJ - Y2K Pepper Tee S: $45 939530123 M: $45 939530124 L: $45 939530125 XL: $45 939530126 2XL: $45 939530127', 'JJ - Photo Dateback Tee S: $45 939530103 M: $45 939530104 L: $45 939530105 XL: $45 939530106 2XL: $45 939530107 3XL: $45 939530108']
# __result = process_multiple_texts(texts=__texts)
# print(__result)
