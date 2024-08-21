import spacy
from spacy.training import offsets_to_biluo_tags

# Initialize spaCy
nlp = spacy.blank("en")

# Your text and entities
text = "JJ - Y2K Pepper Tee S: $45 939530123 M: $45 939530124 L: $45 939530125 XL: $45 939530126 2XL: $45 939530127"
entities = [
             (0, 19, "NAME"),
             (20, 21, "SIZE"),
             (24, 26, "PRICE"),
             (27, 36, "ITEM_ID"),
             (37, 38, "SIZE"),
             (41, 43, "PRICE"),
             (44, 53, "ITEM_ID"),
             (54, 55, "SIZE"),
             (58, 60, "PRICE"),
             (61, 70, "ITEM_ID"),
             (71, 73, "SIZE"),
             (76, 78, "PRICE"),
             (79, 88, "ITEM_ID"),
             (89, 92, "SIZE"),
             (95, 97, "PRICE"),
             (98, 107, "ITEM_ID"),
         ]
# Convert to BILUO tags
doc = nlp.make_doc(text)
tags = offsets_to_biluo_tags(doc, entities)

# Print the tokens and their corresponding tags
for token, tag in zip(doc, tags):
    print(f"{token.text:<5} {tag}")
