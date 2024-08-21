import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random

TRAIN_DATA = [
    ("Button Pack $15", {"entities": [(0, 11, "NAME"), (13, 15, "PRICE")]}),
    ("JANET- TOUR BOOK 2024 $0 939530109", {"entities": [(0, 21, "NAME"), (23, 24, "PRICE"), (25, 34, "ITEM_ID")]}),
    ("ME SUMMER/FALL 2023 COUR TAM S: $45", {"entities": [(0, 28, "NAME"), (30, 30, "SIZE"), (33, 35, "PRICE")]}),
    ("ME SUMMER/FALL 2024 TOUR TEE S: $45 M: $45 L: $45 XL: $45 2XL: $45 3XL: $45",
     {"entities": [
         (0, 28, "NAME"),
         (29, 30, "SIZE"),
         (33, 35, "PRICE"),
         (36, 37, "SIZE"),
         (40, 42, "PRICE"),
         (43, 44, "SIZE"),
         (47, 49, "PRICE"),
         (50, 52, "SIZE"),
         (55, 57, "PRICE"),
         (58, 61, "SIZE"),
         (64, 66, "PRICE"),
         (67, 70, "SIZE"),
         (73, 75, "PRICE")
     ]}),
    (
    "JJ - Photo Dateback Tee S: $45 939530103 M: $45 939530104 L: $45 939530105 XL: $45 939530106 2XL: $45 939530107 3XL: $45 939530108",
    {"entities": [
        (0, 23, "NAME"),
        (24, 25, "SIZE"),
        (28, 30, "PRICE"),
        (31, 40, "ITEM_ID"),
        (41, 42, "SIZE"),
        (45, 47, "PRICE"),
        (48, 57, "ITEM_ID"),
        (58, 59, "SIZE"),
        (62, 64, "PRICE"),
        (65, 74, "ITEM_ID"),
        (75, 77, "SIZE"),
        (80, 82, "PRICE"),
        (83, 92, "ITEM_ID"),
        (93, 96, "SIZE"),
        (99, 101, "PRICE"),
        (102, 111, "ITEM_ID"),
        (112, 115, "SIZE"),
        (118, 120, "PRICE"),
        (121, 130, "ITEM_ID"),
    ]}),
    (
        "ME BROKEN TOUR RAGLAN S: $50 MET24TRG001SM M: $50 MET24TRG001MD L: $50 MET24TRG001LG XL: $50 MET24TRG001XL 2XL: $50 MET24TRG0012XL",
        {"entities": [
            (0, 21, "NAME"),
            (22, 23, "SIZE"), (26, 28, "PRICE"), (29, 42, "ITEM_ID"),
            (43, 44, "SIZE"), (47, 49, "PRICE"), (50, 63, "ITEM_ID"),
            (64, 65, "SIZE"), (68, 70, "PRICE"), (71, 84, "ITEM_ID"),
            (85, 87, "SIZE"), (90, 92, "PRICE"), (93, 106, "ITEM_ID"),
            (107, 110, "SIZE"), (113, 115, "PRICE"), (116, 130, "ITEM_ID")
        ]}),
    ("ME TIE DYE PHOTO TEE S: $45 M: $45 L: $45 XL: $45 2XL: $45",
     {"entities": [
         (0, 20, "NAME"), (21, 22, "SIZE"),
         (25, 27, "PRICE"),
         (28, 29, "SIZE"), (32, 34, "PRICE"),
         (35, 36, "SIZE"),
         (39, 41, "PRICE"), (42, 44, "SIZE"),
         (47, 49, "PRICE"),
         (50, 53, "SIZE"), (56, 58, "PRICE")]}),
    ("ME I’M NOT BROKEN TRUCKER One-Size: $35",
     {"entities": [(0, 25, "NAME"), (26, 34, "SIZE"), (37, 39, "PRICE")]}),
    ("ME I'm Not Broken Live from Topeka Correctional Facility LP $30",
     {"entities": [(0, 59, "NAME"), (61, 63, "PRICE")]}),
    (
        "ME I'M NOT BROKEN PHOTO TEE S: $45 MET24TTS003SM M: $45 MET24TTS003MD L: $45 MET24TTS003LG XL: $45 MET24TTS003XL 2XL: $45 MET24TTS0032XL",
        {"entities": [
            (0, 27, "NAME"),
            (28, 29, "SIZE"), (32, 34, "PRICE"), (35, 48, "ITEM_ID"),
            (49, 50, "SIZE"), (53, 55, "PRICE"), (56, 69, "ITEM_ID"),
            (70, 71, "SIZE"), (74, 76, "PRICE"), (77, 90, "ITEM_ID"),
            (91, 93, "SIZE"), (96, 98, "PRICE"), (99, 112, "ITEM_ID"),
            (113, 116, "SIZE"), (119, 121, "PRICE"), (122, 136, "ITEM_ID")
        ]}),
    ("ME Tour 2024 Bandana $20 MET24TAC007S",
     {"entities":
          [(0, 20, "NAME"), (22, 24, "PRICE"), (25, 37, "ITEM_ID")]}),
    ("ME Guitar Koozie $5 MET24TAC009",
     {"entities":
          [(0, 16, "NAME"), (18, 19, "PRICE"), (20, 31, "ITEM_ID")]}),
    (
        "SIG129074 - ATHLETIC HEATHER T TAKK S: $45 SIG129074-S M: $45 SIG129074-M L: $45 SIG129074-L XL: $45 SIG129074-XL 2XL: $45 SIG129074-2XL",
        {"entities":
            [
                (0, 35, "NAME"),
                (36, 37, "SIZE"),
                (40, 42, "PRICE"),
                (43, 54, "ITEM_ID"),

                (55, 56, "SIZE"),
                (59, 61, "PRICE"),
                (62, 73, "ITEM_ID"),

                (74, 75, "SIZE"),
                (78, 80, "PRICE"),
                (81, 92, "ITEM_ID"),

                (93, 95, "SIZE"),
                (98, 100, "PRICE"),
                (101, 113, "ITEM_ID"),

                (114, 117, "SIZE"),
                (120, 122, "PRICE"),
                (123, 136, "ITEM_ID")
            ]}),
    (
        "SIG129071 - WHITE T BW NEGATIVE ATTA SPECTRUM S: $45 SIG129071-S M: $45 SIG129071-M L: $45 SIG129071-L XL: $45 SIG129071-XL 2XL: $45 SIG129071-2XL",
        {"entities":
            [
                (0, 45, "NAME"),
                (46, 47, "SIZE"),
                (50, 52, "PRICE"),
                (53, 64, "ITEM_ID"),
                (65, 66, "SIZE"),
                (69, 71, "PRICE"),
                (72, 83, "ITEM_ID"),
                (84, 85, "SIZE"),
                (88, 90, "PRICE"),
                (91, 102, "ITEM_ID"),
                (103, 105, "SIZE"),
                (108, 110, "PRICE"),
                (111, 123, "ITEM_ID"),
                (124, 127, "SIZE"),
                (130, 132, "PRICE"),
                (133, 146, "ITEM_ID")
            ]
        }),
    (
        "CDV136530 - BLACK T BORN TO DO IT S: $45 CDV136530-S M: $45 CDV136530-M L: $45 CDV136530-L XL: $45 CDV136530-XL 2XL: $45 CDV136530-2XL",
        {"entities":
            [
                (0, 33, "NAME"),
                (34, 35, "SIZE"),
                (38, 40, "PRICE"),
                (41, 52, "ITEM_ID"),
                (53, 54, "SIZE"),
                (57, 59, "PRICE"),
                (60, 71, "ITEM_ID"),
                (72, 73, "SIZE"),
                (76, 78, "PRICE"),
                (79, 90, "ITEM_ID"),
                (91, 93, "SIZE"),
                (96, 98, "PRICE"),
                (99, 111, "ITEM_ID"),
                (112, 115, "SIZE"),
                (118, 120, "PRICE"),
                (121, 134, "ITEM_ID")
            ]
        }),
    ("CDV136532 - BLACK HAT BORN TO DO IT One-Size: $35 CDV136532-OS",
     {"entities":
         [
             (0, 35, "NAME"),
             (36, 44, "SIZE"),
             (47, 49, "PRICE"),
             (50, 62, "ITEM_ID"),
         ]
     }),
    ("2 # Item $12 On-size:",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
         ]
     }),
    ("2 $ Item $12 On-size",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
         ]
     }),
    ("2 $ Item $12 On-size: MESFAWERHS2DFN",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
             (22, 36, "ITEM_ID"),
         ]
     }),
    ("2 $ Item $12 On-size: MESFAWERHS2DFN-XL $10 XL: MESFAWERHS2DFN-XL",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
             (22, 39, "ITEM_ID"),

             (41, 43, "PRICE"),
             (44, 46, "SIZE"),
             (48, 65, "ITEM_ID"),
         ]
     }),
    ("2 @ Item $12 On-size: MESFAWERHS2DFN-XL $10 XL: MESFAWERHS2DFN-XL",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
             (22, 39, "ITEM_ID"),

             (41, 43, "PRICE"),
             (44, 46, "SIZE"),
             (48, 65, "ITEM_ID"),
         ]
     }),
    ("2 @ Item $12 On-size",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
         ]
     }),
    ("25@2Item $12 On-size: MESFAWERHS2DFN-XL $10 XL: MESFAWERHS2DFN-XL",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
             (22, 39, "ITEM_ID"),

             (41, 43, "PRICE"),
             (44, 46, "SIZE"),
             (48, 65, "ITEM_ID"),
         ]
     }),
    ("2542Item $12 On-size: MESFAWERHS2DFN-XL $10 XL: MESFAWERHS2DFN-XL",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
             (22, 39, "ITEM_ID"),

             (41, 43, "PRICE"),
             (44, 46, "SIZE"),
             (48, 65, "ITEM_ID"),
         ]
     }),
    ("254Item3 $12 On-size: MESFAWERHS2DFN-XL $10 XL: MESFAWERHS2DFN-XL",
     {"entities":
         [
             (0, 8, "NAME"),
             (10, 12, "PRICE"),
             (13, 20, "SIZE"),
             (22, 39, "ITEM_ID"),

             (41, 43, "PRICE"),
             (44, 46, "SIZE"),
             (48, 65, "ITEM_ID"),
         ]
     }),
    ("SIG129073 - RED / NATURA TOTE BW BURNING $25 SIG129073-NS",
     {"entities":
         [
             (0, 40, "NAME"),
             (42, 44, "PRICE"),
             (45, 57, "ITEM_ID")
         ]
     }),
    ("JJ - Y2K Pepper Tee S: $45 939530123 M: $45 939530124 L: $45 939530125 XL: $45 939530126 2XL: $45 939530127",
     {"entities":
         [
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
     }),
    ("JJ - Vintage Black White Photo Tee S: $45 939530113 M: $45 939530114 L: $45 939530115 XL: $45 939530116 2XL: $45 939530117",
     {"entities":
         [
             (0, 34, "NAME"),
             (35, 36, "SIZE"),
             (39, 41, "PRICE"),
             (42, 51, "ITEM_ID"),
             (52, 53, "SIZE"),
             (56, 58, "PRICE"),
             (59, 68, "ITEM_ID"),
             (69, 70, "SIZE"),
             (73, 75, "PRICE"),
             (76, 85, "ITEM_ID"),
             (86, 88, "SIZE"),
             (91, 93, "PRICE"),
             (94, 103, "ITEM_ID"),
             (104, 107, "SIZE"),
             (110, 112, "PRICE"),
             (113, 122, "ITEM_ID"),
         ]
     }),
    ("JJ - Nasty Tee S: $45 939530143 M: $45 939530144 L: $45 939530145 XL: $45 939530146 2XL: $45 939530147",
     {"entities":
         [
             (0, 14, "NAME"),
             (15, 16, "SIZE"),
             (19, 21, "PRICE"),
             (22, 31, "ITEM_ID"),
             (32, 33, "SIZE"),
             (36, 38, "PRICE"),
             (39, 48, "ITEM_ID"),
             (49, 50, "SIZE"),
             (53, 55, "PRICE"),
             (56, 65, "ITEM_ID"),
             (66, 68, "SIZE"),
             (71, 73, "PRICE"),
             (74, 83, "ITEM_ID"),
             (84, 87, "SIZE"),
             (90, 92, "PRICE"),
             (93, 102, "ITEM_ID"),
         ]
     }),
    (
        "SIG129070 - BLACK T BW ATTA SPECTRUM EUROPE AND NORTH AMERICA ORCHESTRA TOUR DATE S: $45 SIG129070-S M: $45 SIG129070-M L: $45 SIG129070-L XL: $45 SIG129070-XL 2XL: $45 SIG129070-2XL",
        {"entities":
            [
                (0, 81, "NAME"),
                (82, 83, "SIZE"),
                (86, 88, "PRICE"),
                (89, 100, "ITEM_ID"),
                (101, 102, "SIZE"),
                (105, 107, "PRICE"),
                (108, 119, "ITEM_ID"),
                (120, 121, "SIZE"),
                (124, 126, "PRICE"),
                (127, 138, "ITEM_ID"),
                (139, 141, "SIZE"),
                (144, 146, "PRICE"),
                (147, 159, "ITEM_ID"),
                (160, 163, "SIZE"),
                (166, 168, "PRICE"),
                (169, 182, "ITEM_ID")
            ]
        }),

]


def train_spacy_ner(train_data, iterations):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                nlp.update(examples, drop=0.5, losses=losses)
            print(f"Iteration {itn}, Losses: {losses}")

    return nlp


# Train the model
nlp = train_spacy_ner(TRAIN_DATA, iterations=1000)
nlp.to_disk("items-pdf-model")
