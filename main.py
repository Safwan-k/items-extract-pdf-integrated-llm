import csv
import difflib
import io
import os
import sys

import pymupdf4llm
import pathlib
import re
from items_custom_model.run_model import process_multiple_texts
from image_processing import extract_and_save_images, extract_text_from_pdf, bbox_to_image_dict, \
    update_items_with_images


def parse_text(text):
    # Updated pattern to capture single items at the end
    pattern = r'\|\s*([^|]+?)\s*(?=\||$)'
    matches = re.findall(pattern, text, re.DOTALL)
    items = [item.strip() for item in matches]
    # Filter out any items that are '---'
    cleaned_items = [
        item for item in items
        if not re.match(r'-+', item) and not re.match(r'Col\d+', item)
    ]

    return cleaned_items


def pdf_data_extraction():
    try:
        global __pdf_path
        global image_bbox_list
        md_text_list = pymupdf4llm.to_markdown(__pdf_path, page_chunks=True)
        image_bbox_list = extract_and_save_images(md_text_list, __merchant_name, __pdf_path)
        print(image_bbox_list)
        md_text = "\n".join(item.get("text", "") for item in md_text_list)
        pathlib.Path("output.md").write_bytes(md_text.encode())
        byte_string = md_text.encode()
        __text = byte_string.decode('utf-8')
        result = parse_text(__text)
        return result
    except Exception as e:
        print(f'exception Occurred to parse data {e}')


def callCustomModel(__result_from_pdf_extraction):
    try:
        print(__result_from_pdf_extraction)
        return process_multiple_texts(__result_from_pdf_extraction)
    except Exception as e:
        print(f'API call exception occurred {e}')


def match_image_with_trained_result(__trained_processed_image_data, __trained_result):
    image_map = {}
    for item in __trained_processed_image_data:
        for items in item:
            text = items[0]['TEXT']
            image_coords = items[1]['IMAGE']
            image_map[text] = image_coords

    # Update the desc with image coordinates
    for item in __trained_result:
        item_names = item['NAME']
        for name in item_names:
            closest_match = difflib.get_close_matches(name, image_map.keys(), n=1)
            if closest_match:
                item['IMAGE'] = image_map[closest_match[0]]
                break  # Assuming each name should map to one image

    return __trained_result


def make_csv(updated_trained_result_with_image, csv_path, folder_path):
    result = []

    for index, item in enumerate(updated_trained_result_with_image):
        name = item.get('NAME', [None])[0] if item.get('NAME') else None  # Handle name being potentially None
        price = item.get('PRICE', [None])[0] if item.get('PRICE') else None  # Handle price being potentially None
        sizes = item.get('SIZE', [])
        item_ids = item.get('ITEM_ID', [])
        image = item.get('IMAGE', None)  # Handle image being potentially None

        if sizes and item_ids:
            for size, item_id in zip(sizes, item_ids):
                result.append({
                    "name": name,
                    "size": size,
                    "price": price,
                    "item_id": item_id,
                    "image": image
                })
        elif sizes:
            # If only sizes are present
            for size in sizes:
                result.append({
                    "name": name,
                    "size": size,
                    "price": price,
                    "item_id": None,
                    "image": image
                })
        elif item_ids:
            # If only item_ids are present
            for item_id in item_ids:
                result.append({
                    "name": name,
                    "size": None,
                    "price": price,
                    "item_id": item_id,
                    "image": image
                })
        else:
            # If neither sizes nor item_ids are present
            result.append({
                "name": name,
                "size": None,
                "price": price,
                "item_id": None,
                "image": image
            })

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

        # Full path to the CSV file
    csv_path = os.path.join(folder_path, csv_path)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["name", "size", "price", "item_id", "image"])
    writer.writeheader()
    for item in result:
        writer.writerow(item)

    # Write the content to the CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        csvfile.write(output.getvalue())


def main():
    extracted_data = pdf_data_extraction()
    __trained_result = callCustomModel(extracted_data)
    print(__trained_result)
    __trained_processed_image_data = extract_text_from_pdf(__pdf_path)
    __image_dict_result = bbox_to_image_dict(image_bbox_list)
    __updated_image_result = match_image_with_trained_result(__trained_processed_image_data, __trained_result)
    __updated_trained_result_with_image = update_items_with_images(__updated_image_result, __image_dict_result)
    make_csv(__updated_trained_result_with_image, __csv_name, __csv_path)

    print(__updated_trained_result_with_image)


if __name__ == "__main__":
    # __pdf_path = "CDV.pdf"
    # __output_dir = '/Users/safwanoffice/PycharmProjects/items-pdf-integrated-llm/outputs'
    # __merchant_name = 'CDV'
    # __csv_path = __merchant_name + '.csv'
    __pdf_path = sys.argv[1]
    __csv_name = sys.argv[2]
    __csv_path = sys.argv[3]
    __merchant_name = sys.argv[4]
    image_bbox_list = []
    main()
