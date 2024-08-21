import pymupdf4llm
import pathlib
import re
from items_custom_model.run_model import process_multiple_texts
from image_processing import extract_and_save_images


def parse_text(text):
    pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    matches = re.findall(pattern, text, re.DOTALL)
    items = [item.strip() for sublist in matches for item in sublist]
    cleaned_items = [item for item in items if item.strip() != '---']
    return cleaned_items


def pdf_data_extraction():
    try:
        global __pdf_path
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
        print()
        return process_multiple_texts(__result_from_pdf_extraction)
    except Exception as e:
        print(f'API call exception occurred {e}')


def main():
    extracted_data = pdf_data_extraction()
    trained_result = callCustomModel(extracted_data)
    print(trained_result)


if __name__ == "__main__":
    __pdf_path = "CDV.pdf"
    __output_dir = '/Users/safwanoffice/PycharmProjects/items-pdf-integrated-llm/outputs'
    __merchant_name = 'JONS'
    main()
