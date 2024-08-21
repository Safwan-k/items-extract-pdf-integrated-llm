import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import boto3 as boto3
import fitz
import uuid

from secrets_custom_s3_bucket import __aws_secret_access_key, __aws_access_key_id, bucket


def extract_and_save_images(image_list, merchant_name, pdf_path):
    image_info = []
    image_url = []
    doc = fitz.open(pdf_path)

    for i in range(len(image_list)):

        for index, image in enumerate(image_list[i]['images']):
            image_pix = doc[i].get_pixmap(clip=image['bbox'])
            image_bytes = image_pix.tobytes()
            image_filename = f"page{i + 1}_img{index + 1}.png"
            date_str = datetime.now().strftime("%Y-%m-%d")

            # TODO change in production
            random_uuid = uuid.uuid4()

            object_name = f'{merchant_name}_{date_str}/{image_filename}'
            final_image_url = 'https://upload-file-pdf.s3.ap-south-1.amazonaws.com/' + object_name
            image_url.append(final_image_url)
            print(image_url)

            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(uploading_image, object_name, image_bytes)

            image_info.append({
                "page": i + 1,
                "index": index + 1,
                "filename": final_image_url,
                "position": {
                    "x0": round(image['bbox'][0], 2),
                    "y0": round(image['bbox'][1], 2),
                    "x1": round(image['bbox'][2], 2),
                    "y1": round(image['bbox'][3], 2)
                },
            })

    doc.close()

    return image_info


def uploading_image(image_filename, image_bytes):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=__aws_access_key_id,
            aws_secret_access_key=__aws_secret_access_key
        )
        content_type = 'image/png' if True else 'text/plain'
        s3_client.put_object(Bucket=bucket, Key=image_filename, Body=image_bytes, ContentType=content_type)
        print(f"Uploaded {image_filename} to {bucket}")

    except Exception as e:
        print(f"Failed to upload {image_filename}: {e}")


def round_tuple_values(t, decimal_places=2):
    return tuple(round(v, decimal_places) for v in t)


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    num_pages = document.page_count

    all_pages_bboxlog = []
    for page_num in range(num_pages):
        page = document.load_page(page_num)
        # Extract text with its font information
        text_bbox = page.get_bboxlog()

        text_bbox = [entry for entry in text_bbox if entry[0] in ['fill-image', 'fill-text', 'stroke-path']]

        result = []
        current_group = []
        has_image = False

        # Process the data
        for item in text_bbox:
            type_, coordinates = item
            if type_ == 'stroke-path':
                if current_group:
                    if current_group and any('IMAGE' in element for element in current_group):
                        # Combine all 'text' entries into a single 'text' entry
                        combined_text = " ".join([element['TEXT'] for element in current_group if 'TEXT' in element])
                        # Replace the first text element with the combined text
                        current_group = [{'TEXT': combined_text}] + [element for element in current_group if
                                                                     'IMAGE' in element]
                        result.append(current_group)
                    current_group = []
                has_image = False
            else:
                if type_ == 'fill-image':
                    current_group.append({
                        "IMAGE": round_tuple_values(coordinates)
                    })
                    has_image = True
                elif has_image and type_ == 'fill-text':
                    continue  # Skip this 'fill-text' if there was a 'fill-image' before it
                else:
                    current_group.append({
                        "TEXT": page.get_textbox(coordinates)
                    })

        # Append the last group if exists

        if current_group and any('IMAGE' in element for element in current_group):
            combined_text = " ".join([element['TEXT'] for element in current_group if 'TEXT' in element])
            current_group = [{'TEXT': combined_text}] + [element for element in current_group if 'IMAGE' in element]
            result.append(current_group)

        # Print result
        print(json.dumps(result, indent=4))
        all_pages_bboxlog.append(result)

    return all_pages_bboxlog


def bbox_to_image_dict(image_data):
    image_dict = {}
    for item in image_data:
        pos = (round(item['position']['x0'], 2), round(item['position']['y0'], 2), round(item['position']['x1'], 2),
               round(item['position']['y1'], 2))
        image_dict[pos] = item['filename']
    return image_dict


def update_items_with_images(trained_result, image_dict):
    print(image_dict)
    """Update items' image field based on bbox and image_dict."""
    for item in trained_result:
        if 'IMAGE' not in item:
            item['IMAGE'] = ''
        elif item['IMAGE']:
            bbox = item['IMAGE']
            print(bbox)
            images_closest = check_bbox(bbox, image_dict)
            print(images_closest)
            item['IMAGE'] = images_closest

    return trained_result


def check_bbox(bbox_to_check, image_dict):
    for bbox, url in image_dict.items():
        # Check if bbox_to_check exactly matches a bbox in image_dict
        if bbox_to_check == bbox:
            print("Exact match found")
            return url
        # Check if bbox_to_check is within the bbox in image_dict
        if (bbox_to_check[0] >= bbox[0] and
                bbox_to_check[1] >= bbox[1] and
                bbox_to_check[2] <= bbox[2] and
                bbox_to_check[3] <= bbox[3]):
            print("Bounding box is within:")
            return url
    return []
