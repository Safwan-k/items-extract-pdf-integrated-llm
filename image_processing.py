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

        page_width = doc[i].rect.width

        for index, image in enumerate(image_list[i]['images']):
            if round(image['bbox'][0], 2) < page_width / 2:
                column = "Left"
            else:
                column = "Right"
            print(column)

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
                "column": column
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
