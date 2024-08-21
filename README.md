
# PDF Data Extraction and Image Matching

This project involves extracting text and images from a PDF using PyMuPDF and LLM (Language Model), processing the extracted data, uploading images to an S3 bucket, and converting the processed text into JSON format using a custom model. Finally, the text and images are matched and the result is saved as a CSV file.

## Prerequisites

- Python 3.x
- Required Python packages:
  - PyMuPDF (`fitz`)
  - boto3 (for S3 operations)
  - spaCy (for text processing)
  - Custom Model file on the folder

Install the necessary packages using:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Extract Text and Images from PDF

The `main.py` script will extract text and images from a PDF, process the text, upload images to S3, and match the text with images.

To run the script:
```bash
python3 main.py
```

### 2. Check Alignment of Text Using spaCy

The `test_alignment.py` script tests the alignment of text using spaCy.

To run the test:
```bash
python3 test_alignment.py
```

### 3. Run the Custom Model Manually for Text

The `run_model.py` script allows you to manually invoke the custom model on the parsed text.

To run the script:
```bash
python3 run_model.py
```

## Explanation of Key Steps

### Extract Text and Images from PDF

1. **PDF Text and Image Extraction**: Extract text and the list of image bounding boxes (BBOX) from the PDF using PyMuPDF. The text is filtered and processed to retain only relevant entries.

2. **S3 Upload**: The extracted images are uploaded to an S3 bucket. The BBOX coordinates are associated with the corresponding image URLs.

3. **Text Processing**: The extracted text is parsed and combined if necessary, ensuring that all relevant information is captured before the image entries.

4. **Custom Model Invocation**: The processed text is passed to a custom model to convert it into a structured JSON format.

### Image Matching and CSV Creation

1. **Bounding Box to Image URL Mapping**: From the `imageList`, assign the specific bounding box to the image coordinates and retrieve the image URL.

2. **Match Image with Trained Result**: Match the image BBOX with the item name in the JSON result from the custom model. Attach the image BBOX to the corresponding item.

3. **Update Items with Images**: Attach the image URL to each item based on the matching BBOX.

4. **Create CSV**: Create a CSV file from the final item list, with each size or item ID as a new entry, while retaining the same item name.

## Sample Output

The output will be a CSV file that lists each item, its associated sizes, prices, item IDs, and image URLs.

Example CSV structure:
```
NAME,SIZE,PRICE,ITEM_ID,IMAGE
CDV136573 - WHITE T BORN ALBUM PHOTO,S,45,CDV136573-S,https://item_image_url
CDV136573 - WHITE T BORN ALBUM PHOTO,M,45,CDV136573-M,https://item_image_url
...
```

## Troubleshooting

- Ensure all dependencies are installed and correctly configured.
- Verify your S3 bucket permissions for uploading images.
- Double-check the alignment and processing of text and images.
