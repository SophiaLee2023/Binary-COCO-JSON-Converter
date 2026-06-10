import glob
import json
import os
import cv2

from src.create_annotations import (create_image_annotation, 
                                    create_annotation_format,
                                    find_contours,
                                    get_coco_json_format,
                                    create_category_annotation)


category_ids = {
    'flood_water': 1,
}


MASK_EXT = 'png'
ORIGINAL_EXT = 'jpg'


def process_image(mask_path):
    base = os.path.basename(mask_path).split('.')[0]
    original_file = f'input/images/{base}.{ORIGINAL_EXT}'

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    coco = get_coco_json_format()
    coco['categories'] = create_category_annotation(category_ids)

    image_info = create_image_annotation(file_name=original_file, width=width, height=height)
    coco['images'] = [image_info]
    
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotations = []
    annotation_id = 1

    for contour in contours:
        ann = create_annotation_format(
            contour,
            image_info['id'],
            category_ids,
            annotation_id
        )
        if ann['area'] > 0:
            annotations.append(ann)
            annotation_id += 1

    coco['annotations'] = annotations
    return coco, len(annotations), base


if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
                
    for mask_path in glob.glob(f'input/masks/*.{MASK_EXT}'):
        coco_json, ann_count, base = process_image(mask_path)

        out_path = f'output/{base}.json'
        with open(out_path, 'w') as f:
            json.dump(coco_json, f, indent=4)

        print(f"Created {ann_count} annotations → {out_path}")