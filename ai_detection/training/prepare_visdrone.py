"""
Extract and Convert VisDrone-DET Dataset to YOLOv8 Format for TRINETRA.
Parses VisDrone annotations (bbox_left, bbox_top, bbox_width, bbox_height, score, category)
into standard normalized YOLOv8 format (class_id x_center y_center width height).
"""

import os
import zipfile
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_visdrone")

# VisDrone class mapping to YOLO classes
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def prepare_visdrone(train_zip: str, val_zip: str = None, output_dir: str = "data/visdrone", max_samples: int = 800):
    logger.info(f"Extracting and converting VisDrone-DET from: {train_zip}")

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    with zipfile.ZipFile(train_zip, 'r') as z:
        # Find all images in the zip
        all_imgs = sorted([x for x in z.namelist() if x.endswith(".jpg") and "images" in x])
        if not all_imgs:
            all_imgs = sorted([x for x in z.namelist() if x.endswith(".jpg")])

        # Pick calibrated sample
        selected_imgs = all_imgs[:max_samples]
        split_idx = int(len(selected_imgs) * 0.85)
        train_imgs = selected_imgs[:split_idx]
        val_imgs = selected_imgs[split_idx:]

        for split_list, target_img_dir, target_lbl_dir, split_name in [
            (train_imgs, train_img_dir, train_lbl_dir, "train"),
            (val_imgs, val_img_dir, val_lbl_dir, "val")
        ]:
            processed = 0
            for img_path in split_list:
                base_name = os.path.basename(img_path)
                stem = os.path.splitext(base_name)[0]
                # Look for annotation file
                ann_candidates = [
                    f"VisDrone2019-DET-train/annotations/{stem}.txt",
                    f"annotations/{stem}.txt",
                ]
                ann_path = next((c for c in ann_candidates if c in z.namelist()), None)

                # Read image data
                img_data = z.read(img_path)
                out_img_path = os.path.join(target_img_dir, base_name)
                with open(out_img_path, "wb") as f:
                    f.write(img_data)

                # Get image dimensions
                try:
                    with Image.open(out_img_path) as im:
                        width, height = im.size
                except Exception:
                    width, height = 1920, 1080

                dw = 1.0 / max(1, width)
                dh = 1.0 / max(1, height)

                lines = []
                if ann_path:
                    raw_text = z.read(ann_path).decode('utf-8', errors='ignore')
                    for row in raw_text.strip().splitlines():
                        parts = [p.strip() for p in row.split(",") if p.strip()]
                        if len(parts) >= 8:
                            score = parts[4]
                            cat = int(parts[5])
                            # Score 0 = ignored region; cat 0 = ignored, cat 11 = others
                            if score != "0" and 1 <= cat <= 10:
                                x, y, w, h = map(float, parts[:4])
                                cls_id = cat - 1  # 0-indexed
                                xc = (x + w / 2.0) * dw
                                yc = (y + h / 2.0) * dh
                                nw = w * dw
                                nh = h * dh
                                if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0:
                                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

                with open(os.path.join(target_lbl_dir, f"{stem}.txt"), "w") as f:
                    f.writelines(lines)

                processed += 1

            logger.info(f"Processed {processed} VisDrone images & YOLO annotations for {split_name}.")

    # Generate data.yaml
    names_dict = "\n".join([f"  {i}: {name}" for i, name in enumerate(VISDRONE_CLASSES)])
    yaml_content = f"""path: {os.path.abspath(output_dir).replace(chr(92), '/')}
train: images/train
val: images/val
test: images/val

names:
{names_dict}
"""
    yaml_file = os.path.join(output_dir, "data.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    logger.info(f"VisDrone dataset ready: {yaml_file}")
    return yaml_file


if __name__ == "__main__":
    train_zip_path = os.path.expanduser("~/Downloads/VisDrone2019-DET-train.zip")
    if os.path.exists(train_zip_path):
        prepare_visdrone(train_zip_path)
    else:
        print("VisDrone2019-DET-train.zip not found in Downloads.")
