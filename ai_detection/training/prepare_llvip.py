"""
Extract and Convert LLVIP Low-Light & Thermal Night Surveillance Dataset to YOLOv8 Format.
Extracts a calibrated split (500 train, 100 val) and converts Pascal VOC XML annotations to YOLO format.
"""

import os
import xml.etree.ElementTree as ET
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_llvip")


def prepare_llvip_dataset(zip_path: str, output_dir: str = "data/llvip", train_count: int = 500, val_count: int = 100):
    logger.info(f"Extracting & converting LLVIP dataset from: {zip_path}")

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Collect visible train images
        all_train_imgs = sorted([x for x in z.namelist() if x.startswith("LLVIP/visible/train/") and x.endswith(".jpg")])
        all_test_imgs = sorted([x for x in z.namelist() if x.startswith("LLVIP/visible/test/") and x.endswith(".jpg")])

        selected_train = all_train_imgs[:train_count]
        selected_val = all_test_imgs[:val_count] if all_test_imgs else all_train_imgs[train_count:train_count+val_count]

        for split_imgs, target_img_dir, target_lbl_dir, split_name in [
            (selected_train, train_img_dir, train_lbl_dir, "train"),
            (selected_val, val_img_dir, val_lbl_dir, "val")
        ]:
            extracted = 0
            for img_path in split_imgs:
                base_name = os.path.basename(img_path)
                file_stem = os.path.splitext(base_name)[0]
                xml_path = f"LLVIP/Annotations/{file_stem}.xml"

                # Extract image
                img_data = z.read(img_path)
                with open(os.path.join(target_img_dir, base_name), "wb") as f:
                    f.write(img_data)

                # Extract and parse XML if present
                lines = []
                if xml_path in z.namelist():
                    xml_data = z.read(xml_path).decode('utf-8')
                    try:
                        root = ET.fromstring(xml_data)
                        size = root.find('size')
                        width = float(size.find('width').text) if size is not None else 1280.0
                        height = float(size.find('height').text) if size is not None else 1024.0
                        dw = 1.0 / max(1.0, width)
                        dh = 1.0 / max(1.0, height)

                        for obj in root.findall('object'):
                            bndbox = obj.find('bndbox')
                            if bndbox is not None:
                                xmin = float(bndbox.find('xmin').text)
                                ymin = float(bndbox.find('ymin').text)
                                xmax = float(bndbox.find('xmax').text)
                                ymax = float(bndbox.find('ymax').text)

                                xc = ((xmin + xmax) / 2.0) * dw
                                yc = ((ymin + ymax) / 2.0) * dh
                                bw = (xmax - xmin) * dw
                                bh = (ymax - ymin) * dh
                                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    except Exception as e:
                        logger.warning(f"Error parsing {xml_path}: {e}")

                with open(os.path.join(target_lbl_dir, f"{file_stem}.txt"), "w") as f:
                    f.writelines(lines)

                extracted += 1

            logger.info(f"Processed {extracted} images and YOLO labels for split: {split_name}")

    # Write data.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir).replace(chr(92), '/')}
train: images/train
val: images/val
test: images/val

names:
  0: human
"""
    yaml_file = os.path.join(output_dir, "data.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    logger.info(f"Generated LLVIP dataset descriptor: {yaml_file}")
    return yaml_file


if __name__ == "__main__":
    zip_p = os.path.expanduser("~/Downloads/LLVIP.zip")
    if os.path.exists(zip_p):
        prepare_llvip_dataset(zip_p)
    else:
        print("LLVIP.zip not found in Downloads")
