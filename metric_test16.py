
from ultralytics import YOLO
import numpy as np
import cv2
import csv
import math
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import os
import glob


def calculate_stomatal_metrics(centroids, image_width, image_height, pixel_conversion):
    num_stomata = len(centroids)

    pixels_per_mm = pixel_conversion * 10.0 
    image_area_mm2 = (image_width * image_height) / (pixels_per_mm ** 2)
    density = num_stomata / image_area_mm2 if image_area_mm2 > 0 else 0

    if num_stomata < 2:
        return 0.0, 0.0, 0.0, density

    SEve = 0.0
    if num_stomata >= 3: 
        dist_matrix_val = distance_matrix(centroids, centroids)
        mst = minimum_spanning_tree(dist_matrix_val)

        edges = mst.data[mst.data > 0]
        total_length = np.sum(edges)

        if total_length > 0:
            PD = edges / total_length  # 公式4
            constant = 1 / (num_stomata - 1)
            sum_min = np.sum(np.minimum(PD, constant))  # ∑min(PD_l, 1/(N-1))
            SEve = (sum_min - constant) / (1 - constant)  
        else:  
            SEve = 0.0

    centroid = np.mean(centroids, axis=0)  
    dG = np.linalg.norm(centroids - centroid, axis=1)  
    mean_dG = np.mean(dG)  

    delta_d = np.sum(dG - mean_dG)  
    delta_abs_d = np.sum(np.abs(dG - mean_dG))  

    # 公式11
    numerator = delta_d + mean_dG
    denominator = delta_abs_d + mean_dG

    if denominator == 0: 
        SDiv = 1.0
    else:
        SDiv = numerator / denominator

    theoretical_distance = 1 / (2 * np.sqrt(density)) if density > 0 else 0.0

    if num_stomata > 1:
        dist_matrix_val = distance_matrix(centroids, centroids)
        np.fill_diagonal(dist_matrix_val, np.inf)
        nn_distances = np.min(dist_matrix_val, axis=1)
        nn_distances_mm = nn_distances / pixels_per_mm
        observed_distance = np.mean(nn_distances_mm)
    else:  # num_stomata == 1
        observed_distance = 0.0

    if theoretical_distance > 0:
        SAgg = observed_distance / theoretical_distance
    elif observed_distance == 0:
        SAgg = 0.0
    else:
        SAgg = float('inf')

    return SEve, SDiv, SAgg, density


model_path = "/root/autodl-tmp/ultralytics-yolo11-20250721/ultralytics-yolo11-main/runs/train/exp35/weights/best_fp32.pt"
input_dir = "/root/autodl-tmp/ultralytics-main/mydata_prediction/input_image" 
output_dir = "/root/autodl-tmp/ultralytics-main/mydata_prediction/output"
pixel_conversion = 300  
confidence_threshold = 0.7 

os.makedirs(output_dir, exist_ok=True)

summary_seg_csv = os.path.join(output_dir, "all_images_segmentation_stats.csv")
summary_dist_csv = os.path.join(output_dir, "all_images_distribution_metrics.csv")

model = YOLO(model_path)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
              glob.glob(os.path.join(input_dir, "*.jpeg")) + \
              glob.glob(os.path.join(input_dir, "*.png")) + \
              glob.glob(os.path.join(input_dir, "*.tif")) + \
              glob.glob(os.path.join(input_dir, "*.tiff")) + \
              glob.glob(os.path.join(input_dir, "*.bmp"))


with open(summary_seg_csv, mode='w', newline='', encoding='utf-8-sig') as seg_file, \
        open(summary_dist_csv, mode='w', newline='', encoding='utf-8-sig') as dist_file:
    seg_writer = csv.writer(seg_file)
    dist_writer = csv.writer(dist_file)

    seg_header = ["Sample Name", "Class Name", "ID", "Area (px)", "Perimeter (px)", "Width (px)",
                  "Height (px)", "Angle (degrees)", "Count", "Centroid X", "Centroid Y"]
    seg_writer.writerow(seg_header)

    dist_header = ["Image Name", "SEve (Stomatal Evenness)", "SDiv (Stomatal Divergence)",
                   "SAgg (Stomatal Aggregation)", "Density (stomata/mm²)",
                   "Stomata Count", "Pore Count"]
    dist_writer.writerow(dist_header)

    for image_path in image_files:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n处理图片: {image_name}")

        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)

        segmentation_csv = os.path.join(image_output_dir, f"{image_name}_segmentation_stats.csv")
        distribution_csv = os.path.join(image_output_dir, f"{image_name}_distribution_metrics.csv")
        visualization_filename = os.path.join(image_output_dir, f"{image_name}_visualization.jpg")

        category_counts = {}
        category_ids = {}

        stomata_centroids = []
        pore_centroids = []
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"warning: {image_path}")
            continue

        image_height, image_width = img.shape[:2]

        results = model.predict(source=image_path, conf=confidence_threshold)

        with open(segmentation_csv, mode='w', newline='', encoding='utf-8-sig') as image_seg_file:
            image_seg_writer = csv.writer(image_seg_file)
            image_seg_writer.writerow(seg_header)

            for result in results:
                original_image = result.orig_img.copy()

                mask_original = np.zeros_like(original_image[:, :, 0])
                mask_ellipse = np.zeros_like(original_image[:, :, 0])

                for i, detection in enumerate(result):
                    class_name = detection.names[int(detection.boxes.cls[0])]
                    class_id = int(detection.boxes.cls[0])

                    if class_name not in category_ids:
                        category_ids[class_name] = 0
                    category_counts[class_name] = category_counts.get(class_name, 0) + 1

                    color = colors[class_id % len(colors)]
                    target_id = category_ids[class_name]
                    category_ids[class_name] += 1

                    if detection.masks is not None and len(detection.masks.xy) > 0:
                        mask = np.zeros_like(original_image[:, :, 0])

                        all_points = []
                        for contour_points in detection.masks.xy:
                            points = contour_points.astype(np.int32)
                            all_points.append(points)

                        cv2.fillPoly(mask, [np.array(all_points)], 255)
                        
                        mask_original = cv2.bitwise_or(mask_original, mask)

                        contour_points_concat = np.concatenate([np.array(p).reshape(-1, 1, 2) for p in all_points])
                        original_area = cv2.contourArea(contour_points_concat)

                        if original_area < 10:  
                            print(f"SKIP: {class_name}{target_id}, area: {original_area:.1f}")
                            continue

                        if len(contour_points_concat) >= 5:  
                            try:
                                ellipse = cv2.fitEllipse(contour_points_concat)

                                (center_x, center_y), (major_axis, minor_axis), angle = ellipse

                                ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)

                                if ellipse_area < 1e-5:  
                                    raise ValueError("invalid")

                                area_ratio = math.sqrt(original_area / ellipse_area)

                                if area_ratio > 3.0 or area_ratio < 0.33:
                                    print(f"WARNING: {area_ratio:.2f}, original")
                                    raise ValueError("oversize")

                                adjusted_major_axis = major_axis * area_ratio
                                adjusted_minor_axis = minor_axis * area_ratio

                                adjusted_ellipse = ((center_x, center_y),
                                                    (adjusted_major_axis, adjusted_minor_axis),
                                                    angle)

                                ellipse_contour = cv2.ellipse2Poly(
                                    (int(center_x), int(center_y)),
                                    (int(adjusted_major_axis / 2), int(adjusted_minor_axis / 2)),
                                    int(angle), 0, 360, 5
                                )

                                ellipse_contour_area = cv2.contourArea(ellipse_contour)

                                area_diff = abs(ellipse_contour_area - original_area) / original_area
                                if area_diff > 0.3: 
                                    print(f"OVERSIZE: {area_diff * 100:.1f}%, ORIGINAL")
                                    raise ValueError("OVERSIZE")

                                cv2.drawContours(original_image, [ellipse_contour], -1, color, thickness=2)

                                cv2.fillPoly(mask_ellipse, [ellipse_contour], 255)

                                rect = cv2.minAreaRect(ellipse_contour)
                                box = cv2.boxPoints(rect)
                                box = box.astype(np.int32)

                                width, height = rect[1]
                                angle = rect[2]

                                if width < height:
                                    width, height = height, width
                                    angle = angle - 90 if angle > 0 else angle + 90

                                if angle < -45:
                                    angle += 90
                                elif angle > 90:
                                    angle -= 90

                                perimeter = cv2.arcLength(ellipse_contour, closed=True)

                                mask_area = ellipse_contour_area

                                if class_name == "stomata":
                                    stomata_centroids.append((center_x, center_y))
                                elif class_name == "pore":
                                    pore_centroids.append((center_x, center_y))
                                    
                                cv2.polylines(original_image, [box], isClosed=True, color=color, thickness=2)

                                label = f"{class_name}{target_id}"
                                cv2.putText(original_image, label, (int(center_x), int(center_y)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                            
                                row = [image_name, class_name, target_id, mask_area, perimeter, width, height,
                                       angle, category_counts[class_name], center_x, center_y]
                                image_seg_writer.writerow(row)
                                seg_writer.writerow(row)

                            except Exception as e:
                                print(f"target {class_name}{target_id} fail: {str(e)}")
                                use_original_contour = True
                        else:
                            use_original_contour = True

                        if len(contour_points_concat) < 5:

                            contour = contour_points_concat

                            cv2.drawContours(original_image, [contour], -1, color, thickness=2)

                            cv2.fillPoly(mask_ellipse, [contour], 255)

                            rect = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect)
                            box = box.astype(np.int32)

                            width, height = rect[1]
                            angle = rect[2]

                            if width < height:
                                width, height = height, width
                                angle = angle - 90 if angle > 0 else angle + 90

                            if angle < -45:
                                angle += 90
                            elif angle > 90:
                                angle -= 90

                            perimeter = cv2.arcLength(contour, closed=True)

                            mask_area = original_area

                            center_x, center_y = rect[0]

                            if class_name == "stomata":
                                stomata_centroids.append((center_x, center_y))
                            elif class_name == "pore":
                                pore_centroids.append((center_x, center_y))

                            cv2.polylines(original_image, [box], isClosed=True, color=color, thickness=2)

                            label = f"{class_name}{target_id}"
                            cv2.putText(original_image, label, (int(center_x), int(center_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            row = [image_name, class_name, target_id, mask_area, perimeter, width, height,
                                   angle, category_counts[class_name], center_x, center_y]
                            image_seg_writer.writerow(row)
                            seg_writer.writerow(row)

                cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_mask_original.png"), mask_original)
                cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_mask_ellipse.png"), mask_ellipse)

                cv2.imwrite(visualization_filename, original_image)

        if stomata_centroids:
            SEve, SDiv, SAgg, density = calculate_stomatal_metrics(
                np.array(stomata_centroids), image_width, image_height, pixel_conversion)
        else:
            SEve, SDiv, SAgg, density = 0.0, 0.0, 0.0, 0.0

        dist_row = [image_name, f"{SEve:.4f}", f"{SDiv:.4f}",
                    f"{SAgg:.4f}", f"{density:.1f}",
                    len(stomata_centroids), len(pore_centroids)]

        with open(distribution_csv, mode="w", newline="", encoding="utf-8-sig") as image_dist_file:
            image_dist_writer = csv.writer(image_dist_file)
            image_dist_writer.writerow(dist_header)
            image_dist_writer.writerow(dist_row)

        dist_writer.writerow(dist_row)
        
        if stomata_centroids:
            print(f" arrangement indices:")
            print(f"    SEve: {SEve:.4f}, SDiv: {SDiv:.4f}, SAgg: {SAgg:.4f}, 密度: {density:.1f}个/mm²")

        print(f"  result save in: {image_output_dir}")
