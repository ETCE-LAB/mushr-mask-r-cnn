#  Copyright (C) 2022, 2023 Anant Sujatanagarjuna

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from detectron2.utils.visualizer import GenericMask, VisImage
import numpy as np

def get_points(poly):
    iterator = iter(poly)
    points=[[int(x), int(next(iterator))] for x in iterator ]
    return np.array(points, np.int32)

def get_polygon_predictions(img_rgb_mushr, predictions):

    # boxes = [(float(box[0]), float(box[1]), float(box[2]), float(box[3])) for box in predictions.pred_boxes]
    # scores = [float(score.cpu()) for score in predictions.scores]
    # classes = predictions.pred_classes.tolist()
    # masks = np.asarray(predictions.pred_masks.cpu())
    # dummy_output = VisImage(np.asarray(img_rgb_mushr).clip(0,255).astype(np.uint8), scale=1.0)
    # masks = [GenericMask(x, dummy_output.height, dummy_output.width) for x in masks]
    # has_holes = [bool(generic_mask.has_holes) for generic_mask in masks]
    # polygons = [[polygon.tolist() for polygon in generic_mask.polygons] for generic_mask in masks]
    # areas = [int(generic_mask.area()) for generic_mask in masks]

    instances = predictions["instances"]
    boxes = [(float(box[0]), float(box[1]), float(box[2]), float(box[3])) for box in instances.pred_boxes]
    scores = [float(score.cpu()) for score in instances.scores]
    classes = instances.pred_classes.tolist()
    masks = np.asarray(instances.pred_masks.cpu())
    dummy_output = VisImage(np.asarray(img_rgb_mushr).clip(0,255).astype(np.uint8), scale=1.0)
    masks = [GenericMask(x, dummy_output.height, dummy_output.width) for x in masks]
    has_holes = [bool(generic_mask.has_holes) for generic_mask in masks]
    polygons = [[polygon.tolist() for polygon in generic_mask.polygons] for generic_mask in masks]
    areas = [generic_mask.area() for generic_mask in masks]

    return {
        "boxes": boxes,
        "classes": classes,
        "scores": scores,
        "has_holes": has_holes,
        "polygons": polygons,
        "masks": masks,
        "area": areas
    }

def get_clusters(polygon_predictions):

    num_mushrooms = len(polygon_predictions["boxes"])

    centers = [((x1+x2)/2, (y1+y2)/2) for (x1, y1, x2, y2) in polygon_predictions["boxes"]]

    clustering_eps = 2 * 2.5 * (sum(polygon_predictions["area"])/num_mushrooms)**0.5 / 3.1415

    dbscan = DBSCAN(eps=clustering_eps,
                    min_samples=1,
                    metric="euclidean")

    flushes = dbscan.fit(centers)

    return flushes.labels_, centers

def get_flushes(img_rgb_mushr, predictions):

    polygon_predictions = get_polygon_predictions(img_rgb_mushr, predictions)
    
    clusters, centers = get_clusters(polygon_predictions)
    flushes = {}
    num = 0
    flush_map = {}

    for i, cluster in enumerate(clusters):
        if cluster not in flush_map:
            if cluster < 0:
                continue
            flush_map[cluster] = []
        flush_map[cluster].append(i)

    for cluster, indices in flush_map.items():
        flushes[cluster] = {
            "indices":indices,
            "classes":[polygon_predictions["classes"][i] for i in indices],
            "scores":[polygon_predictions["scores"][i] for i in indices],
            "area":[polygon_predictions["area"][i] for i in indices],
            "centers":[centers[i] for i in indices],
            "polygons":[polygon_predictions["polygons"][i] for i in indices],
            "percentages": {1:0,2:0,3:0}
        }

    for cluster, flush in flushes.items():
        total_areas = {1:.0,2:.0,3:.0}
        for i, cl in enumerate(flush["classes"]):
            total_areas[cl]+= flush["area"][i] * flush["scores"][i]
            
        total_area = total_areas[1] + total_areas[2] + total_areas[3]
        flushes[cluster]["percentages"][1] = 100 * total_areas[1]/total_area
        flushes[cluster]["percentages"][2] = 100 * total_areas[2]/total_area
        flushes[cluster]["percentages"][3] = 100 * total_areas[3]/total_area
            
    return flushes

def stage(percentages):
    gm = {1: "Not Ready",
          2: "Ready",
          3: "Overdue"}
    k = max(percentages, key=percentages.get)


    return(f"{gm[k]}: {percentages[k]}% ")
    
