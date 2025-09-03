from copy import deepcopy

import kornia as K
import shapely.affinity
import shapely.geometry
import torch
from shapely.validation import make_valid


def make_valid_polygon(shapely_object):
    """
    Given a shapely object, turn it into a valid shapely polygon type.
    In most cases, shapely's make_valid function will fix things.
    Sometimes due to (due to precision errors and things), the polygon
    may turn into Multipolygon (the relevant polygon + some small artifacts),
    in such cases, we can safely take the convex hull.
    """
    valid_object = make_valid(shapely_object)
    if valid_object.is_empty:
        return valid_object
    if valid_object.geom_type in ["MultiPolygon", "GeometryCollection"]:
        multi_polygon = shapely.geometry.MultiPolygon()
        for obj in valid_object:
            if obj.geom_type == "Polygon":
                multi_polygon = multi_polygon.union(obj)
        valid_object = multi_polygon.convex_hull
    if valid_object.geom_type != "Polygon":
        valid_object = shapely.geometry.Polygon()
    return valid_object


def get_bboxes_as_shapely_objects_from_coco_annotations(annotations):
    """
    Given a list of coco annotations, returns bboxes as shapely objects.
    """
    shapely_objects = []
    for obj in annotations:
        shapely_objects.append(shapely.geometry.box(*obj["bbox"]))
    return shapely_objects


def get_segmentation_as_shapely_polygons_from_coco_annotations(annotations):
    """
    Given a list of coco annotations, returns segmentations as shapely objects.
    Note: This function assumes that the segmentation is one continuos region
    i.e. a single polygon.
    """
    shapely_objects = []
    for obj in annotations:
        shapely_object = shapely.geometry.MultiPolygon()
        for obj_segment in obj["segmentation"]:
            x_coords = obj_segment[0::2]
            y_coords = obj_segment[1::2]
            coordinates = [[x, y] for x, y in zip(x_coords, y_coords)]
            shapely_object = shapely_object.union(shapely.geometry.Polygon(coordinates))
        shapely_polygon = make_valid_polygon(shapely_object)
        shapely_objects.append(shapely_polygon)
    return shapely_objects


def convert_shapely_polygon_into_coco_segmentation(shapely_object):
    """
    Given a shapely polygon, return it in coco segmentation format.
    """
    segmentation = []
    for point in shapely_object.exterior.coords[:-1]:
        segmentation.append(point[0])
        segmentation.append(point[1])
    return [segmentation]


def get_axis_aligned_bbox_for_segmentation(shapely_object):
    """
    Given a segmentation polygon, compute the axis aligned bounding box.
    """
    x_coords = [x for x, y in shapely_object.exterior.coords]
    y_coords = [y for x, y in shapely_object.exterior.coords]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def apply_kornia_transformation_to_shapely_objects(
    shapely_objects, transformation, image_shape_as_hw, keep_empty=False
):
    """
    Given a list of shapely objects, applies the given 2D affine transformation to them.
    Each "object" is assumed to be a polygon.
    """

    transformation = transformation.squeeze().reshape(-1)
    transformation_parameters = torch.index_select(
        transformation, 0, torch.tensor([0, 1, 3, 4, 2, 5])
    )
    image_bounding_box = shapely.geometry.box(0, 0, *image_shape_as_hw[::-1])
    is_a_single_object = isinstance(shapely_objects, shapely.geometry.Polygon)
    if is_a_single_object:
        shapely_objects = [shapely_objects]
    transformed_objects = []
    for obj in shapely_objects:
        transformed_obj = make_valid_polygon(
            shapely.affinity.affine_transform(obj, transformation_parameters)
        )
        visible_obj = make_valid_polygon(transformed_obj.intersection(image_bounding_box))
        if not visible_obj.is_empty or keep_empty:
            transformed_objects.append(visible_obj)
    if is_a_single_object and len(transformed_objects) == 0:
        transformed_objects.append(shapely.geometry.Polygon())
    return transformed_objects if not is_a_single_object else transformed_objects[0]


def merge_shapely_polygons_into_annotations(transformed_segmentations, annotations):
    """
    Given a list of shapely objects (geometrically transformed segmentations) along
    with the corresponding coco annotations list, update the coco annotations with
    the transformed values.
    """
    transformed_annotations = []
    for shapely_object, annotation in zip(transformed_segmentations, annotations):
        annotation_copy = deepcopy(annotation)
        if shapely_object.is_empty or shapely_object.geom_type != "Polygon":
            continue
        annotation_copy["segmentation"] = convert_shapely_polygon_into_coco_segmentation(
            shapely_object
        )
        annotation_copy["bbox"] = get_axis_aligned_bbox_for_segmentation(shapely_object)
        transformed_annotations.append(annotation_copy)
    return transformed_annotations


def resize_image_and_annotations(input_image_as_tensor, output_shape_as_hw, annotations=None):
    """
    Given an input image (and optionally its annotations), resize it to the specified output
    dimension.
    """
    resize_augmentation = K.augmentation.Resize(output_shape_as_hw, return_transform=True)
    resized_input_tensor, transformation = resize_augmentation(input_image_as_tensor)
    resized_input_tensor = resized_input_tensor.squeeze()
    if annotations is None:
        return resized_input_tensor, None
    segmentations_as_shapely_objects = get_segmentation_as_shapely_polygons_from_coco_annotations(
        annotations
    )
    transformed_segmentations = apply_kornia_transformation_to_shapely_objects(
        segmentations_as_shapely_objects,
        transformation,
        resized_input_tensor.shape[-2:],
    )
    transformed_annotations = merge_shapely_polygons_into_annotations(
        transformed_segmentations, annotations
    )
    return resized_input_tensor, transformed_annotations

def resize_pairimages_and_annotations(input_image1_as_tensor, input_image2_as_tensor, output_shape_as_hw, annotations=None):
    """
    Given an input image (and optionally its annotations), resize it to the specified output
    dimension.
    """
    resize_augmentation = K.augmentation.Resize(output_shape_as_hw, return_transform=True)
    resized_input1_tensor, transformation = resize_augmentation(input_image1_as_tensor)
    resized_input1_tensor = resized_input1_tensor.squeeze()
    resized_input2_tensor, transformation = resize_augmentation(input_image2_as_tensor)
    resized_input2_tensor = resized_input2_tensor.squeeze()

    if annotations is None:
        return resized_input1_tensor, resized_input2_tensor, None

    segmentations_as_shapely_objects = get_segmentation_as_shapely_polygons_from_coco_annotations(
        annotations
    )
    transformed_segmentations = apply_kornia_transformation_to_shapely_objects(
        segmentations_as_shapely_objects,
        transformation,
        resized_input1_tensor.shape[-2:],
    )
    transformed_annotations = merge_shapely_polygons_into_annotations(
        transformed_segmentations, annotations
    )
    return resized_input1_tensor, resized_input2_tensor, transformed_annotations


def resize_image_and_mask(input_image_as_tensor, output_shape_as_hw, mask_annotations=None):
    """
    Given an input image (and optionally its annotations), resize it to the specified output
    dimension.
    """
    resize_augmentation = K.augmentation.Resize(output_shape_as_hw, return_transform=True)
    resized_input_tensor, transformation = resize_augmentation(input_image_as_tensor)
    resized_mask_tensor, transformation = resize_augmentation(mask_annotations.unsqueeze(0))

    return resized_input_tensor.squeeze(0), resized_mask_tensor.squeeze(0)


def resize_pairimages_and_mask(input_image1_as_tensor, input_image2_as_tensor, output_shape_as_hw, mask=None):
    """
    Given an input image (and optionally its annotations), resize it to the specified output
    dimension.
    """
    resize_augmentation = K.augmentation.Resize(output_shape_as_hw, return_transform=True)
    resized_input1_tensor, transformation = resize_augmentation(input_image1_as_tensor.unsqueeze(0))
    if isinstance(input_image2_as_tensor, list):
        resized_input2_tensor = []
        for i in range(len(input_image2_as_tensor)):
            resized_tensor, transformation = resize_augmentation(input_image2_as_tensor[i].unsqueeze(0)) 
            resized_input2_tensor.append(resized_tensor.squeeze(0))
        resized_mask_tensor, transformation = resize_augmentation(mask.unsqueeze(0))
        return resized_input1_tensor.squeeze(0), resized_input2_tensor, resized_mask_tensor.squeeze(0)
    else:
        resized_input2_tensor, transformation = resize_augmentation(input_image2_as_tensor.unsqueeze(0)) 
    
        resized_mask_tensor, transformation = resize_augmentation(mask.unsqueeze(0))

        return resized_input1_tensor.squeeze(0), resized_input2_tensor.squeeze(0), resized_mask_tensor.squeeze(0)