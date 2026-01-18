from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.geometry import Point
import numpy as np


def find_nearest_neighbours(trees):
    """Her ağaç için en yakın komşunun index'ini döndürür."""
    if len(trees) < 2:
        return dict()

    polygons = [tree.polygon for tree in trees]

    neighbors = {}
    for i, tree in enumerate(trees):
        min_dist = float('inf')
        nearest_idx = 0
        
        for j, other in enumerate(trees):
            if i != j:
                dist = tree.polygon.distance(other.polygon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = j
        
        neighbors[i] = nearest_idx
    
    return neighbors


def get_rotated_rect_features(rect):
    """
    Bir minimum_rotated_rectangle'dan 5 özellik çıkarır.
    Returns: (center_x, center_y, width, height, angle)
    """
    coords = list(rect.exterior.coords)[:4]
    
    # Merkez
    cx, cy = rect.centroid.x, rect.centroid.y
    
    # Kenar uzunlukları
    edge1 = Point(coords[0]).distance(Point(coords[1]))
    edge2 = Point(coords[1]).distance(Point(coords[2]))
    width, height = min(edge1, edge2), max(edge1, edge2)
    
    # Açı (uzun kenarın x-ekseniyle yaptığı açı)
    if edge1 > edge2:
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
    else:
        dx = coords[2][0] - coords[1][0]
        dy = coords[2][1] - coords[1][1]
    angle = np.arctan2(dy, dx)
    
    return (cx, cy, width, height, angle)


def get_pairwise_rotated_rects(trees):
    """
    Her ağaç çifti için minimum_rotated_rectangle döndürür.
    Sonra bunların unary_union'ını alabilirsin.
    
    Returns: 
        list[Polygon]: Her çiftin minimum_rotated_rectangle'ları
    """
    if len(trees) < 2:
        if len(trees) == 1:
            return [trees[0].polygon.minimum_rotated_rectangle]
        return []
    
    neighbors = find_nearest_neighbours(trees)
    pair_rects = []

    for tree_idx, neighbor_idx in neighbors.items():
        tree = trees[tree_idx]
        neighbor = trees[neighbor_idx]
        
        pair_union = unary_union([tree.polygon, neighbor.polygon])
        pair_rect = pair_union.minimum_rotated_rectangle
        pair_rects.append(pair_rect)
    
    return pair_rects


def get_union(poly1, poly2):
    """İki polygon'un birleşimini döndürür."""
    return unary_union([poly1, poly2])


# Kullanım:
# pair_rects = get_pairwise_rotated_rects(placed_trees)
# all_rects_union = unary_union(pair_rects)