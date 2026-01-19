from decimal import Decimal
from typing import List, Dict, Tuple, Optional
import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
from shapely.strtree import STRtree

import gymnasium as gym
from gymnasium import spaces

from utils import get_pairwise_rotated_rects, find_nearest_neighbours

# Yarışmada kullanılan scale factor
scale_factor = Decimal('1e15')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))


class TreePackingEnvironment(gym.Env):
    """
    Gymnasium Environment for Santa 2025 Tree Packing Problem.
    
    Agent her adımda bir ağacı (x, y, rotation) ile yerleştirir.
    Amaç: Tüm ağaçları minimum kare bounding box içine sığdırmak.
    """
    
    def __init__(self, n_trees: int, max_trees: int = 200, seed: int = None):
        """
        Args:
            n_trees: Bu episode'da yerleştirilecek ağaç sayısı
            max_trees: Maksimum ağaç sayısı (padding için)
            seed: Random seed
        """
        super().__init__()
        self.n_trees = n_trees
        self.max_trees = max_trees
        self.seed = seed

        self.placed_trees: List[ChristmasTree] = []
        self.current_index = 0
        self._strtree: Optional[STRtree] = None
        self.prev_union_centroid = None  # Track centroid of pairwise rectangles union
        
        # Occupancy grid settings
        self.grid_size = 256  # 256x256 grid for high resolution fill ratios
        
        # Action Space: (x, y, rotation) continuous
        self.action_space = spaces.Box(
            low=np.array([-10.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 360.0], dtype=np.float32)
        )
        
        # Observation Space
        # tree_features: Her ağaç için (x, y, angle, minx, miny, maxx, maxy, ...) = 10 dim
        # pairwise_features: Her ağaç için komşu ile rect özellikleri = 10 dim
        # scalars: Global özellikler (bbox_size, fill_ratio, n_placed, ...) = 5 dim
        self.observation_space = spaces.Dict({
            "tree_features": spaces.Box(-np.inf, np.inf, shape=(max_trees, 10), dtype=np.float32),
            "pairwise_features": spaces.Box(-np.inf, np.inf, shape=(max_trees, 10), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(max_trees,), dtype=np.float32),
            "occupancy_grid": spaces.Box(0, 1, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        })

    # ==================== PROPERTIES ====================
    
    @property
    def bounding_box_size(self) -> float:
        """Mevcut kare bounding box kenar uzunluğu."""
        if not self.placed_trees:
            return 0.0
        
        all_polygons = [tree.polygon for tree in self.placed_trees]
        bounds = unary_union(all_polygons).bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        return max(width, height)

    @property
    def total_tree_area(self) -> float:
        """Yerleştirilmiş ağaçların toplam alanı."""
        return sum(tree.polygon.area for tree in self.placed_trees)

    @property
    def fill_ratio(self) -> float:
        """Doluluk oranı = tree_area / bbox_area."""
        box_size = self.bounding_box_size
        if box_size == 0.0:
            return 0.0
        return self.total_tree_area / (box_size ** 2)

    @property
    def union_of_min_rotated_rectangles(self):
        """Tüm pair rotated rectangle'ların birleşimi."""
        pair_rects = get_pairwise_rotated_rects(self.placed_trees)
        return unary_union(pair_rects)

    # ==================== GYM METHODS ====================
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """
        Environment'ı sıfırlar.

        Returns:
            observation: İlk state dictionary
            info: Ek bilgiler
        """
        super().reset(seed=seed)

        self.placed_trees = []
        self.current_index = 0
        self._strtree = None
        self.prev_union_centroid = None
        
        observation = self.get_observation()
        info = {
            "n_placed": 0,
            "n_remaining": self.n_trees,
            "bbox_size": 0.0
        }
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Bir adım ilerler: Verilen action ile yeni ağaç yerleştirir.
        
        Args:
            action: (x, y, rotation) array
            
        Returns:
            observation: Yeni state
            reward: Bu adımın ödülü
            terminated: Episode bitti mi (tüm ağaçlar yerleşti)
            truncated: Kesintiye uğradı mı
            info: Ek bilgiler
        """
        x, y, rotation = action[0], action[1], action[2]
        
        # İLK AĞAÇ HER ZAMAN MERKEZE
        if len(self.placed_trees) == 0:
            x, y, rotation = 5.0, 5.0, 0.0
        else:
            # Ağaç boyutunu hesaba katarak sınırları daralt
            margin = 0.5
            x = np.clip(x, margin, 10.0 - margin)
            y = np.clip(y, margin, 10.0 - margin)
            rotation = np.clip(rotation, 0.0, 360.0)
        
        # Mevcut bbox ALANI (reward hesabı için)
        old_bbox_area = self.bounding_box_size ** 2 if self.bounding_box_size > 0 else 0
        
        # Yeni ağaç oluştur
        new_tree = ChristmasTree(
            center_x=str(x),
            center_y=str(y),
            angle=str(rotation)
        )
        
        # Çarpışma kontrolü - şimdi çarpışma alanı döndürüyor
        collision_area = self._check_collision(new_tree)
        has_collision = collision_area > 0
        
        if not has_collision:
            # Geçerli yerleştirme
            self.placed_trees.append(new_tree)
            self.current_index += 1
            self._update_strtree()
        
        # Yeni bbox ALANI
        new_bbox_area = self.bounding_box_size ** 2 if self.bounding_box_size > 0 else 0
        
        # Reward hesapla (alan bazlı)
        reward = self._calculate_reward(old_bbox_area, new_bbox_area, collision_area)
        
        # Episode bitti mi?
        terminated = self.current_index >= self.n_trees
        truncated = False
        
        # Observation
        observation = self.get_observation()
        
        # Info
        info = {
            "n_placed": len(self.placed_trees),
            "n_remaining": self.n_trees - self.current_index,
            "bbox_size": self.bounding_box_size,
            "bbox_area": new_bbox_area,
            "fill_ratio": self.fill_ratio,
            "collision": has_collision,
            "collision_area": collision_area
        }
        
        return observation, reward, terminated, truncated, info

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Mevcut state'i observation dictionary olarak döndürür.
        
        Returns:
            Dict containing:
                - tree_features: (max_trees, 10) her ağacın özellikleri
                - pairwise_features: (max_trees, 10) pair rect özellikleri
                - scalars: (5,) global özellikler
                - mask: (max_trees,) hangi slotlar dolu
        """
        n = len(self.placed_trees)
        
        # Tree features: (max_trees, 10)
        tree_features = np.zeros((self.max_trees, 10), dtype=np.float32)
        for i, tree in enumerate(self.placed_trees):
            tree_features[i] = self._get_tree_features(tree)
        
        # Pairwise features: (max_trees, 10)
        pairwise_features = np.zeros((self.max_trees, 10), dtype=np.float32)
        if n >= 2:
            pairwise = self._get_pairwise_features()
            pairwise_features[:n] = pairwise
        
        # Scalars: (5,)
        # Normalize edilmiş değerler
        scalars = np.array([
            n / self.max_trees,                    # n_placed (normalized)
            self.bounding_box_size / 1e16,         # bbox_size (normalized by scale)
            self.fill_ratio,                       # fill_ratio [0-1]
            (self.n_trees - n) / self.n_trees,     # remaining_ratio [0-1]
            self.total_tree_area / 1e30            # total_area (normalized)
        ], dtype=np.float32)
        
        # Mask: (max_trees,)
        mask = np.zeros(self.max_trees, dtype=np.float32)
        mask[:n] = 1.0
        
        # Occupancy Grid: (grid_size, grid_size)
        occupancy_grid = self._get_occupancy_grid()
        
        return {
            "tree_features": tree_features,
            "pairwise_features": pairwise_features,
            "scalars": scalars,
            "mask": mask,
            "occupancy_grid": occupancy_grid
        }

    # ==================== HELPER METHODS ====================
    
    def _get_occupancy_grid(self) -> np.ndarray:
        """
        256x256 occupancy grid oluştur - Her hücre doluluk ORANI (0-1).
        Yüksek çözünürlükte rasterize edip downsample yaparak fill ratio hesaplar.
        Grid her zaman [-10, 10] x [-10, 10] alanını kapsar.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        if not self.placed_trees:
            return grid
        
        try:
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
            
            # SABİT: Action space sınırları [-10, 10] (scale_factor ile çarpılmış)
            norm = float(scale_factor)
            minx = -10.0 * norm
            miny = -10.0 * norm
            maxx = 10.0 * norm
            maxy = 10.0 * norm
            
            # Yüksek çözünürlükte rasterize (4x) - fill ratio için
            high_res = self.grid_size * 4  # 1024x1024
            transform = from_bounds(minx, miny, maxx, maxy, high_res, high_res)
            
            shapes = [(tree.polygon, 1.0) for tree in self.placed_trees]
            
            # Yüksek çözünürlükte binary rasterize
            high_res_grid = rasterize(
                shapes,
                out_shape=(high_res, high_res),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
                all_touched=True
            )
            
            # Downsample: 4x4 blokların ortalaması = fill ratio
            # Reshape to (256, 4, 256, 4) then mean over axes (1, 3)
            grid = high_res_grid.reshape(
                self.grid_size, 4, self.grid_size, 4
            ).mean(axis=(1, 3))
            
            # Y ekseni ters çevir + contiguous memory
            grid = np.flipud(grid).copy()
            
        except ImportError:
            pass
        
        return grid
    
    def _check_collision(self, new_tree: ChristmasTree) -> float:
        """
        Yeni ağacın mevcut ağaçlarla çarpışma alanını hesaplar.
        
        Args:
            new_tree: Kontrol edilecek yeni ağaç
            
        Returns:
            float: Toplam çarpışma alanı (0 = çarpışma yok)
        """
        if not self.placed_trees:
            return 0.0
        
        if self._strtree is None:
            self._update_strtree()
        
        # STRtree ile potansiyel çarpışanları bul
        candidate_indices = self._strtree.query(new_tree.polygon)
        
        total_intersection_area = 0.0
        
        for idx in candidate_indices:
            existing_tree = self.placed_trees[idx]
            # Gerçek çarpışma: intersects AND NOT touches
            if new_tree.polygon.intersects(existing_tree.polygon):
                if not new_tree.polygon.touches(existing_tree.polygon):
                    # Çarpışma alanını hesapla
                    intersection = new_tree.polygon.intersection(existing_tree.polygon)
                    total_intersection_area += intersection.area
        
        return total_intersection_area

    def _calculate_reward(self, old_bbox_area: float, new_bbox_area: float,
                         collision_area: float) -> float:
        """
        Reward fonksiyonu - Tamamen ALAN bazlı.
        
        Tüm metrikler alan oranları olarak hesaplanır.
        """
        reward = 0.0
        norm_sq = float(scale_factor) ** 2
        
        # Yeni ağacın alanı (referans)
        if self.placed_trees:
            new_tree_area = self.placed_trees[-1].polygon.area
        else:
            new_tree_area = norm_sq * 0.01  # fallback
        
        # ==================== COLLISION PENALTY ====================
        # collision_area / new_tree_area oranı → ne kadar overlap var?
        if collision_area > 0:
            overlap_ratio = collision_area / new_tree_area
            # 0.1 overlap = -2, 1.0 overlap (tam üst üste) = -20
            collision_penalty = min(overlap_ratio * 20.0, 20.0)
            reward -= collision_penalty
            return reward  # Collision varsa diğerlerini hesaplama
        
        # ==================== TREE PLACEMENT BONUS ====================
        # Başarılı ağaç yerleştirme (collision yok): +2 puan
        reward += 2.0
        
        # ==================== BBOX EXPANSION PENALTY ====================
        # (new_bbox - old_bbox) / new_tree_area → ağaç alanına göre ne kadar genişledi?
        if old_bbox_area > 0:
            expansion = new_bbox_area - old_bbox_area
            if expansion > 0:
                expansion_ratio = expansion / new_tree_area
                # 1x ağaç alanı kadar genişleme = -1
                expansion_penalty = min(expansion_ratio * 1.0, 5.0)
                reward -= expansion_penalty
        else:
            reward += 0.5  # İlk ağaç bonusu
        
        # ==================== FILL RATIO BONUS ====================
        # fill_ratio = total_tree_area / bbox_area
        # Yüksek fill ratio = kompakt yerleşim = iyi
        reward += self.fill_ratio * 2.0
        
        # ==================== EPISODE END BONUS ====================
        if self.current_index >= self.n_trees:
            # final_score ∝ 1/bbox_size, log scale
            bbox_size = np.sqrt(new_bbox_area) / float(scale_factor)
            if bbox_size > 0:
                final_bonus = -np.log(bbox_size + 1e-6)
                reward += final_bonus
        
        return reward

    def _update_strtree(self):
        """STRtree'yi güncelleyerek yeniden oluşturur."""
        if self.placed_trees:
            polygons = [tree.polygon for tree in self.placed_trees]
            self._strtree = STRtree(polygons)
        else:
            self._strtree = None

    def _get_tree_features(self, tree: ChristmasTree) -> np.ndarray:
        """
        Tek bir ağaç için feature vektörü çıkarır.
        
        Returns:
            (10,) array: [x, y, angle, minx, miny, maxx, maxy, width, height, area]
        """
        bounds = tree.polygon.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Normalize by scale_factor for network input
        norm = float(scale_factor)
        
        return np.array([
            float(tree.center_x),           # x (original, not scaled)
            float(tree.center_y),           # y
            float(tree.angle) / 360.0,      # angle (normalized 0-1)
            bounds[0] / norm,               # minx (normalized)
            bounds[1] / norm,               # miny
            bounds[2] / norm,               # maxx
            bounds[3] / norm,               # maxy
            width / norm,                   # width
            height / norm,                  # height
            tree.polygon.area / (norm ** 2) # area (normalized)
        ], dtype=np.float32)

    def _get_pairwise_features(self) -> np.ndarray:
        """
        Her ağaç için en yakın komşu ile pairwise feature'lar.
        
        Returns:
            (n_trees, 10) array
        """
        n = len(self.placed_trees)
        if n < 2:
            return np.zeros((n, 10), dtype=np.float32)
        
        neighbors = find_nearest_neighbours(self.placed_trees)
        features = np.zeros((n, 10), dtype=np.float32)
        
        norm = float(scale_factor)
        
        for i, tree in enumerate(self.placed_trees):
            neighbor_idx = neighbors.get(i, 0)
            neighbor = self.placed_trees[neighbor_idx]
            
            # Mesafe
            distance = tree.polygon.distance(neighbor.polygon)
            
            # Pair union bounds
            pair_union = unary_union([tree.polygon, neighbor.polygon])
            pair_bounds = pair_union.bounds
            pair_width = pair_bounds[2] - pair_bounds[0]
            pair_height = pair_bounds[3] - pair_bounds[1]
            
            # Relative position
            dx = float(neighbor.center_x - tree.center_x)
            dy = float(neighbor.center_y - tree.center_y)
            
            features[i] = np.array([
                distance / norm,                    # distance
                dx,                                 # relative x
                dy,                                 # relative y
                pair_bounds[0] / norm,              # pair minx
                pair_bounds[1] / norm,              # pair miny
                pair_bounds[2] / norm,              # pair maxx
                pair_bounds[3] / norm,              # pair maxy
                pair_width / norm,                  # pair width
                pair_height / norm,                 # pair height
                neighbor_idx / self.max_trees       # neighbor index (normalized)
            ], dtype=np.float32)
        
        return features

    def render(self):
        """Matplotlib ile görselleştirme (opsiyonel)."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for tree in self.placed_trees:
            x, y = tree.polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, color='green')
            ax.plot(x, y, color='darkgreen', linewidth=1)
        
        # Bounding box
        if self.placed_trees:
            union = unary_union([t.polygon for t in self.placed_trees])
            bounds = union.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            side = max(width, height)
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (cx - side/2, cy - side/2), side, side,
                fill=False, edgecolor='red', linewidth=2, linestyle='--'
            )
            ax.add_patch(rect)
        
        ax.set_aspect('equal')
        ax.set_title(f'Trees: {len(self.placed_trees)}, BBox: {self.bounding_box_size:.2e}')
        plt.show()














