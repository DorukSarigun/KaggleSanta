from decimal import Decimal
from typing import List, Dict, Tuple, Optional, Any
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
        # pairwise_features: Her ağaç için k=3 komşu ile rect özellikleri = 30 dim (10 * 3)
        # scalars: Global özellikler (bbox_size, fill_ratio, n_placed, ...) = 5 dim
        # free_space_features: Boş alan bilgisi = 4 dim [center_x, center_y, area_ratio, n_regions]
        # collision_features: Çarpışma bilgisi = 6 dim [last_collision_area, collision_risk, avg_min_dist,
        #                     min_clearance, crowding_factor, collision_history_ratio]
        self.k_neighbors = 3  # Multi-neighbor için k değeri
        self.collision_history = []  # Son N placement için collision durumu
        self.collision_history_size = 10  # Track last 10 placements

        self.observation_space = spaces.Dict({
            "tree_features": spaces.Box(-np.inf, np.inf, shape=(max_trees, 10), dtype=np.float32),
            "pairwise_features": spaces.Box(-np.inf, np.inf, shape=(max_trees, 10 * self.k_neighbors), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "free_space_features": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "collision_features": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
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
        self.collision_history = []
        self.last_collision_area = 0.0
        self.collision_count = 0  # Track total collisions for episode
        
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

        # Track collision history
        self.last_collision_area = collision_area
        self.collision_history.append(1.0 if has_collision else 0.0)
        if len(self.collision_history) > self.collision_history_size:
            self.collision_history.pop(0)

        if not has_collision:
            # Geçerli yerleştirme
            self.placed_trees.append(new_tree)
            self.current_index += 1
            self._update_strtree()
        else:
            # Collision oldu - sayacı artır
            self.collision_count += 1
        
        # Yeni bbox ALANI
        new_bbox_area = self.bounding_box_size ** 2 if self.bounding_box_size > 0 else 0
        
        # Reward hesapla (alan bazlı)
        reward = self._calculate_reward(old_bbox_area, new_bbox_area, collision_area)
        
        # Episode bitti mi? (3 collision VEYA tüm ağaçlar yerleşti)
        max_collisions = 3
        terminated = self.current_index >= self.n_trees or self.collision_count >= max_collisions
        truncated = False
        
        # Observation
        observation = self.get_observation()
        
        # Info (bbox_size normalized for display)
        norm = float(scale_factor)
        info = {
            "n_placed": len(self.placed_trees),
            "n_remaining": self.n_trees - self.current_index,
            "bbox_size": self.bounding_box_size / norm,  # Normalized to [0, 10]
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
                - pairwise_features: (max_trees, 30) k=3 komşu için rect özellikleri
                - scalars: (5,) global özellikler
                - free_space_features: (4,) boş alan bilgisi
                - mask: (max_trees,) hangi slotlar dolu
                - occupancy_grid: (256, 256) doluluk grid'i
        """
        n = len(self.placed_trees)

        # Tree features: (max_trees, 10)
        tree_features = np.zeros((self.max_trees, 10), dtype=np.float32)
        for i, tree in enumerate(self.placed_trees):
            tree_features[i] = self._get_tree_features(tree)

        # Pairwise features: (max_trees, 30) - k=3 komşu
        pairwise_features = np.zeros((self.max_trees, 10 * self.k_neighbors), dtype=np.float32)
        if n >= 2:
            pairwise = self._get_multi_neighbor_pairwise_features()
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

        # Free Space Features: (4,) - BOŞ ALAN BİLGİSİ (KRİTİK İYİLEŞTİRME)
        free_space_features = self._get_free_space_features()

        # Collision Features: (6,) - ÇARPIŞMA BİLGİSİ (YENİ!)
        collision_features = self._get_collision_features()

        # Mask: (max_trees,)
        mask = np.zeros(self.max_trees, dtype=np.float32)
        mask[:n] = 1.0

        # Occupancy Grid: (grid_size, grid_size)
        occupancy_grid = self._get_occupancy_grid()

        return {
            "tree_features": tree_features,
            "pairwise_features": pairwise_features,
            "scalars": scalars,
            "free_space_features": free_space_features,
            "collision_features": collision_features,
            "mask": mask,
            "occupancy_grid": occupancy_grid
        }

    # ==================== HELPER METHODS ====================
    
    def _get_occupancy_grid(self, size: int = None) -> np.ndarray:
        """
        NxN occupancy grid oluştur - Her hücre doluluk ORANI (0-1).
        Yüksek çözünürlükte rasterize edip downsample yaparak fill ratio hesaplar.
        Grid her zaman [-10, 10] x [-10, 10] alanını kapsar.
        
        Args:
            size: Grid size (default: self.grid_size = 256)
        """
        if size is None:
            size = self.grid_size
            
        grid = np.zeros((size, size), dtype=np.float32)
        
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
            high_res = size * 4
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
            grid = high_res_grid.reshape(
                size, 4, size, 4
            ).mean(axis=(1, 3))
            
            # Y ekseni ters çevir + contiguous memory
            grid = np.flipud(grid).copy()
            
        except ImportError:
            pass
        
        return grid
    
    def _get_collision_features(self) -> np.ndarray:
        """
        Collision-aware features to help agent learn collision avoidance.

        Returns:
            6-dim array: [last_collision_area, collision_risk, avg_min_dist,
                         min_clearance, crowding_factor, collision_history_ratio]
        """
        norm = float(scale_factor)
        n = len(self.placed_trees)

        # 1. Last collision area (normalized)
        last_collision = min(self.last_collision_area / (norm ** 2), 1.0)

        # 2. Collision risk: ratio of trees with close neighbors
        collision_risk = 0.0
        avg_min_dist = 1.0
        min_clearance = 1.0

        if n >= 2:
            min_distances = []
            for tree in self.placed_trees:
                min_dist = float('inf')
                for other in self.placed_trees:
                    if tree is not other:
                        dist = tree.polygon.distance(other.polygon) / norm
                        min_dist = min(min_dist, dist)
                min_distances.append(min_dist)

            # Average minimum distance
            avg_min_dist = np.mean(min_distances)

            # Minimum clearance across all trees
            min_clearance = np.min(min_distances)

            # Collision risk: how many trees are very close (< 0.3 distance)
            close_threshold = 0.3
            close_count = sum(1 for d in min_distances if d < close_threshold)
            collision_risk = close_count / n

        # 3. Crowding factor: based on fill ratio and tree density
        crowding_factor = self.fill_ratio if n > 0 else 0.0

        # 4. Collision history ratio
        collision_history_ratio = 0.0
        if len(self.collision_history) > 0:
            collision_history_ratio = sum(self.collision_history) / len(self.collision_history)

        return np.array([
            last_collision,           # 0-1, normalized collision area
            collision_risk,           # 0-1, ratio of risky placements
            min(avg_min_dist, 1.0),  # 0-1, average minimum distance
            min(min_clearance, 1.0), # 0-1, minimum clearance
            crowding_factor,         # 0-1, current crowding
            collision_history_ratio  # 0-1, historical collision rate
        ], dtype=np.float32)

    def _get_free_space_features(self) -> np.ndarray:
        """
        Boş alan analizi - en büyük boş bölgeyi bul.
        
        Returns:
            4-dim array: [center_x, center_y, area_ratio, n_regions]
        """
        try:
            from scipy.ndimage import label
        except ImportError:
            return np.zeros(4, dtype=np.float32)
        
        if not self.placed_trees:
            # Tüm alan boş
            return np.array([0.5, 0.5, 1.0, 1], dtype=np.float32)
        
        # Get occupancy grid
        grid = self._get_occupancy_grid()
        
        # Boş hücreler (fill ratio < 0.1)
        free_mask = (grid < 0.1).astype(np.uint8)
        
        # Connected components
        labeled, n_regions = label(free_mask)
        
        if n_regions == 0:
            return np.array([0.5, 0.5, 0.0, 0], dtype=np.float32)
        
        # En büyük bölgeyi bul
        largest_area = 0
        largest_idx = 1
        for i in range(1, n_regions + 1):
            area = np.sum(labeled == i)
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        # En büyük bölgenin merkezi
        coords = np.argwhere(labeled == largest_idx)
        if len(coords) > 0:
            center_y, center_x = coords.mean(axis=0)
        else:
            center_x, center_y = self.grid_size / 2, self.grid_size / 2
        
        # Normalize to [0, 1]
        center_x = center_x / self.grid_size
        center_y = center_y / self.grid_size
        area_ratio = largest_area / (self.grid_size ** 2)
        
        return np.array([center_x, center_y, area_ratio, min(n_regions / 10.0, 1.0)], dtype=np.float32)
    
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
        TUNED Reward Function - Exploration Bonus + Softer Collision.

        Changes:
        1. Softer collision penalty: [-8, -2] instead of [-20, -5]
        2. Exploration bonus: reward for placing far from last tree
        3. Placement bonus remains
        4. Kaggle improvement bonus
        """
        reward = 0.0
        norm = float(scale_factor)
        norm_sq = norm ** 2

        n_placed = len(self.placed_trees)

        # ==================== 1. COLLISION PENALTY (REMAINING TREES) ====================
        if collision_area > 0:
            # Penalty = negative of remaining trees
            # Early collision (40 left) = -40, Late collision (5 left) = -5
            remaining_trees = self.n_trees - self.current_index
            collision_penalty = -float(remaining_trees)
            return collision_penalty

        # ==================== 2. PLACEMENT BONUS ====================
        reward = 3.0  # Increased placement bonus

        # ==================== 3. EXPLORATION BONUS ====================
        # Reward for placing tree far from the last tree
        if n_placed >= 2:
            last_tree = self.placed_trees[-1]
            prev_tree = self.placed_trees[-2]
            
            # Distance between last two trees
            dx = (float(last_tree.center_x) - float(prev_tree.center_x)) / norm
            dy = (float(last_tree.center_y) - float(prev_tree.center_y)) / norm
            dist = np.sqrt(dx**2 + dy**2)
            
            # Exploration bonus: reward for spreading out
            # dist in [0, ~5], normalize and give bonus
            exploration_bonus = min(dist / 2.0, 1.0) * 0.5  # Up to +0.5
            reward += exploration_bonus

        # ==================== 4. KAGGLE IMPROVEMENT ====================
        if old_bbox_area > 0 and new_bbox_area > 0 and n_placed > 0:
            old_bbox_size = np.sqrt(old_bbox_area) / norm
            old_kaggle = (old_bbox_size ** 2) / max(n_placed - 1, 1) if n_placed > 1 else 0

            new_bbox_size = np.sqrt(new_bbox_area) / norm
            new_kaggle = (new_bbox_size ** 2) / n_placed

            kaggle_improvement = old_kaggle - new_kaggle
            reward += kaggle_improvement * 10.0

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

    def _find_k_nearest_neighbours(self, k: int = 3) -> Dict[int, List[int]]:
        """
        Her ağaç için k en yakın komşunun index'lerini döndürür.

        Returns:
            Dict[int, List[int]]: {tree_idx: [neighbor_idx_1, neighbor_idx_2, ...]}
        """
        n = len(self.placed_trees)
        if n < 2:
            return {}

        neighbors = {}
        for i, tree in enumerate(self.placed_trees):
            distances = []
            for j, other in enumerate(self.placed_trees):
                if i != j:
                    dist = tree.polygon.distance(other.polygon)
                    distances.append((j, dist))

            # En yakın k komşuyu seç
            distances.sort(key=lambda x: x[1])
            neighbors[i] = [idx for idx, _ in distances[:min(k, len(distances))]]

        return neighbors

    def _get_multi_neighbor_pairwise_features(self) -> np.ndarray:
        """
        Her ağaç için k=3 en yakın komşu ile pairwise feature'lar.
        Daha zengin komşuluk bilgisi sağlar.

        Returns:
            (n_trees, 30) array - her ağaç için 3 komşu * 10 feature
        """
        n = len(self.placed_trees)
        k = self.k_neighbors

        if n < 2:
            return np.zeros((n, 10 * k), dtype=np.float32)

        neighbors = self._find_k_nearest_neighbours(k)
        features = np.zeros((n, 10 * k), dtype=np.float32)

        norm = float(scale_factor)

        for i, tree in enumerate(self.placed_trees):
            neighbor_indices = neighbors.get(i, [])

            for neighbor_rank, neighbor_idx in enumerate(neighbor_indices):
                if neighbor_rank >= k:
                    break

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

                # Her komşu için 10 feature (offset = neighbor_rank * 10)
                offset = neighbor_rank * 10
                features[i, offset:offset+10] = np.array([
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

    def _get_pairwise_features(self) -> np.ndarray:
        """
        Her ağaç için en yakın komşu ile pairwise feature'lar.
        (Geriye uyumluluk için korundu)

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














