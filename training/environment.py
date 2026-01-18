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
        
        # Action Space: (x, y, rotation) continuous
        self.action_space = spaces.Box(
            low=np.array([-20.0, -20.0, 0.0], dtype=np.float32),
            high=np.array([20.0, 20.0, 360.0], dtype=np.float32)
        )
        
        # Observation Space
        # tree_features: Her ağaç için (x, y, angle, minx, miny, maxx, maxy, ...) = 10 dim
        # pairwise_features: Her ağaç için komşu ile rect özellikleri = 10 dim
        # scalars: Global özellikler (bbox_size, fill_ratio, n_placed, ...) = 5 dim
        self.observation_space = spaces.Dict({
            "tree_features": spaces.Box(-np.inf, np.inf, shape=(max_trees, 10), dtype=np.float32),
            "pairwise_features": spaces.Box(-np.inf, np.inf, shape=(max_trees, 10), dtype=np.float32),
            "scalars": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(max_trees,), dtype=np.float32)
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
        
        return {
            "tree_features": tree_features,
            "pairwise_features": pairwise_features,
            "scalars": scalars,
            "mask": mask
        }

    # ==================== HELPER METHODS ====================
    
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
        Reward fonksiyonu (alan bazlı) - İyileştirilmiş versiyon.

        Args:
            old_bbox_area: Önceki bounding box ALANI
            new_bbox_area: Yeni bounding box ALANI
            collision_area: Çarpışma alanı (0 = çarpışma yok)

        Returns:
            Reward değeri
        """
        reward = 0.0

        # Çarpışma varsa alana orantılı ceza
        if collision_area > 0:
            # FIX: Sabit referans alan kullan (tek ağaç alanı ~0.5 * 1e15^2)
            single_tree_area = 5e29  # Yaklaşık bir ağaç alanı
            avg_tree_area = self.total_tree_area / max(1, len(self.placed_trees)) if self.placed_trees else single_tree_area
            normalized_collision = collision_area / avg_tree_area
            # Cap collision penalty at -5.0 for stability
            reward -= min(normalized_collision * 10.0, 5.0)
            return reward  # Çarpışmada erken dön

        # Bbox ALAN büyümesine göre ceza/ödül
        if old_bbox_area == 0:
            # İlk ağaç
            reward += 1.0
        else:
            area_expansion = new_bbox_area - old_bbox_area
            if area_expansion <= 0:
                # Kutu büyümedi veya küçüldü → bonus
                reward += 2.0
            else:
                # FIX: Kutu büyüdü → expansion'a orantılı ceza (max -3.0)
                normalized_expansion = area_expansion / old_bbox_area
                reward -= min(normalized_expansion * 5.0, 3.0)

        # Fill ratio bonusu (0 to 1.0 range)
        reward += self.fill_ratio * 1.0

        # Compactness bonusu: pair rotated rectangles union alanı
        # Küçük union = kompakt yerleşim = yüksek bonus
        if len(self.placed_trees) >= 2:
            try:
                rects_union = self.union_of_min_rotated_rectangles
                if rects_union is not None and not rects_union.is_empty:
                    rects_union_area = rects_union.area

                    if new_bbox_area > 0:
                        # Compactness ratio: küçük = iyi (ağaçlar birbirine yakın)
                        compactness = rects_union_area / new_bbox_area
                        # Küçük compactness → yüksek bonus
                        compactness_bonus = max(0, 1.0 - compactness) * 1.0
                        reward += compactness_bonus
            except Exception:
                # Shapely topology hatası - bu bonusu atla
                pass

        # FIX: Episode sonu bonusu - normalized ve dengeli ölçek
        if self.current_index >= self.n_trees:
            # Yarışma skoru: 1 / bbox_size (küçük bbox = yüksek skor)
            # Logaritmik ölçeğe çevir ki step rewards ile uyumlu olsun
            bbox_size_normalized = np.sqrt(new_bbox_area) / float(scale_factor)
            if bbox_size_normalized > 0:
                # Log reward: küçük bbox = pozitif, büyük bbox = negatif
                # Tipik range: -5 to +5 (step rewards ile uyumlu)
                final_bonus = -np.log(bbox_size_normalized + 1e-6)
                reward += final_bonus * 1.0

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














