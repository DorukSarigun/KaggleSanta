# 🧠 Optimizasyon Yöntemleri Detaylı Kılavuzu

> **Kapsam:** Reinforcement Learning, Simulated Annealing, Genetik Algoritma
> **Uygulama Alanı:** Santa 2025 - Noel Ağacı Paketleme Problemi

---

## 📑 İçindekiler

1. [Simulated Annealing (Tavlama Benzetimi)](#1-simulated-annealing-tavlama-benzetimi)
2. [Genetik Algoritma](#2-genetik-algoritma)
3. [Reinforcement Learning (Pekiştirmeli Öğrenme)](#3-reinforcement-learning-pekiştirmeli-öğrenme)
4. [Karşılaştırma ve Seçim Kriterleri](#4-karşılaştırma-ve-seçim-kriterleri)
5. [Santa 2025 İçin Uygulama Önerileri](#5-santa-2025-i̇çin-uygulama-önerileri)

---

# 1. Simulated Annealing (Tavlama Benzetimi)

## 1.1 Temel Kavram

Simulated Annealing, metallerin tavlama sürecinden esinlenen bir olasılıksal optimizasyon algoritmasıdır. Metallerde, yüksek sıcaklıkta atomlar serbestçe hareket eder; sıcaklık yavaşça düşürüldüğünde atomlar minimum enerji durumuna (kristal yapı) yerleşir.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FİZİKSEL ANALOJİ                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   YÜKSEK SICAKLIK          →         DÜŞÜK SICAKLIK             │
│   ~~~~~~~~~~~~~~~~                   ~~~~~~~~~~~~~~~~            │
│   • Atomlar rastgele                 • Atomlar düzenli          │
│   • Yüksek enerji                    • Düşük enerji             │
│   • Kaotik hareket                   • Kristal yapı             │
│                                                                 │
│   OPTİMİZASYONDA:                                               │
│   • Kötü çözümler kabul              • Sadece iyi çözümler      │
│   • Geniş keşif                      • İnce ayar                │
│   • Lokal min. kaçış                 • Yakınsama                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1.2 Matematiksel Formülasyon

### Kabul Olasılığı (Metropolis Kriteri)

```
P(kabul) = {
    1                           eğer ΔE < 0 (daha iyi çözüm)
    exp(-ΔE / T)                eğer ΔE ≥ 0 (daha kötü çözüm)
}

Burada:
- ΔE = E_yeni - E_eski (enerji/maliyet farkı)
- T = Mevcut sıcaklık
```

### Soğutma Programları

| Program | Formül | Özellik |
|---------|--------|---------|
| Geometrik | T(k+1) = α × T(k), α ∈ [0.8, 0.99] | En yaygın |
| Lineer | T(k+1) = T(k) - β | Hızlı ama riskli |
| Logaritmik | T(k) = T₀ / log(k+1) | Teorik garanti |
| Adaptif | T(k+1) = T(k) × f(kabul_oranı) | Akıllı |

## 1.3 Algoritma Pseudocode

```
ALGORITHM SimulatedAnnealing(initial_solution, T_initial, T_final, cooling_rate)
    
    current ← initial_solution
    best ← current
    T ← T_initial
    
    WHILE T > T_final DO
        FOR i ← 1 TO iterations_per_temperature DO
            
            // 1. Komşu çözüm üret
            neighbor ← generate_neighbor(current)
            
            // 2. Enerji farkını hesapla
            ΔE ← energy(neighbor) - energy(current)
            
            // 3. Kabul kararı
            IF ΔE < 0 THEN
                current ← neighbor                    // Her zaman kabul
            ELSE IF random() < exp(-ΔE / T) THEN
                current ← neighbor                    // Olasılıkla kabul
            END IF
            
            // 4. En iyi çözümü güncelle
            IF energy(current) < energy(best) THEN
                best ← current
            END IF
            
        END FOR
        
        // 5. Sıcaklığı düşür
        T ← T × cooling_rate
        
    END WHILE
    
    RETURN best
```

## 1.4 Python Implementasyonu

```python
import math
import random
import copy
from typing import Callable, TypeVar, Tuple

T = TypeVar('T')  # Generic solution type

class SimulatedAnnealing:
    """
    Genel amaçlı Simulated Annealing optimizasyonu.
    
    Attributes:
        energy_func: Çözümün maliyetini/enerjisini hesaplayan fonksiyon
        neighbor_func: Komşu çözüm üreten fonksiyon
        T_initial: Başlangıç sıcaklığı
        T_final: Bitiş sıcaklığı
        cooling_rate: Soğutma katsayısı (0 < α < 1)
        iterations_per_temp: Her sıcaklıkta yapılacak iterasyon sayısı
    """
    
    def __init__(
        self,
        energy_func: Callable[[T], float],
        neighbor_func: Callable[[T], T],
        T_initial: float = 1000.0,
        T_final: float = 0.001,
        cooling_rate: float = 0.995,
        iterations_per_temp: int = 100
    ):
        self.energy_func = energy_func
        self.neighbor_func = neighbor_func
        self.T_initial = T_initial
        self.T_final = T_final
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        
        # İstatistikler
        self.history = []
        self.acceptance_history = []
    
    def acceptance_probability(self, delta_e: float, temperature: float) -> float:
        """Metropolis kabul olasılığını hesapla."""
        if delta_e < 0:
            return 1.0
        return math.exp(-delta_e / temperature)
    
    def optimize(self, initial_solution: T, verbose: bool = False) -> Tuple[T, float]:
        """
        Optimizasyonu çalıştır.
        
        Args:
            initial_solution: Başlangıç çözümü
            verbose: İlerleme bilgisi yazdır
            
        Returns:
            (en_iyi_çözüm, en_iyi_enerji)
        """
        current = copy.deepcopy(initial_solution)
        best = copy.deepcopy(current)
        
        current_energy = self.energy_func(current)
        best_energy = current_energy
        
        temperature = self.T_initial
        iteration = 0
        
        while temperature > self.T_final:
            accepted = 0
            
            for _ in range(self.iterations_per_temp):
                # Komşu üret
                neighbor = self.neighbor_func(copy.deepcopy(current))
                neighbor_energy = self.energy_func(neighbor)
                
                # Enerji farkı
                delta_e = neighbor_energy - current_energy
                
                # Kabul kararı
                if random.random() < self.acceptance_probability(delta_e, temperature):
                    current = neighbor
                    current_energy = neighbor_energy
                    accepted += 1
                    
                    # En iyi güncelle
                    if current_energy < best_energy:
                        best = copy.deepcopy(current)
                        best_energy = current_energy
                
                iteration += 1
            
            # İstatistik kaydet
            acceptance_rate = accepted / self.iterations_per_temp
            self.history.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_energy': current_energy,
                'best_energy': best_energy,
                'acceptance_rate': acceptance_rate
            })
            
            if verbose and iteration % 10000 == 0:
                print(f"Iter {iteration:,} | T={temperature:.4f} | "
                      f"Current={current_energy:.6f} | Best={best_energy:.6f} | "
                      f"Accept={acceptance_rate:.2%}")
            
            # Soğut
            temperature *= self.cooling_rate
        
        return best, best_energy
    
    def adaptive_optimize(self, initial_solution: T, target_acceptance: float = 0.4) -> Tuple[T, float]:
        """
        Adaptif soğutma ile optimizasyon.
        Kabul oranına göre sıcaklığı ayarlar.
        """
        current = copy.deepcopy(initial_solution)
        best = copy.deepcopy(current)
        
        current_energy = self.energy_func(current)
        best_energy = current_energy
        
        temperature = self.T_initial
        
        while temperature > self.T_final:
            accepted = 0
            
            for _ in range(self.iterations_per_temp):
                neighbor = self.neighbor_func(copy.deepcopy(current))
                neighbor_energy = self.energy_func(neighbor)
                delta_e = neighbor_energy - current_energy
                
                if random.random() < self.acceptance_probability(delta_e, temperature):
                    current = neighbor
                    current_energy = neighbor_energy
                    accepted += 1
                    
                    if current_energy < best_energy:
                        best = copy.deepcopy(current)
                        best_energy = current_energy
            
            # Adaptif soğutma
            acceptance_rate = accepted / self.iterations_per_temp
            if acceptance_rate > target_acceptance + 0.1:
                # Çok fazla kabul - daha hızlı soğut
                temperature *= 0.95
            elif acceptance_rate < target_acceptance - 0.1:
                # Çok az kabul - daha yavaş soğut
                temperature *= 0.999
            else:
                temperature *= self.cooling_rate
        
        return best, best_energy


# ═══════════════════════════════════════════════════════════════
# SANTA 2025 İÇİN UYGULAMA
# ═══════════════════════════════════════════════════════════════

class TreePackingSA:
    """Santa 2025 için Simulated Annealing uygulaması."""
    
    def __init__(self, trees):
        self.trees = trees
        self.n_trees = len(trees)
    
    def energy(self, solution):
        """
        Çözümün enerjisi = Bounding box alanı + Çarpışma cezası
        
        solution: [(x1, y1, θ1), (x2, y2, θ2), ...]
        """
        # Ağaçları güncelle
        for i, (x, y, angle) in enumerate(solution):
            self.trees[i].update_position(x, y, angle)
        
        # Çarpışma kontrolü
        collision_count = self.count_collisions()
        collision_penalty = collision_count * 1000  # Büyük ceza
        
        # Bounding box
        bbox = self.calculate_bounding_box()
        
        return bbox + collision_penalty
    
    def neighbor(self, solution):
        """Rastgele bir ağacı küçük miktarda hareket ettir."""
        new_solution = [list(s) for s in solution]
        
        # Rastgele ağaç seç
        idx = random.randint(0, self.n_trees - 1)
        
        # Hareket tipi seç
        move_type = random.choice(['translate', 'rotate', 'both'])
        
        if move_type in ['translate', 'both']:
            # Küçük öteleme (Gaussian dağılım)
            new_solution[idx][0] += random.gauss(0, 0.05)  # x
            new_solution[idx][1] += random.gauss(0, 0.05)  # y
        
        if move_type in ['rotate', 'both']:
            # Küçük dönme
            new_solution[idx][2] += random.gauss(0, 5)  # derece
            new_solution[idx][2] %= 360  # 0-360 arası tut
        
        return [tuple(s) for s in new_solution]
    
    def count_collisions(self):
        """Çarpışan ağaç çifti sayısı."""
        count = 0
        for i in range(self.n_trees):
            for j in range(i + 1, self.n_trees):
                if self.trees[i].polygon.intersects(self.trees[j].polygon):
                    if not self.trees[i].polygon.touches(self.trees[j].polygon):
                        count += 1
        return count
    
    def calculate_bounding_box(self):
        """Minimum kare kenar uzunluğu."""
        from shapely.ops import unary_union
        combined = unary_union([t.polygon for t in self.trees])
        minx, miny, maxx, maxy = combined.bounds
        return max(maxx - minx, maxy - miny)
    
    def optimize(self, iterations=100000):
        """SA optimizasyonunu çalıştır."""
        # Başlangıç çözümü
        initial = [(float(t.center_x), float(t.center_y), float(t.angle)) 
                   for t in self.trees]
        
        sa = SimulatedAnnealing(
            energy_func=self.energy,
            neighbor_func=self.neighbor,
            T_initial=100.0,
            T_final=0.0001,
            cooling_rate=0.99995,
            iterations_per_temp=50
        )
        
        best_solution, best_energy = sa.optimize(initial, verbose=True)
        return best_solution, best_energy
```

## 1.5 Parametre Ayarlama Rehberi

```
┌─────────────────────────────────────────────────────────────────┐
│                   PARAMETRE AYARLAMA                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  T_initial (Başlangıç Sıcaklığı):                               │
│  ├── Çok yüksek → Gereksiz keşif, yavaş yakınsama               │
│  ├── Çok düşük → Lokal minimumda takılma                        │
│  └── İdeal: İlk 100 iterasyonda %80 kabul oranı                 │
│                                                                 │
│  T_final (Bitiş Sıcaklığı):                                     │
│  └── Genelde 10⁻⁴ veya 10⁻⁶                                     │
│                                                                 │
│  cooling_rate (Soğutma Oranı):                                  │
│  ├── 0.80 - 0.90 → Hızlı, düşük kalite                          │
│  ├── 0.95 - 0.99 → Orta                                         │
│  └── 0.999 - 0.9999 → Yavaş, yüksek kalite                      │
│                                                                 │
│  iterations_per_temp:                                           │
│  └── Genelde 50-200, problem boyutuna bağlı                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1.6 Avantajlar ve Dezavantajlar

| ✅ Avantajlar | ❌ Dezavantajlar |
|--------------|-----------------|
| Lokal minimumlardan kaçabilir | Parametre ayarı kritik |
| Basit implementasyon | Yakınsama garantisi yok |
| Her problem tipine uygulanabilir | Hesaplama maliyeti yüksek olabilir |
| Paralelize edilebilir | Optimum bulma garantisi yok |
| Anlaşılması kolay | Soğutma programı seçimi zor |

---

# 2. Genetik Algoritma

## 2.1 Temel Kavram

Genetik Algoritma (GA), Darwin'in evrim teorisinden esinlenen bir meta-sezgisel optimizasyon yöntemidir. "En uygun olanın hayatta kalması" ilkesiyle çalışır.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVRİM ANALOJİSİ                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BİYOLOJİ                      OPTİMİZASYON                    │
│   ~~~~~~~~                      ~~~~~~~~~~~~                    │
│   Birey (Organizma)      →      Çözüm Adayı                     │
│   Popülasyon             →      Çözüm Kümesi                    │
│   Kromozom (DNA)         →      Kodlanmış Çözüm                 │
│   Gen                    →      Karar Değişkeni                 │
│   Uygunluk (Fitness)     →      Amaç Fonksiyonu                 │
│   Çaprazlama             →      Çözüm Kombinasyonu              │
│   Mutasyon               →      Rastgele Değişiklik             │
│   Seçilim                →      En İyileri Tutma                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Algoritma Akışı

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│    ┌─────────────────┐                                           │
│    │  BAŞLANGIÇ      │                                           │
│    │  POPÜLASYONU    │◄────────────────────────────────┐         │
│    │  OLUŞTUR        │                                 │         │
│    └────────┬────────┘                                 │         │
│             │                                          │         │
│             ▼                                          │         │
│    ┌─────────────────┐                                 │         │
│    │  FITNESS        │                                 │         │
│    │  HESAPLA        │                                 │         │
│    └────────┬────────┘                                 │         │
│             │                                          │         │
│             ▼                                          │         │
│    ┌─────────────────┐      Hayır                      │         │
│    │  DURDURMA       │─────────────────────────────────┘         │
│    │  KRİTERİ?       │                                           │
│    └────────┬────────┘                                           │
│             │ Evet                                                │
│             ▼                                                    │
│    ┌─────────────────┐                                           │
│    │  EN İYİ         │                                           │
│    │  ÇÖZÜM          │                                           │
│    └─────────────────┘                                           │
│                                                                  │
│    ─────────────── EVRİM DÖNGÜSÜ ───────────────                 │
│                                                                  │
│    ┌─────────────────┐                                           │
│    │  SEÇİLİM        │  Rulet, Turnuva, Elit...                  │
│    └────────┬────────┘                                           │
│             │                                                    │
│             ▼                                                    │
│    ┌─────────────────┐                                           │
│    │  ÇAPRAZLAMA     │  Tek nokta, İki nokta, Uniform...         │
│    └────────┬────────┘                                           │
│             │                                                    │
│             ▼                                                    │
│    ┌─────────────────┐                                           │
│    │  MUTASYON       │  Bit flip, Gaussian, Swap...              │
│    └────────┬────────┘                                           │
│             │                                                    │
│             ▼                                                    │
│    ┌─────────────────┐                                           │
│    │  YENİ           │                                           │
│    │  POPÜLASYON     │                                           │
│    └─────────────────┘                                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 2.3 Genetik Operatörler

### 2.3.1 Seçilim (Selection)

```python
# ═══════════════════════════════════════════════════════════════
# SEÇİLİM YÖNTEMLERİ
# ═══════════════════════════════════════════════════════════════

def roulette_wheel_selection(population, fitness_scores):
    """
    Rulet Tekerleği Seçimi
    
    Fitness'a orantılı seçim olasılığı.
    Yüksek fitness = yüksek seçilme şansı
    """
    # Fitness'ları normalize et (minimizasyon için ters çevir)
    max_fit = max(fitness_scores)
    adjusted = [max_fit - f + 1 for f in fitness_scores]
    total = sum(adjusted)
    probabilities = [f / total for f in adjusted]
    
    # Kümülatif dağılım
    cumulative = []
    cum_sum = 0
    for p in probabilities:
        cum_sum += p
        cumulative.append(cum_sum)
    
    # Rastgele seç
    r = random.random()
    for i, cum in enumerate(cumulative):
        if r <= cum:
            return population[i]
    return population[-1]


def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Turnuva Seçimi
    
    Rastgele k birey seç, en iyisini döndür.
    tournament_size büyükse → seçim baskısı artar
    """
    indices = random.sample(range(len(population)), tournament_size)
    tournament = [(population[i], fitness_scores[i]) for i in indices]
    winner = min(tournament, key=lambda x: x[1])  # Minimizasyon
    return winner[0]


def rank_selection(population, fitness_scores):
    """
    Sıralama Seçimi
    
    Fitness değerine değil, sıralamaya göre seçim.
    Fitness farklarının çok büyük olduğu durumlarda kullanışlı.
    """
    # Sırala (en düşük fitness = en iyi)
    sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1])
    n = len(population)
    
    # Sıralama bazlı olasılık: rank 1 → n puan, rank n → 1 puan
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probabilities = [r / total for r in ranks]
    
    # Seçim
    r = random.random()
    cum = 0
    for i, p in enumerate(probabilities):
        cum += p
        if r <= cum:
            return sorted_pop[i][0]
    return sorted_pop[-1][0]


def elitism(population, fitness_scores, elite_count=2):
    """
    Elitizm
    
    En iyi bireyleri doğrudan sonraki nesle aktar.
    İyi çözümlerin kaybolmasını önler.
    """
    sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1])
    return [p for p, f in sorted_pop[:elite_count]]
```

### 2.3.2 Çaprazlama (Crossover)

```python
# ═══════════════════════════════════════════════════════════════
# ÇAPRAZLAMA YÖNTEMLERİ
# ═══════════════════════════════════════════════════════════════

def single_point_crossover(parent1, parent2):
    """
    Tek Nokta Çaprazlama
    
    Parent1: [A B C | D E F]
    Parent2: [1 2 3 | 4 5 6]
                  ↓
    Child1:  [A B C | 4 5 6]
    Child2:  [1 2 3 | D E F]
    """
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    İki Nokta Çaprazlama
    
    Parent1: [A B | C D E | F G]
    Parent2: [1 2 | 3 4 5 | 6 7]
                    ↓
    Child1:  [A B | 3 4 5 | F G]
    Child2:  [1 2 | C D E | 6 7]
    """
    if len(parent1) < 3:
        return single_point_crossover(parent1, parent2)
    
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


def uniform_crossover(parent1, parent2, prob=0.5):
    """
    Uniform Çaprazlama
    
    Her gen için bağımsız olarak hangi ebeveynden alınacağını seç.
    
    Parent1: [A B C D E F]
    Parent2: [1 2 3 4 5 6]
    Mask:    [0 1 1 0 1 0]
                  ↓
    Child1:  [A 2 3 D 5 F]
    Child2:  [1 B C 4 E 6]
    """
    child1 = []
    child2 = []
    
    for g1, g2 in zip(parent1, parent2):
        if random.random() < prob:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    
    return child1, child2


def blend_crossover(parent1, parent2, alpha=0.5):
    """
    Blend Crossover (BLX-α)
    
    Sürekli değerler için. Ebeveyn değerlerinin ağırlıklı ortalaması
    ve biraz ötesinde değerler üretir.
    """
    child1 = []
    child2 = []
    
    for g1, g2 in zip(parent1, parent2):
        min_g = min(g1, g2)
        max_g = max(g1, g2)
        range_g = max_g - min_g
        
        # Aralığı genişlet
        new_min = min_g - alpha * range_g
        new_max = max_g + alpha * range_g
        
        child1.append(random.uniform(new_min, new_max))
        child2.append(random.uniform(new_min, new_max))
    
    return child1, child2
```

### 2.3.3 Mutasyon (Mutation)

```python
# ═══════════════════════════════════════════════════════════════
# MUTASYON YÖNTEMLERİ
# ═══════════════════════════════════════════════════════════════

def bit_flip_mutation(chromosome, mutation_rate=0.01):
    """
    Bit Flip Mutasyonu (Binary kodlama için)
    
    Her bit için mutation_rate olasılıkla flip yap.
    [0, 1, 0, 0, 1] → [0, 1, 1, 0, 1]
    """
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def gaussian_mutation(chromosome, mutation_rate=0.1, sigma=0.1):
    """
    Gaussian Mutasyon (Sürekli değerler için)
    
    Her gen için mutation_rate olasılıkla Gaussian gürültü ekle.
    """
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] += random.gauss(0, sigma)
    return mutated


def swap_mutation(chromosome, mutation_rate=0.1):
    """
    Swap Mutasyonu (Permütasyon için)
    
    İki rastgele genin yerini değiştir.
    [A, B, C, D, E] → [A, D, C, B, E]
    """
    if random.random() < mutation_rate:
        mutated = chromosome.copy()
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    return chromosome


def scramble_mutation(chromosome, mutation_rate=0.1):
    """
    Scramble Mutasyonu
    
    Rastgele bir alt diziyi karıştır.
    [A, B, C, D, E] → [A, D, B, C, E]
    """
    if random.random() < mutation_rate:
        mutated = chromosome.copy()
        start = random.randint(0, len(mutated) - 2)
        end = random.randint(start + 1, len(mutated))
        subset = mutated[start:end]
        random.shuffle(subset)
        mutated[start:end] = subset
        return mutated
    return chromosome


def creep_mutation(chromosome, mutation_rate=0.1, creep_range=0.1):
    """
    Creep Mutasyonu
    
    Küçük artımlı değişiklikler (fine-tuning için iyi).
    """
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            creep = random.uniform(-creep_range, creep_range)
            mutated[i] += creep
    return mutated
```

## 2.4 Tam Python Implementasyonu

```python
import random
import copy
from typing import List, Callable, Tuple, Any
from dataclasses import dataclass

@dataclass
class GAConfig:
    """Genetik Algoritma konfigürasyonu."""
    population_size: int = 100
    generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_count: int = 2
    tournament_size: int = 3


class GeneticAlgorithm:
    """
    Genel amaçlı Genetik Algoritma implementasyonu.
    """
    
    def __init__(
        self,
        fitness_func: Callable[[List], float],
        create_individual: Callable[[], List],
        crossover_func: Callable[[List, List], Tuple[List, List]],
        mutation_func: Callable[[List], List],
        config: GAConfig = None
    ):
        self.fitness_func = fitness_func
        self.create_individual = create_individual
        self.crossover = crossover_func
        self.mutate = mutation_func
        self.config = config or GAConfig()
        
        # İstatistikler
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': []
        }
    
    def initialize_population(self) -> List[List]:
        """Başlangıç popülasyonunu oluştur."""
        return [self.create_individual() 
                for _ in range(self.config.population_size)]
    
    def evaluate_population(self, population: List[List]) -> List[float]:
        """Tüm bireylerin fitness değerlerini hesapla."""
        return [self.fitness_func(ind) for ind in population]
    
    def select_parents(self, population: List[List], fitness_scores: List[float]) -> List[List]:
        """Turnuva seçimi ile ebeveynler seç."""
        parents = []
        for _ in range(self.config.population_size - self.config.elite_count):
            # Turnuva
            indices = random.sample(range(len(population)), self.config.tournament_size)
            tournament = [(population[i], fitness_scores[i]) for i in indices]
            winner = min(tournament, key=lambda x: x[1])  # Minimizasyon
            parents.append(copy.deepcopy(winner[0]))
        return parents
    
    def create_offspring(self, parents: List[List]) -> List[List]:
        """Çaprazlama ve mutasyon ile yeni nesil oluştur."""
        offspring = []
        
        # İki ebeveynden çocuk üret
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            # Çaprazlama
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutasyon
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.config.population_size - self.config.elite_count]
    
    def get_elite(self, population: List[List], fitness_scores: List[float]) -> List[List]:
        """En iyi bireyleri seç (elitizm)."""
        sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        return [copy.deepcopy(p) for p, f in sorted_pop[:self.config.elite_count]]
    
    def evolve(self, verbose: bool = True) -> Tuple[List, float]:
        """
        Evrim sürecini çalıştır.
        
        Returns:
            (en_iyi_birey, en_iyi_fitness)
        """
        # Başlangıç popülasyonu
        population = self.initialize_population()
        fitness_scores = self.evaluate_population(population)
        
        best_ever = None
        best_ever_fitness = float('inf')
        
        for generation in range(self.config.generations):
            # İstatistikler
            best_idx = fitness_scores.index(min(fitness_scores))
            best_fitness = fitness_scores[best_idx]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            worst_fitness = max(fitness_scores)
            
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['worst_fitness'].append(worst_fitness)
            
            # En iyi çözümü güncelle
            if best_fitness < best_ever_fitness:
                best_ever_fitness = best_fitness
                best_ever = copy.deepcopy(population[best_idx])
            
            if verbose and generation % 50 == 0:
                print(f"Gen {generation:4d} | Best: {best_fitness:.6f} | "
                      f"Avg: {avg_fitness:.6f} | Best Ever: {best_ever_fitness:.6f}")
            
            # Elitleri koru
            elite = self.get_elite(population, fitness_scores)
            
            # Ebeveyn seçimi
            parents = self.select_parents(population, fitness_scores)
            
            # Yeni nesil oluştur
            offspring = self.create_offspring(parents)
            
            # Yeni popülasyon = elit + yeni çocuklar
            population = elite + offspring
            
            # Yeni popülasyonu değerlendir
            fitness_scores = self.evaluate_population(population)
        
        return best_ever, best_ever_fitness


# ═══════════════════════════════════════════════════════════════
# SANTA 2025 İÇİN UYGULAMA
# ═══════════════════════════════════════════════════════════════

class TreePackingGA:
    """Santa 2025 için Genetik Algoritma uygulaması."""
    
    def __init__(self, num_trees: int, tree_template):
        self.num_trees = num_trees
        self.tree_template = tree_template
        
    def create_individual(self) -> List[float]:
        """
        Rastgele bir birey oluştur.
        Genom: [x1, y1, θ1, x2, y2, θ2, ..., xN, yN, θN]
        """
        individual = []
        for _ in range(self.num_trees):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            angle = random.uniform(0, 360)
            individual.extend([x, y, angle])
        return individual
    
    def fitness(self, individual: List[float]) -> float:
        """
        Fitness fonksiyonu.
        Düşük değer = daha iyi (minimizasyon).
        """
        trees = self.decode_individual(individual)
        
        # Çarpışma kontrolü
        collision_count = self.count_collisions(trees)
        collision_penalty = collision_count * 100
        
        # Bounding box
        bbox_size = self.calculate_bounding_box(trees)
        
        return bbox_size + collision_penalty
    
    def decode_individual(self, individual: List[float]):
        """Genomu ağaç listesine dönüştür."""
        trees = []
        for i in range(self.num_trees):
            x = individual[i * 3]
            y = individual[i * 3 + 1]
            angle = individual[i * 3 + 2]
            tree = self.tree_template.create_at(x, y, angle)
            trees.append(tree)
        return trees
    
    def crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """İki noktalı çaprazlama (ağaç bazında)."""
        # Ağaç sınırlarında çaprazlama yap
        tree_point1 = random.randint(0, self.num_trees - 1)
        tree_point2 = random.randint(tree_point1, self.num_trees)
        
        point1 = tree_point1 * 3
        point2 = tree_point2 * 3
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def mutate(self, individual: List, mutation_rate: float = 0.1) -> List:
        """Gaussian mutasyon."""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                if i % 3 == 2:  # Açı
                    mutated[i] += random.gauss(0, 15)
                    mutated[i] %= 360
                else:  # x veya y
                    mutated[i] += random.gauss(0, 0.2)
        
        return mutated
    
    def count_collisions(self, trees) -> int:
        """Çarpışan çift sayısı."""
        count = 0
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                if trees[i].polygon.intersects(trees[j].polygon):
                    if not trees[i].polygon.touches(trees[j].polygon):
                        count += 1
        return count
    
    def calculate_bounding_box(self, trees) -> float:
        """Minimum kare kenar uzunluğu."""
        from shapely.ops import unary_union
        if not trees:
            return 0.0
        combined = unary_union([t.polygon for t in trees])
        minx, miny, maxx, maxy = combined.bounds
        return max(maxx - minx, maxy - miny)
    
    def optimize(self):
        """GA optimizasyonunu çalıştır."""
        config = GAConfig(
            population_size=100,
            generations=500,
            crossover_rate=0.85,
            mutation_rate=0.15,
            elite_count=5,
            tournament_size=3
        )
        
        ga = GeneticAlgorithm(
            fitness_func=self.fitness,
            create_individual=self.create_individual,
            crossover_func=self.crossover,
            mutation_func=self.mutate,
            config=config
        )
        
        best_individual, best_fitness = ga.evolve(verbose=True)
        return self.decode_individual(best_individual), best_fitness
```

## 2.5 Parametre Ayarlama Rehberi

```
┌─────────────────────────────────────────────────────────────────┐
│                   GA PARAMETRE AYARLAMA                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  population_size (Popülasyon Boyutu):                           │
│  ├── Küçük (20-50): Hızlı ama düşük çeşitlilik                  │
│  ├── Orta (50-200): Dengeli                                     │
│  └── Büyük (200+): Yüksek çeşitlilik, yavaş                     │
│                                                                 │
│  crossover_rate (Çaprazlama Oranı):                             │
│  ├── Düşük (0.5-0.7): Daha fazla keşif                          │
│  └── Yüksek (0.8-0.95): Daha fazla sömürü                       │
│                                                                 │
│  mutation_rate (Mutasyon Oranı):                                │
│  ├── Düşük (0.001-0.01): İnce ayar                              │
│  ├── Orta (0.01-0.1): Dengeli                                   │
│  └── Yüksek (0.1-0.3): Rastgele aramaya yaklaşır                │
│                                                                 │
│  elite_count (Elit Sayısı):                                     │
│  └── Genelde popülasyonun %1-5'i                                │
│                                                                 │
│  tournament_size (Turnuva Boyutu):                              │
│  ├── 2: Düşük seçim baskısı                                     │
│  ├── 3-5: Orta                                                  │
│  └── 7+: Yüksek seçim baskısı                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2.6 Avantajlar ve Dezavantajlar

| ✅ Avantajlar | ❌ Dezavantajlar |
|--------------|-----------------|
| Global arama kapasitesi | Hesaplama maliyeti yüksek |
| Paralelize edilebilir | Parametre ayarı zor |
| Çoklu çözüm bulabilir | Erken yakınsama riski |
| Kesikli/sürekli problemler | Çözüm temsili tasarımı kritik |
| Kısıtları ceza ile ekleyebilir | Hiperparametre sayısı fazla |

---

# 3. Reinforcement Learning (Pekiştirmeli Öğrenme)

## 3.1 Temel Kavram

Reinforcement Learning (RL), bir ajanın çevre ile etkileşerek deneme-yanılma yoluyla optimal davranışı öğrendiği bir makine öğrenmesi paradigmasıdır.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL TEMEL DÖNGÜSÜ                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│         ┌─────────────────────────────────────────┐             │
│         │              ÇEVRE (Environment)        │             │
│         │                                         │             │
│         │  • Santa 2025: Ağaç yerleşim alanı      │             │
│         │  • Durum: Ağaçların pozisyonları        │             │
│         │  • Ödül: -bbox_size (minimize)          │             │
│         │                                         │             │
│         └──────────┬──────────────────┬───────────┘             │
│                    │                  │                         │
│              Durum (s)           Ödül (r)                       │
│                    │                  │                         │
│                    ▼                  │                         │
│         ┌─────────────────────────────┴───────────┐             │
│         │              AJAN (Agent)               │             │
│         │                                         │             │
│         │  • Politika: π(s) → a                   │             │
│         │  • Değer fonksiyonu: V(s) veya Q(s,a)   │             │
│         │  • Öğrenme algoritması: DQN, PPO, A2C   │             │
│         │                                         │             │
│         └──────────────────┬──────────────────────┘             │
│                            │                                    │
│                       Aksiyon (a)                               │
│                            │                                    │
│                            ▼                                    │
│                   Ağacı hareket ettir                           │
│                   (x, y, açı değişikliği)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 Temel Kavramlar

### 3.2.1 Markov Decision Process (MDP)

```
MDP = (S, A, P, R, γ)

S  = Durum uzayı (States)
     Santa 2025: Tüm ağaçların (x, y, θ) değerleri
     
A  = Aksiyon uzayı (Actions)  
     Santa 2025: Hangi ağacı, hangi yöne, ne kadar hareket ettir
     
P  = Geçiş olasılıkları P(s'|s,a)
     Santa 2025: Deterministik (aksiyonun sonucu kesin)
     
R  = Ödül fonksiyonu R(s, a, s')
     Santa 2025: -bounding_box_size - collision_penalty
     
γ  = İndirim faktörü (0 < γ ≤ 1)
     Gelecekteki ödüllerin şimdiki değeri
```

### 3.2.2 Değer Fonksiyonları

```python
# Durum Değer Fonksiyonu V(s)
# Bir durumdan başlayıp politikayı takip ederek beklenen toplam ödül
V_π(s) = E_π[Σ γ^t * R_t | S_0 = s]

# Aksiyon-Değer Fonksiyonu Q(s,a)
# Bir durumda belirli bir aksiyonu alıp sonra politikayı takip ederek beklenen ödül
Q_π(s,a) = E_π[Σ γ^t * R_t | S_0 = s, A_0 = a]

# Bellman Denklemi
V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * max_a' Q(s', a')
```

## 3.3 RL Algoritmaları

### 3.3.1 Q-Learning (Tablo Bazlı)

```python
# ═══════════════════════════════════════════════════════════════
# Q-LEARNING ALGORİTMASI
# ═══════════════════════════════════════════════════════════════

class QLearning:
    """
    Tablo bazlı Q-Learning.
    Küçük, ayrık durum/aksiyon uzayları için.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        
        # Epsilon-greedy için
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-tablosu: Q[state][action] = değer
        self.Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]
    
    def choose_action(self, state: int) -> int:
        """Epsilon-greedy aksiyon seçimi."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return self.Q[state].index(max(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Q-değerini güncelle."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.Q[next_state])
        
        # TD Güncellemesi
        self.Q[state][action] += self.lr * (target - self.Q[state][action])
    
    def decay_epsilon(self):
        """Epsilon değerini azalt."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, n_episodes: int = 1000):
        """Eğitim döngüsü."""
        rewards_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            self.decay_epsilon()
            rewards_history.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = sum(rewards_history[-100:]) / min(100, len(rewards_history))
                print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return rewards_history
```

### 3.3.2 Deep Q-Network (DQN)

```python
# ═══════════════════════════════════════════════════════════════
# DEEP Q-NETWORK (DQN)
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """Q-değerlerini tahmin eden sinir ağı."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Deneyim tekrarı için bellek."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network ajanı.
    
    Özellikler:
    - Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Ağlar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Sayaç
        self.steps = 0
    
    def choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy aksiyon seçimi."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def update(self):
        """Q-ağını güncelle."""
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Batch örnekle
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Mevcut Q değerleri
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Hedef Q değerleri
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss ve güncelleme
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Target network güncelle
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, n_episodes: int = 1000):
        """DQN eğitim döngüsü."""
        rewards_history = []
        losses = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            episode_loss = 0
            steps = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.buffer.push(state, action, reward, next_state, done)
                loss = self.update()
                
                state = next_state
                total_reward += reward
                episode_loss += loss
                steps += 1
            
            self.decay_epsilon()
            rewards_history.append(total_reward)
            losses.append(episode_loss / max(1, steps))
            
            if episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                avg_loss = np.mean(losses[-50:])
                print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | "
                      f"Avg: {avg_reward:8.2f} | Loss: {avg_loss:.4f} | "
                      f"ε: {self.epsilon:.4f}")
        
        return rewards_history
```

### 3.3.3 Proximal Policy Optimization (PPO)

```python
# ═══════════════════════════════════════════════════════════════
# PROXIMAL POLICY OPTIMIZATION (PPO)
# ═══════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """Actor-Critic ağ yapısı."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Ortak katmanlar
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (politika)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (değer)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state):
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), value


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) ajanı.
    
    PPO, TRPO'nun daha basit ve etkili bir versiyonudur.
    Politika güncelleme adımlarını kırpar (clip) ve stabilite sağlar.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Generalized Advantage Estimation (GAE) hesapla.
        
        GAE, varyans-bias dengesini kontrol eder.
        λ=0 → TD(0), düşük varyans, yüksek bias
        λ=1 → Monte Carlo, yüksek varyans, düşük bias
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO güncelleme adımı."""
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Avantajları normalize et
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            # Mini-batch'ler üzerinde güncelle
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Yeni politika ile olasılıklar
                action_probs, values = self.network(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Oran (importance sampling)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Toplam loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def train(self, env, n_iterations: int = 1000, steps_per_iteration: int = 2048):
        """PPO eğitim döngüsü."""
        
        rewards_history = []
        
        for iteration in range(n_iterations):
            # Trajectory topla
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            
            state = env.reset()
            episode_reward = 0
            
            for _ in range(steps_per_iteration):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, value = self.network.get_action(state_tensor)
                
                next_state, reward, done, _ = env.step(action.item())
                
                states.append(state)
                actions.append(action.item())
                log_probs.append(log_prob.item())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())
                
                episode_reward += reward
                state = next_state
                
                if done:
                    rewards_history.append(episode_reward)
                    episode_reward = 0
                    state = env.reset()
            
            # Son state değeri
            with torch.no_grad():
                _, next_value = self.network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            
            # GAE hesapla
            advantages = self.compute_gae(rewards, values, dones, next_value.item())
            returns = [adv + val for adv, val in zip(advantages, values)]
            
            # Güncelle
            self.update(states, actions, log_probs, returns, advantages)
            
            if iteration % 10 == 0 and rewards_history:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Iteration {iteration:4d} | Avg Reward: {avg_reward:.2f}")
        
        return rewards_history
```

## 3.4 Santa 2025 İçin RL Ortamı

```python
# ═══════════════════════════════════════════════════════════════
# SANTA 2025 RL ORTAMI
# ═══════════════════════════════════════════════════════════════

import gym
from gym import spaces

class TreePackingEnv(gym.Env):
    """
    Santa 2025 için OpenAI Gym uyumlu ortam.
    
    State: Tüm ağaçların [x, y, θ] değerleri (flatten)
    Action: (tree_id, move_type, magnitude)
    Reward: -delta_bbox (küçülme pozitif) - collision_penalty
    """
    
    def __init__(self, num_trees: int, max_steps: int = 1000):
        super().__init__()
        
        self.num_trees = num_trees
        self.max_steps = max_steps
        self.current_step = 0
        
        # Durum uzayı: N ağaç × 3 değer (x, y, θ)
        self.observation_space = spaces.Box(
            low=-10, high=10,
            shape=(num_trees * 3,),
            dtype=np.float32
        )
        
        # Aksiyon uzayı: 
        # - tree_id: hangi ağaç (0 to N-1)
        # - move_type: 0=x+, 1=x-, 2=y+, 3=y-, 4=rotate+, 5=rotate-
        # Toplam: N × 6 ayrık aksiyon
        self.action_space = spaces.Discrete(num_trees * 6)
        
        self.trees = None
        self.best_bbox = float('inf')
    
    def reset(self):
        """Ortamı sıfırla."""
        self.current_step = 0
        self.trees = self._initialize_trees()
        self.best_bbox = self._calculate_bbox()
        return self._get_state()
    
    def step(self, action: int):
        """Bir adım at."""
        self.current_step += 1
        
        # Aksiyonu decode et
        tree_id = action // 6
        move_type = action % 6
        
        # Hareketi uygula
        old_bbox = self._calculate_bbox()
        self._apply_action(tree_id, move_type)
        
        # Çarpışma kontrolü
        collision = self._check_collision()
        
        if collision:
            # Hareketi geri al
            self._undo_action(tree_id, move_type)
            reward = -1  # Ceza
        else:
            new_bbox = self._calculate_bbox()
            # Ödül: bbox küçültme
            reward = (old_bbox - new_bbox) * 10
            
            if new_bbox < self.best_bbox:
                reward += 5  # Bonus
                self.best_bbox = new_bbox
        
        done = self.current_step >= self.max_steps
        info = {'bbox': self._calculate_bbox(), 'collisions': collision}
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Durumu döndür."""
        state = []
        for tree in self.trees:
            state.extend([
                float(tree.center_x),
                float(tree.center_y),
                float(tree.angle) / 360.0  # Normalize
            ])
        return np.array(state, dtype=np.float32)
    
    def _apply_action(self, tree_id: int, move_type: int):
        """Aksiyonu uygula."""
        step_size = 0.05
        angle_step = 5.0
        
        tree = self.trees[tree_id]
        
        if move_type == 0:    # x+
            tree.center_x += step_size
        elif move_type == 1:  # x-
            tree.center_x -= step_size
        elif move_type == 2:  # y+
            tree.center_y += step_size
        elif move_type == 3:  # y-
            tree.center_y -= step_size
        elif move_type == 4:  # rotate+
            tree.angle = (tree.angle + angle_step) % 360
        elif move_type == 5:  # rotate-
            tree.angle = (tree.angle - angle_step) % 360
        
        tree.rebuild_polygon()
    
    def _undo_action(self, tree_id: int, move_type: int):
        """Aksiyonu geri al."""
        # Ters hareket
        reverse_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
        self._apply_action(tree_id, reverse_map[move_type])
    
    def _initialize_trees(self):
        """Ağaçları başlat (greedy ile)."""
        # Mevcut initialize_trees fonksiyonunu kullan
        trees, _ = initialize_trees(self.num_trees)
        return trees
    
    def _calculate_bbox(self) -> float:
        """Bounding box boyutunu hesapla."""
        from shapely.ops import unary_union
        combined = unary_union([t.polygon for t in self.trees])
        minx, miny, maxx, maxy = combined.bounds
        return max(maxx - minx, maxy - miny)
    
    def _check_collision(self) -> bool:
        """Çarpışma var mı kontrol et."""
        for i in range(self.num_trees):
            for j in range(i + 1, self.num_trees):
                if self.trees[i].polygon.intersects(self.trees[j].polygon):
                    if not self.trees[i].polygon.touches(self.trees[j].polygon):
                        return True
        return False


# Kullanım örneği:
# env = TreePackingEnv(num_trees=10)
# agent = DQNAgent(state_dim=30, action_dim=60)
# agent.train(env, n_episodes=1000)
```

## 3.5 RL vs Diğer Yöntemler

```
┌─────────────────────────────────────────────────────────────────┐
│                 RL NE ZAMAN KULLANILMALI?                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ RL UYGUN:                                                   │
│  ├── Sıralı karar problemleri                                   │
│  ├── Öğrenilecek bir "politika" var                             │
│  ├── Dinamik çevre                                              │
│  ├── Benzer problemler tekrar çözülecek                         │
│  └── Simülasyon yapılabilir                                     │
│                                                                 │
│  ❌ RL UYGUN DEĞİL:                                             │
│  ├── Tek seferlik optimizasyon (SA/GA daha iyi)                 │
│  ├── Küçük arama uzayı (brute force yeterli)                    │
│  └── Eğitim verisi/zamanı kısıtlı                               │
│                                                                 │
│  SANTA 2025 İÇİN:                                               │
│  ├── RL: Transfer öğrenme ile N→N+1 geçişi                      │
│  ├── RL: Yerleştirme stratejisi öğrenme                         │
│  └── Hibrit: Greedy→SA→RL fine-tuning                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3.6 Avantajlar ve Dezavantajlar

| ✅ Avantajlar | ❌ Dezavantajlar |
|--------------|-----------------|
| Politika öğrenir (transfer) | Uzun eğitim süresi |
| Sıralı kararlar için ideal | Sample efficiency düşük |
| Dinamik ortamlara uyum | Hyperparameter hassasiyeti |
| End-to-end öğrenme | Reward engineering zor |
| Paralelize edilebilir | Stabilite sorunları |

---

# 4. Karşılaştırma ve Seçim Kriterleri

## 4.1 Özet Karşılaştırma Tablosu

| Kriter | Simulated Annealing | Genetik Algoritma | Reinforcement Learning |
|--------|---------------------|-------------------|------------------------|
| **Öğrenme Tipi** | Yok | Yok | Politika öğrenir |
| **Paralelize** | Bağımsız çalıştır | Popülasyon bazlı | Çoklu ortam |
| **Hesaplama** | Orta | Yüksek | Çok yüksek (eğitim) |
| **Parametre Sayısı** | 4-5 | 6-8 | 10+ |
| **Lokal Min. Kaçış** | İyi | Çok iyi | Orta |
| **Yakınsama** | Garanti yok | Garanti yok | Garanti yok |
| **Transfer** | Yok | Sınırlı | Var |
| **Implementasyon** | Kolay | Orta | Zor |
| **Debug** | Kolay | Orta | Zor |

## 4.2 Problem Tipine Göre Seçim

```
┌─────────────────────────────────────────────────────────────────┐
│                 PROBLEM → YÖNTEM EŞLEŞTİRME                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Sürekli optimizasyon, tek çözüm:                               │
│  └── Simulated Annealing ✓                                      │
│                                                                 │
│  Kombinatorik optimizasyon, çeşitlilik:                         │
│  └── Genetik Algoritma ✓                                        │
│                                                                 │
│  Sıralı karar, politika öğrenme:                                │
│  └── Reinforcement Learning ✓                                   │
│                                                                 │
│  Hızlı sonuç, basit problem:                                    │
│  └── SA > GA > RL                                               │
│                                                                 │
│  Büyük arama uzayı, paralel kaynak:                             │
│  └── GA > SA > RL                                               │
│                                                                 │
│  Benzer problemler, uzun vadeli:                                │
│  └── RL > GA > SA                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 5. Santa 2025 İçin Uygulama Önerileri

## 5.1 Önerilen Hibrit Strateji

```
┌─────────────────────────────────────────────────────────────────┐
│                    3 AŞAMALI OPTİMİZASYON                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AŞAMA 1: BAŞLANGIÇ (Greedy veya GA)                            │
│  ════════════════════════════════════                           │
│  • Hızlı, kabul edilebilir başlangıç çözümü                     │
│  • GA ile çeşitli başlangıç noktaları üret                      │
│  • En iyi 5-10 çözümü sakla                                     │
│                                                                 │
│  AŞAMA 2: GLOBAL İYİLEŞTİRME (SA)                               │
│  ════════════════════════════════════                           │
│  • Her başlangıç çözümü için SA çalıştır                        │
│  • Paralel execution                                            │
│  • Uzun soğutma programı                                        │
│                                                                 │
│  AŞAMA 3: FINE-TUNING (Local Search veya RL)                    │
│  ════════════════════════════════════                           │
│  • En iyi SA sonuçları üzerinde                                 │
│  • Açı grid search (her 1°)                                     │
│  • Mikro pozisyon ayarları                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 Pratik Tavsiyeler

```python
# ═══════════════════════════════════════════════════════════════
# SANTA 2025 İÇİN PRATİK UYGULAMA
# ═══════════════════════════════════════════════════════════════

def optimize_all_configurations():
    """Tüm N değerleri için optimize et."""
    
    results = {}
    
    for n in range(1, 201):
        print(f"\n{'='*50}")
        print(f"Optimizing N={n}")
        print('='*50)
        
        # Aşama 1: Başlangıç çözümleri (paralel GA)
        initial_solutions = parallel_ga_starts(n, num_starts=5)
        
        # Aşama 2: SA ile iyileştir (paralel)
        sa_results = []
        for sol in initial_solutions:
            optimized, score = simulated_annealing(sol, 
                                                   T_initial=100,
                                                   T_final=1e-6,
                                                   cooling_rate=0.99999)
            sa_results.append((optimized, score))
        
        # En iyiyi seç
        best_solution, best_score = min(sa_results, key=lambda x: x[1])
        
        # Aşama 3: Fine-tuning
        final_solution, final_score = local_search(best_solution)
        final_solution, final_score = angle_optimization(final_solution)
        
        results[n] = {
            'trees': final_solution,
            'bbox': final_score
        }
        
        print(f"N={n}: bbox = {final_score:.12f}")
    
    return results


def parallel_ga_starts(n, num_starts=5):
    """Paralel GA başlangıçları."""
    from multiprocessing import Pool
    
    def single_ga_run(_):
        ga = TreePackingGA(n)
        return ga.optimize()[0]
    
    with Pool(num_starts) as pool:
        solutions = pool.map(single_ga_run, range(num_starts))
    
    return solutions
```

## 5.3 Sonuç

| Yöntem | Santa 2025 Uygunluğu | Not |
|--------|---------------------|-----|
| **Simulated Annealing** | ⭐⭐⭐⭐⭐ | Ana optimizasyon aracı |
| **Genetik Algoritma** | ⭐⭐⭐⭐ | Başlangıç çeşitliliği |
| **Reinforcement Learning** | ⭐⭐⭐ | İleri seviye, transfer öğrenme |

---

## 📚 Kaynaklar

1. **Simulated Annealing:**
   - Kirkpatrick, S., et al. (1983). "Optimization by Simulated Annealing"
   
2. **Genetic Algorithms:**
   - Holland, J.H. (1975). "Adaptation in Natural and Artificial Systems"
   
3. **Reinforcement Learning:**
   - Sutton, R.S., Barto, A.G. (2018). "Reinforcement Learning: An Introduction"
   - Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"

---

*Bu doküman Santa 2025 Kaggle yarışması için hazırlanmıştır.*

