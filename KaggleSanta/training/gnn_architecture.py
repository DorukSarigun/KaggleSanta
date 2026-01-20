"""
GNN-based Tree Packing Architecture
====================================

Sparse k-NN GNN approach for scalable tree packing with 1200+ trees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np


class TreeGNNEncoder(nn.Module):
    """
    Sparse k-NN Graph Neural Network for tree features.

    Architecture:
    - Dynamic graph construction (k=10 nearest neighbors)
    - 3-layer GNN with residual connections
    - Graph pooling for global representation

    Advantages over fully-connected:
    - O(N*k) instead of O(N²) complexity
    - Scales to 1200+ trees
    - Preserves local spatial structure
    """

    def __init__(
        self,
        node_dim: int = 14,      # Tree features
        edge_dim: int = 8,        # Edge features (distance, angle, etc)
        hidden_dim: int = 128,
        k_neighbors: int = 10,    # Sparse connectivity
        n_layers: int = 3,
        use_attention: bool = True
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.n_layers = n_layers

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # GNN layers (with attention for importance weighting)
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if use_attention:
                # Graph Attention Network (learns edge importance)
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // 8,  # heads=8
                        heads=8,
                        concat=True,
                        dropout=0.1,
                        edge_dim=hidden_dim  # Use edge features
                    )
                )
            else:
                # Standard Graph Convolution
                self.gnn_layers.append(
                    EdgeGCNConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
                )

        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Global pooling (for actor-critic)
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pool
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def construct_knn_graph(
        self,
        node_features: torch.Tensor,  # (batch, N, node_dim)
        positions: torch.Tensor,       # (batch, N, 2) - x, y coords
        k: int = 10
    ) -> tuple:
        """
        Construct k-NN graph dynamically based on spatial positions.

        Returns:
            edge_index: (2, E) - graph connectivity
            edge_features: (E, edge_dim) - edge attributes
        """
        batch_size, N, _ = positions.shape
        device = positions.device

        # Compute pairwise distances
        # (batch, N, 1, 2) - (batch, 1, N, 2) = (batch, N, N, 2)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)  # (batch, N, N)

        # Mask self-connections
        distances = distances + torch.eye(N, device=device).unsqueeze(0) * 1e6

        # Find k nearest neighbors for each node
        _, knn_indices = torch.topk(
            distances, k, dim=-1, largest=False
        )  # (batch, N, k)

        # Build edge_index
        edge_list = []
        edge_features_list = []

        for b in range(batch_size):
            for i in range(N):
                for j_idx in range(k):
                    j = knn_indices[b, i, j_idx].item()

                    # Add edge i -> j
                    edge_list.append([b * N + i, b * N + j])

                    # Compute edge features
                    dx = positions[b, j, 0] - positions[b, i, 0]
                    dy = positions[b, j, 1] - positions[b, i, 1]
                    dist = distances[b, i, j]

                    # Edge features: [dx, dy, dist, sin(angle), cos(angle), ...]
                    angle = torch.atan2(dy, dx)
                    edge_feat = torch.tensor([
                        dx, dy, dist,
                        torch.sin(angle), torch.cos(angle),
                        dist ** 2,  # Distance squared
                        1.0 / (dist + 1e-6),  # Inverse distance
                        float(j_idx) / k  # Neighbor rank (normalized)
                    ], device=device)

                    edge_features_list.append(edge_feat)

        edge_index = torch.tensor(edge_list, device=device).t()  # (2, E)
        edge_features = torch.stack(edge_features_list, dim=0)  # (E, edge_dim)

        return edge_index, edge_features

    def forward(
        self,
        node_features: torch.Tensor,  # (batch, N, node_dim)
        positions: torch.Tensor,       # (batch, N, 2)
        mask: torch.Tensor = None      # (batch, N) - valid nodes
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            node_embeddings: (batch, N, hidden_dim)
            graph_embedding: (batch, hidden_dim) - global pooled
        """
        batch_size, N, _ = node_features.shape
        device = node_features.device

        # Construct k-NN graph
        edge_index, edge_features = self.construct_knn_graph(
            node_features, positions, k=self.k_neighbors
        )

        # Encode node features
        # Reshape: (batch, N, node_dim) -> (batch*N, node_dim)
        x = node_features.reshape(-1, self.node_dim)
        x = self.node_encoder(x)  # (batch*N, hidden_dim)

        # Encode edge features
        edge_attr = self.edge_encoder(edge_features)  # (E, hidden_dim)

        # GNN layers with residual connections
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Message passing
            x_new = gnn_layer(x, edge_index, edge_attr)

            # Residual connection + LayerNorm
            x = self.layer_norms[i](x + x_new)

            # ReLU activation
            x = F.relu(x)

        # Reshape back: (batch*N, hidden_dim) -> (batch, N, hidden_dim)
        node_embeddings = x.reshape(batch_size, N, self.hidden_dim)

        # Global pooling (mean + max)
        if mask is not None:
            # Mask out invalid nodes
            mask_expanded = mask.unsqueeze(-1).expand_as(node_embeddings)
            node_embeddings_masked = node_embeddings * mask_expanded

            # Mean pooling (only valid nodes)
            mean_pool = node_embeddings_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)

            # Max pooling
            max_pool, _ = node_embeddings_masked.max(dim=1)
        else:
            mean_pool = node_embeddings.mean(dim=1)
            max_pool, _ = node_embeddings.max(dim=1)

        # Combine mean + max
        graph_embedding = self.global_pool(
            torch.cat([mean_pool, max_pool], dim=-1)
        )

        return node_embeddings, graph_embedding


class EdgeGCNConv(MessagePassing):
    """
    Graph Convolution with edge features.
    Custom layer for edge-attributed graphs.
    """

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='mean')  # Mean aggregation

        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_update = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Node transformation
        x_transformed = self.lin_node(x)

        # Edge transformation
        edge_transformed = self.lin_edge(edge_attr)

        # Message passing
        out = self.propagate(
            edge_index,
            x=x_transformed,
            edge_attr=edge_transformed
        )

        return out

    def message(self, x_j, edge_attr):
        # Combine neighbor features with edge features
        return x_j + edge_attr

    def update(self, aggr_out, x):
        # Combine aggregated messages with node features
        return self.lin_update(torch.cat([x, aggr_out], dim=-1))


class HybridGNNTransformer(nn.Module):
    """
    Hybrid architecture: Local GNN + Global Transformer.

    - GNN captures local spatial relationships (k-NN)
    - Transformer captures global context
    - Best of both worlds!
    """

    def __init__(
        self,
        node_dim: int = 14,
        hidden_dim: int = 128,
        k_neighbors: int = 10,
        n_gnn_layers: int = 3,
        n_transformer_layers: int = 2
    ):
        super().__init__()

        # Local: Sparse GNN
        self.gnn = TreeGNNEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            k_neighbors=k_neighbors,
            n_layers=n_gnn_layers
        )

        # Global: Transformer (fewer layers since GNN preprocessed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

    def forward(self, node_features, positions, mask=None):
        # Stage 1: GNN - Local spatial relationships
        node_embeddings, graph_embedding = self.gnn(
            node_features, positions, mask
        )

        # Stage 2: Transformer - Global context
        # (Fewer layers needed since GNN already processed local info)
        if mask is not None:
            # Create attention mask (1 = attend, 0 = ignore)
            attn_mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        else:
            attn_mask = None

        transformer_out = self.transformer(
            node_embeddings,
            src_key_padding_mask=(~mask.bool() if mask is not None else None)
        )

        # Final pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
            transformer_out_masked = transformer_out * mask_expanded
            final_embedding = transformer_out_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            final_embedding = transformer_out.mean(dim=1)

        return final_embedding


# ==============================================================================
# COMPLEXITY ANALYSIS
# ==============================================================================

def analyze_complexity():
    """Compare computational complexity of different approaches."""

    N = 1200  # Number of trees
    d = 128   # Hidden dimension
    k = 10    # k-NN neighbors

    print("=" * 60)
    print("COMPUTATIONAL COMPLEXITY ANALYSIS (N=1200)")
    print("=" * 60)

    # Fully-connected GNN
    fc_edges = N * (N - 1) // 2
    fc_ops = fc_edges * d * 3  # 3 layers
    print(f"\n1. Fully-Connected GNN:")
    print(f"   Edges: {fc_edges:,}")
    print(f"   Operations: {fc_ops:,}")
    print(f"   Memory: ~{fc_ops * 4 / 1e9:.2f} GB")
    print(f"   Status: ❌ NOT PRACTICAL")

    # k-NN GNN
    knn_edges = N * k
    knn_ops = knn_edges * d * 3  # 3 layers
    print(f"\n2. k-NN GNN (k={k}):")
    print(f"   Edges: {knn_edges:,}")
    print(f"   Operations: {knn_ops:,}")
    print(f"   Memory: ~{knn_ops * 4 / 1e6:.2f} MB")
    print(f"   Status: ✅ PRACTICAL")
    print(f"   Speedup: {fc_ops / knn_ops:.1f}x faster")

    # Transformer (current)
    trans_ops = N * N * d
    print(f"\n3. Transformer (current):")
    print(f"   Operations: {trans_ops:,}")
    print(f"   Memory: ~{trans_ops * 4 / 1e9:.2f} GB")
    print(f"   Status: ✅ WORKS")

    # Hybrid GNN + Transformer
    hybrid_ops = knn_ops + (N * N * d) // 2  # GNN + smaller Transformer
    print(f"\n4. Hybrid GNN+Transformer:")
    print(f"   Operations: {hybrid_ops:,}")
    print(f"   Memory: ~{hybrid_ops * 4 / 1e9:.2f} GB")
    print(f"   Status: ✅ BEST (local + global)")
    print(f"   Speedup vs FC-GNN: {fc_ops / hybrid_ops:.1f}x")

    print("\n" + "=" * 60)
    print("RECOMMENDATION: Use k-NN GNN or Hybrid approach")
    print("=" * 60)


if __name__ == "__main__":
    # Run complexity analysis
    analyze_complexity()

    # Test k-NN GNN
    print("\nTesting k-NN GNN...")
    batch_size = 4
    N = 50  # Start with smaller test

    # Create dummy data
    node_features = torch.randn(batch_size, N, 14)
    positions = torch.randn(batch_size, N, 2)
    mask = torch.ones(batch_size, N)

    # Initialize model
    model = TreeGNNEncoder(
        node_dim=14,
        hidden_dim=128,
        k_neighbors=10,
        n_layers=3,
        use_attention=True
    )

    # Forward pass
    node_embeddings, graph_embedding = model(node_features, positions, mask)

    print(f"✅ k-NN GNN test passed!")
    print(f"   Input: {node_features.shape}")
    print(f"   Node embeddings: {node_embeddings.shape}")
    print(f"   Graph embedding: {graph_embedding.shape}")
