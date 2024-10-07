from torch_geometric.graphgym.register import register_config


@register_config('sort_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """
    # Use residual connections between the GNN layers.
    cfg.gnn.residual = True

    cfg.gnn.norm = 'batch'

    cfg.gnn.combine = 'ConcatProject'
    cfg.gnn.collapse_method = 'vector'
    cfg.gnn.update_w_orig = True
    cfg.gnn.blank_vector_method = 'learnable'
    cfg.gnn.bias = True
    cfg.gnn.pooling = 'sort_global'

