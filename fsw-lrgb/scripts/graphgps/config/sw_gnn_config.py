from torch_geometric.graphgym.register import register_config


@register_config('sw_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """
    # Use residual connections between the GNN layers.
    cfg.gnn.residual = True

    cfg.gnn.sw_embed_dim = None
    cfg.gnn.learnable_embedding = True
    cfg.gnn.concat_self = True
    cfg.gnn.bias = True
    cfg.gnn.conv_mlp_layers = 1
    cfg.gnn.conv_mlp_hidden_dim = None
    cfg.gnn.conv_mlp_activation_final = 'gelu'
    cfg.gnn.conv_mlp_activation_hidden = 'gelu'
    cfg.gnn.mlp_init = 'xavier_normal' # uniform
    cfg.gnn.batchnorm_final = True 
    cfg.gnn.batchnorm_hidden = True
    cfg.gnn.dropout_final = 0.0
    cfg.gnn.dropout_hidden = 0.0
    cfg.gnn.self_loop_weight = 1.0
    cfg.gnn.edge_weighting = 'unit' # Check gcn.
    cfg.gnn.homog_degree_encoding = False
    cfg.gnn.vertex_degree_pad_thresh = 0.5

