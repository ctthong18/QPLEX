def create_optimized_qplex(config):
    """Tối ưu hóa network architecture dựa trên bài báo"""
    
    # Sử dụng hierarchical RNN cho multi-timescale learning
    q_net_config = {
        'type': 'hierarchical_rnn',
        'hidden_dims': [512, 256],  # Tăng capacity
        'use_rnn': True,
        'rnn_hidden_dim': 256,      # Tăng hidden size
        'rnn_layers': 3,            # Thêm layers
        'rnn_type': 'lstm',
        'use_attention': True,
        'num_attention_heads': 8,   # Tăng attention heads
        'dropout': 0.2              # Thêm regularization
    }
    
    # Sử dụng adaptive mixing network
    mix_net_config = {
        'type': 'adaptive',
        'hidden_dims': [512, 256],
        'use_hypernet': True,
        'dueling': True,
        'complexity_threshold': 0.7,
        'num_levels': 3             # Thêm hierarchical levels
    }
    
    return QPLEXModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'], 
        state_dim=config['state_dim'],
        n_agents=config['n_agents'],
        config={
            'network': {
                'q_network': q_net_config,
                'mixing_network': mix_net_config,
                'use_state_encoder': True,
                'state_encoder_hidden': 256
            }
        }
    )