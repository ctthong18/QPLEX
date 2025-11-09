def get_enhanced_state_representation(env):
    """State representation được cải tiến từ cả 2 bài báo"""
    
    # Global state với multi-scale information
    global_state = np.concatenate([
        env.target_position.flatten(),           # Vị trí hiện tại
        env.target_velocity.flatten(),           # Vận tốc
        env.target_predicted_positions().flatten(), # Dự đoán vị trí (3 bước)
        env.obstacle_positions.flatten(),        # Vị trí obstacles
        env.obstacle_sizes.flatten(),            # Kích thước obstacles
        env.coverage_heatmap.flatten(),          # Coverage map
        env.get_coverage_gaps(),                 # Coverage gaps
        [env.time_step, env.episode_step]        # Temporal information
    ])
    
    # Local observations với neighborhood context
    local_obs = []
    for i, tensor in enumerate(env.tensors):
        tensor_obs = np.concatenate([
            # Basic information
            tensor.position,
            [np.cos(tensor.angle), np.sin(tensor.angle)],
            tensor.coverage_score,
            
            # Target information
            env.target_position - tensor.position,
            env.target_velocity,
            env.get_target_prediction(i),  # Target prediction for this tensor
            
            # Obstacle information  
            env.get_obstacle_distances(tensor.position),
            env.get_obstacle_directions(tensor.position),
            
            # Neighborhood information (Bài báo multi-agent)
            env.get_neighbor_positions(i),
            env.get_neighbor_coverage(i),
            env.get_coordination_signals(i),
            
            # Historical information (Bài báo RNN)
            tensor.get_rotation_history(),
            tensor.get_coverage_history(),
            
            # Energy and performance
            [tensor.energy_level, tensor.performance_score]
        ])
        local_obs.append(tensor_obs)
    
    return np.array(local_obs), global_state