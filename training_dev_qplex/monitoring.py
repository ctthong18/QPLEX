def setup_comprehensive_monitoring(learner, env):
    """Comprehensive monitoring system"""
    
    metrics = {
        'coverage_metrics': [],
        'training_metrics': [], 
        'network_metrics': [],
        'exploration_metrics': []
    }
    
    def log_metrics(episode, eval_reward, coverage_state):
        # Coverage metrics
        coverage_rate = np.mean(coverage_state['coverage_scores'])
        target_coverage = coverage_state['target_coverage_rate']
        obstacle_violations = coverage_state['obstacle_violations']
        
        # Training metrics
        stats = learner.get_training_stats()
        td_error = stats.get('td_error', 0)
        q_values = stats.get('q_values', 0)
        
        metrics['coverage_metrics'].append({
            'episode': episode, 'coverage_rate': coverage_rate,
            'target_coverage': target_coverage, 'obstacle_violations': obstacle_violations
        })
        
        metrics['training_metrics'].append({
            'episode': episode, 'eval_reward': eval_reward,
            'td_error': td_error, 'q_values': q_values
        })
        
        # Log every 100 episodes
        if episode % 100 == 0:
            print(f"Coverage: {coverage_rate:.3f}, Target: {target_coverage:.3f}, "
                  f"TD Error: {td_error:.3f}")
    
    return log_metrics