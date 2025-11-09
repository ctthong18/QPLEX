import numpy as np

class OptimizedTensorReward:
    def __init__(self):
        self.weights = {
            'coverage': 2.0,      # Tăng coverage reward
            'tracking': 3.0,      # Tăng tracking reward  
            'energy': -0.05,      # Giảm energy penalty
            'obstacle': -1.0,     # Tăng obstacle penalty
            'collaboration': 1.0, # Thêm collaboration bonus
            'coverage_balance': 0.5, # Tránh tập trung
        }
        
    def compute(self, coverage_state):
        rewards = np.zeros(coverage_state['n_tensors'])
        
        # 1. Probabilistic coverage reward (Bài báo ACDRL)
        coverage_prob = self._compute_probabilistic_coverage(coverage_state)
        rewards += self.weights['coverage'] * coverage_prob
        
        # 2. Target tracking với distance-based decay
        target_reward = self._compute_target_tracking(coverage_state)
        rewards += self.weights['tracking'] * target_reward
        
        # 3. Energy-efficient penalty (Bài báo vision-based)
        energy_penalty = self.weights['energy'] * coverage_state['rotation_costs']
        rewards += energy_penalty
        
        # 4. Obstacle avoidance với exponential penalty
        obstacle_penalty = self.weights['obstacle'] * np.exp(-coverage_state['obstacle_distances'])
        rewards += obstacle_penalty
        
        # 5. Collaboration bonus (Tránh overlap)
        collaboration_bonus = self.weights['collaboration'] * self._compute_coverage_balance(coverage_state)
        rewards += collaboration_bonus
        
        return rewards
    
    def _compute_probabilistic_coverage(self, state):
        """Probabilistic coverage model từ bài báo ACDRL"""
        coverage_scores = []
        for i in range(state['n_tensors']):
            # Probabilistic sensing model
            distance_to_target = state['target_distances'][i]
            if distance_to_target <= state['reliable_radius']:
                prob = 1.0
            elif distance_to_target <= state['max_radius']:
                decay = np.exp(-0.5 * (distance_to_target - state['reliable_radius']))
                prob = decay
            else:
                prob = 0.0
            coverage_scores.append(prob)
        return np.array(coverage_scores)
    
    def _compute_target_tracking(self, state):
        """Improved target tracking với velocity prediction"""
        tracking_scores = []
        for i in range(state['n_tensors']):
            distance = state['target_distances'][i]
            # Adaptive decay based on target velocity
            velocity_factor = 1.0 / (1.0 + np.linalg.norm(state['target_velocity']))
            score = np.exp(-distance * velocity_factor)
            tracking_scores.append(score)
        return np.array(tracking_scores)
    
    def _compute_coverage_balance(self, state):
        """Encourage balanced coverage distribution"""
        coverage_scores = state['coverage_scores']
        balance_penalty = np.std(coverage_scores)  # Penalize imbalance
        return -balance_penalty