class AdvancedTraining:
    def __init__(self, learner, config):
        self.learner = learner
        self.config = config
        self.best_reward = -np.inf
        self.patience = 0
        self.max_patience = 1000
        
    def should_early_stop(self, eval_reward):
        """Early stopping khi performance plateau"""
        if eval_reward > self.best_reward:
            self.best_reward = eval_reward
            self.patience = 0
            return False
        else:
            self.patience += 1
            return self.patience >= self.max_patience
    
    def adaptive_learning_rate(self, episode, base_lr=3e-4):
        """Adaptive learning rate scheduling"""
        if episode < 10000:
            return base_lr  # Warm-up phase
        elif episode < 30000:
            return base_lr * 0.5  # Learning rate decay
        else:
            return base_lr * 0.1  # Fine-tuning phase
    
    def gradient_analysis(self, model):
        """Phân tích gradient để phát hiện vanishing/exploding gradients"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm