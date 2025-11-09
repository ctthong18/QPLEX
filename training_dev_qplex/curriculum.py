class CurriculumExploration:
    def __init__(self, config):
        self.phase = 0
        self.phase_thresholds = [5000, 15000, 30000]  # Episode thresholds
        self.epsilon_schedule = {
            0: {'start': 1.0, 'end': 0.3, 'decay': 0.999},  # Phase 1: High exploration
            1: {'start': 0.3, 'end': 0.1, 'decay': 0.9995}, # Phase 2: Medium exploration  
            2: {'start': 0.1, 'end': 0.05, 'decay': 0.9998}, # Phase 3: Low exploration
            3: {'start': 0.05, 'end': 0.02, 'decay': 0.9999} # Phase 4: Fine-tuning
        }
        
    def get_epsilon(self, episode):
        # Update phase based on episode count
        for i, threshold in enumerate(self.phase_thresholds):
            if episode < threshold:
                self.phase = i
                break
        else:
            self.phase = len(self.phase_thresholds)
            
        schedule = self.epsilon_schedule[self.phase]
        epsilon = max(schedule['end'], 
                     schedule['start'] * (schedule['decay'] ** episode))
        return epsilon
    
    def get_environment_difficulty(self, episode):
        """Curriculum learning cho environment difficulty"""
        if episode < 5000:
            return 'easy'    # Static target, no obstacles
        elif episode < 15000:
            return 'medium'  # Moving target, few obstacles
        elif episode < 30000:
            return 'hard'    # Fast target, many obstacles
        else:
            return 'expert'  # Multiple targets, dynamic obstacles