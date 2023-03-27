class LinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __repr__(self):
        return f"LinearSchedule(initial_value={self.initial_value})"

    def __call__(self, progress):
        return progress * self.initial_value


# Reduce learning rate linearly after half the training steps
class HalfLinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __repr__(self):
        return f"HalfLinearSchedule(initial_value={self.initial_value})"

    def __call__(self, progress):
        if progress < 0.5:
            return progress * 2 * self.initial_value
        return self.initial_value


# Reduce learning rate linearly after half the training steps
class AlmostHalfLinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __repr__(self):
        return f"HalfLinearSchedule(initial_value={self.initial_value})"

    def __call__(self, progress):
        if progress < 0.5 and progress > 0.1:
            return progress * 2 * self.initial_value
        elif progress < 0.1:
            return 0.1 * self.initial_value
        return self.initial_value
