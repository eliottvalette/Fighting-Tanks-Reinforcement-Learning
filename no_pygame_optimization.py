# no_pygame_optimization.py

class Rect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return (self.left + self.width // 2, self.top + self.height // 2)

    @center.setter
    def center(self, pos):
        self.left = pos[0] - self.width // 2
        self.top = pos[1] - self.height // 2

    @property
    def topleft(self):
        return (self.left, self.top)

    @topleft.setter
    def topleft(self, pos):
        self.left = pos[0]
        self.top = pos[1]

    def colliderect(self, other):
        return (
            self.left < other.right and
            self.right > other.left and
            self.top < other.bottom and
            self.bottom > other.top
        )

    def collidepoint(self, x, y=None):
        if y is None:
            x, y = x  # If passed as a tuple (x, y)
        return self.left <= x <= self.right and self.top <= y <= self.bottom
