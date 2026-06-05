import random
import string


class Babbling:
    def __init__(self):
        self.cycle_count = 0
        self.last_babble = ""
        self.chars = string.ascii_lowercase + "01"

    def update(self, cycle):
        self.cycle_count = cycle
        if cycle % 50 == 0 and random.random() < 0.3:
            length = random.randint(3, 8)
            self.last_babble = "".join(random.choice(self.chars) for _ in range(length))
            print(f"[AURA] {self.last_babble}")

    def get_last(self):
        return self.last_babble
