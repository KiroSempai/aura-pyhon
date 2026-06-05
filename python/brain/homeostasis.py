import numpy as np


class Homeostasis:
    def __init__(self):
        self.energia = 0.7
        self.integridad = 1.0
        self.curiosidad = 0.5
        self.homeostasis = 0.8

    def update(self, dt=1.0, moving=False, collided=False, impact_force=0.0, speed=0.0, significant_movement=False):
        dt = min(float(dt), 0.1)

        # --- Decaimiento base ---
        self.energia -= 0.03 * dt
        self.integridad -= 0.009 * dt

        # --- Curiosidad: siempre sube ---
        self.curiosidad += 0.06 * dt

        if significant_movement:
            self.curiosidad -= 0.18 * dt
        else:
            self.curiosidad += 0.12 * dt

        # --- Energia proporcional a velocidad ---
        if moving:
            speed_cost = 0.15 * dt + 0.08 * speed * dt
            self.energia -= speed_cost
        else:
            self.energia += 0.24 * dt

        # --- Colisiones ---
        if collided:
            damage = 0.03 + impact_force * 0.01
            self.integridad -= min(damage, 0.15)
            self.curiosidad = max(self.curiosidad - 0.10, 0.0)

            if damage > 0.08:
                self.homeostasis -= 0.05
                self.curiosidad = max(self.curiosidad - 0.15, 0.0)
                self.energia -= 0.02

        # --- Recuperacion de integridad ---
        self.integridad += 0.06 * dt

        # --- Interacciones ---
        if self.energia < 0.3:
            self.curiosidad = max(self.curiosidad - 0.04 * dt, 0.0)

        if self.integridad < 0.4:
            self.energia -= 0.03 * dt

        if self.curiosidad > 0.7 and moving:
            self.energia -= 0.08 * dt

        if self.curiosidad > 0.5 and not moving:
            self.energia -= 0.06 * dt

        # --- Estres por curiosidad alta prolongada ---
        if self.curiosidad > 0.85:
            stress = (self.curiosidad - 0.85) * 3.0
            self.homeostasis -= stress * dt * 0.03

        # --- Limites ---
        self.energia = np.clip(float(self.energia), 0.0, 1.0)
        self.integridad = np.clip(float(self.integridad), 0.0, 1.0)
        self.curiosidad = np.clip(float(self.curiosidad), 0.0, 1.0)

        # --- Homeostasis ---
        avg = (self.energia + self.integridad + self.curiosidad) / 3.0
        spread = np.std([self.energia, self.integridad, self.curiosidad])
        imbalance_penalty = spread * 0.10

        low_penalty = 0
        if self.energia < 0.2:
            low_penalty += 0.05
        if self.integridad < 0.2:
            low_penalty += 0.05

        balance_bonus = 0.15 if (
            0.3 <= self.energia <= 0.8
            and 0.3 <= self.integridad <= 0.8
            and 0.3 <= self.curiosidad <= 0.8
        ) else 0.0

        self.homeostasis = avg - imbalance_penalty - low_penalty + balance_bonus

        if self.curiosidad > 0.7:
            self.homeostasis -= 0.06
        if self.curiosidad < 0.2:
            self.homeostasis += 0.04

        self.homeostasis = np.clip(float(self.homeostasis), 0.0, 1.0)

        return self.get_state()

    def get_state(self):
        return {
            "energia": round(float(self.energia), 4),
            "integridad": round(float(self.integridad), 4),
            "curiosidad": round(float(self.curiosidad), 4),
            "homeostasis": round(float(self.homeostasis), 4),
        }
