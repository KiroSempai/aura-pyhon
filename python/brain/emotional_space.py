import numpy as np


class EmotionalSpace:
    def __init__(self):
        self.arousal = 0.0
        self.valence = 0.0
        self.position = np.array([0.0, 0.0])
        self.hue = 180.0
        self.saturation = 0.0
        self.lightness = 0.5

    def update(self, hs_state):
        energia = hs_state["energia"]
        integridad = hs_state["integridad"]
        curiosidad = hs_state["curiosidad"]
        h = hs_state["homeostasis"]

        self.arousal = energia * 0.4 + curiosidad * 0.4 - (1.0 - integridad) * 0.2
        self.arousal = np.clip(self.arousal * 2.0 - 1.0, -1.0, 1.0)

        self.valence = h * 0.6 + integridad * 0.4
        self.valence = np.clip(self.valence * 2.0 - 1.0, -1.0, 1.0)

        self.position = np.array([self.arousal, self.valence])

        angle = np.arctan2(self.valence, self.arousal)
        self.hue = (np.degrees(angle) + 180) % 360

        magnitude = np.sqrt(self.arousal**2 + self.valence**2)
        self.saturation = np.clip(magnitude * 1.5, 0.0, 1.0)
        self.lightness = np.clip(0.5 + magnitude * 0.3, 0.0, 1.0)

        return self.get_state()

    def get_state(self):
        return {
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "hue": round(self.hue, 1),
            "saturation": round(self.saturation, 3),
            "lightness": round(self.lightness, 3),
            "quadrant": self._get_label(),
        }

    def _get_label(self):
        a, v = self.arousal, self.valence
        mag = np.sqrt(a**2 + v**2)
        if mag < 0.15:
            return "neutral"
        if a > 0.3 and v > 0.3:
            return "entusiasmo"
        if a > 0.3 and v < -0.3:
            return "estres"
        if a < -0.3 and v < -0.3:
            return "tristeza"
        if a < -0.3 and v > 0.3:
            return "calma"
        if a > 0.3 and abs(v) < 0.3:
            return "alerta"
        if a < -0.3 and abs(v) < 0.3:
            return "sosiego"
        if abs(a) < 0.3 and v > 0.3:
            return "bienestar"
        if abs(a) < 0.3 and v < -0.3:
            return "displacentero"
        return "neutral"
