import time
import numpy as np
from communication.tcp_client import AuraTCPClient
from brain.homeostasis import Homeostasis
from brain.emotional_space import EmotionalSpace
from brain.cnn import AuraCNN
from brain.dqn import DQNAgent
from brain.replay_memory import ReplayMemory
from brain.babbling import Babbling

STATE_DIM = 37
ACTION_DIM = 5


class Aura:
    def __init__(self):
        self.body = AuraTCPClient()
        self.brain_cnn = AuraCNN()
        self.homeostasis = Homeostasis()
        self.emotions = EmotionalSpace()
        self.babbling = Babbling()
        self.agent = DQNAgent(STATE_DIM, ACTION_DIM)
        self.memory = ReplayMemory(50000)
        self.cycle_count = 0
        self.running = False
        self.last_pos = np.array([0.0, 0.0, 0.0])
        self.last_hs = self.homeostasis.get_state()
        self.idle_cycles = 0
        self.move_dir = np.array([1.0, 0.0])
        self.move_timer = 0
        self.last_time = time.time()

    def start(self):
        print("[AURA] Iniciando sistema Fase 1...")
        time.sleep(0.5)
        self.body.connect()
        self.running = True
        print("[AURA] Sistema operativo. Aprendizaje activo.")

    def _build_state(self, cnn_features, hs_state, emotion):
        if cnn_features is None:
            cnn_features = np.zeros(32, dtype=np.float32)
        hs_vec = np.array([hs_state["energia"], hs_state["integridad"], hs_state["curiosidad"]], dtype=np.float32)
        emo_vec = np.array([emotion["arousal"], emotion["valence"]], dtype=np.float32)
        return np.concatenate([cnn_features, hs_vec, emo_vec]).astype(np.float32)

    def _calc_reward(self, hs_state, collided, prev_hs):
        reward = 0.0
        if hs_state["homeostasis"] > 0.7:
            reward += 0.02
        elif hs_state["homeostasis"] < 0.3:
            reward -= 0.05
        delta_h = hs_state["homeostasis"] - prev_hs["homeostasis"]
        reward += delta_h * 2.0
        delta_e = hs_state["energia"] - prev_hs["energia"]
        if delta_e > 0:
            reward += delta_e * 3.0
        if collided:
            reward -= 0.15
        if self.idle_cycles > 300:
            reward -= 0.02
        return np.clip(reward, -1.0, 1.0)

    def _decide_action(self, hs_state, emotion, vel):
        if hs_state["energia"] < 0.2:
            self.move_dir = np.array([0.0, 0.0])
            return 0.0, 0.0, 0.0

        self.move_timer += 1
        if self.move_timer > 30 or np.linalg.norm(self.move_dir) < 0.1:
            self.move_dir += np.random.uniform(-0.3, 0.3, 2)
            self.move_dir = np.clip(self.move_dir, -1.0, 1.0)
            if np.linalg.norm(self.move_dir) > 1.0:
                self.move_dir = self.move_dir / np.linalg.norm(self.move_dir)
            self.move_timer = 0

        self.move_dir += np.random.uniform(-0.05, 0.05, 2)
        self.move_dir = np.clip(self.move_dir, -1.0, 1.0)
        if np.linalg.norm(self.move_dir) > 1.0:
            self.move_dir = self.move_dir / np.linalg.norm(self.move_dir)

        fx = self.move_dir[0] * 2.0
        fy = self.move_dir[1] * 2.0
        fz = 0.0

        if hs_state["energia"] > 0.5 and hs_state["curiosidad"] > 0.5:
            fx *= 1.8
            fy *= 1.8

        if hs_state["energia"] < 0.4:
            fx *= 0.5
            fy *= 0.5

        if emotion["quadrant"] == "depresion":
            fx *= 0.3
            fy *= 0.3
        elif emotion["quadrant"] == "entusiasmo" or emotion["quadrant"] == "alerta":
            fx *= 1.5
            fy *= 1.5

        return fx, fy, fz

    def _action_to_vector(self, dqn_action):
        dirs = {
            0: np.array([0.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([-0.5, 0.0]),
            3: np.array([0.0, -1.0]),
            4: np.array([0.0, 1.0]),
        }
        return dirs.get(dqn_action, np.array([0.0, 0.0]))

    def cycle(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        dt = max(min(dt, 0.1), 0.001)

        sensor = self.body.receive_sensor_data()

        collided = False
        impact_force = 0.0
        speed = 0.0
        moving = False
        significant_movement = False
        vel = np.array([0.0, 0.0, 0.0])

        if sensor is not None:
            pos = np.array(sensor["position"])
            vel = np.array(sensor["velocity"])
            collided = sensor["collided"]
            impact_force = sensor["impact_force"]
            speed = float(np.linalg.norm(vel))
            moving = speed > 0.01
            novelty = np.linalg.norm(pos - self.last_pos)
            significant_movement = novelty > 0.3
            self.last_pos = pos.copy()

        hs_state = self.homeostasis.update(
            dt=dt, moving=moving, collided=collided, impact_force=impact_force,
            speed=speed, significant_movement=significant_movement
        )
        emotion = self.emotions.update(hs_state)

        state = self._build_state(None, hs_state, emotion)

        if self.agent.epsilon > 0.3:
            fx, fy, fz = self._decide_action(hs_state, emotion, vel)
        else:
            dqn_action = self.agent.act(state, force_exploit=False)
            vec = self._action_to_vector(dqn_action)
            fx = vec[0] * 2.0
            fy = vec[1] * 2.0
            fz = 0.0

        if hs_state["energia"] < 0.2:
            fx, fy, fz = 0.0, 0.0, 0.0

        if moving:
            self.idle_cycles = 0
        else:
            self.idle_cycles += 1

        reward = self._calc_reward(hs_state, collided, self.last_hs)

        next_state = self._build_state(None, hs_state, emotion)
        done = hs_state["homeostasis"] <= 0.0
        dqn_action_taken = self._force_to_action(fx, fy)
        self.memory.push(state, dqn_action_taken, reward, next_state, done)
        self.last_hs = hs_state

        if self.cycle_count % 2 == 0:
            loss = self.agent.train(self.memory, 32)
            if self.cycle_count % 500 == 0 and loss > 0:
                print(f"  -> Loss:{loss:.4f} Eps:{self.agent.epsilon:.3f}")

        self.body.send_action(fx, fy, fz,
                              energia=hs_state['energia'],
                              integridad=hs_state['integridad'],
                              curiosidad=hs_state['curiosidad'],
                              homeostasis=hs_state['homeostasis'],
                              hue=emotion['hue'],
                              saturation=emotion['saturation'],
                              lightness=emotion['lightness'])

        self.babbling.update(self.cycle_count)
        self.cycle_count += 1

        if self.cycle_count % 100 == 0:
            print(f"[AURA] Ciclo {self.cycle_count} | E:{hs_state['energia']:.2f} "
                  f"I:{hs_state['integridad']:.2f} C:{hs_state['curiosidad']:.2f} "
                  f"H:{hs_state['homeostasis']:.2f} | {emotion['quadrant']}")

    def _force_to_action(self, fx, fy):
        mag = np.sqrt(fx*fx + fy*fy)
        if mag < 0.1:
            return 0
        angle = np.arctan2(fy, fx)
        if -0.5 < angle < 0.5:
            return 1
        if angle > 2.0 or angle < -2.0:
            return 2
        if angle < -0.5:
            return 3
        return 4

    def run(self):
        self.start()
        try:
            while self.running:
                if not self.body.connected:
                    print("[AURA] Reconectando...")
                    try:
                        self.body.connect()
                    except:
                        time.sleep(1)
                        continue
                self.cycle()
                time.sleep(0.033)
        except KeyboardInterrupt:
            print("\n[AURA] Deteniendo sistema...")
        finally:
            self.body.close()
            print(f"[AURA] Sistema detenido. Ciclos: {self.cycle_count}")


if __name__ == "__main__":
    aura = Aura()
    aura.run()
