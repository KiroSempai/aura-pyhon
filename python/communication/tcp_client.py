import socket
import struct
import numpy as np
import cv2


class AuraTCPClient:
    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10.0)
        self.socket.connect((self.host, self.port))
        self.connected = True
        self.socket.settimeout(1.0)
        print(f"[TCP] Conectado a Unity en {self.host}:{self.port}")

    def send_action(self, force_x, force_y, force_z, energia=0, integridad=0, curiosidad=0, homeostasis=0, hue=180, saturation=0, lightness=0.5):
        if not self.connected:
            return
        try:
            packet = struct.pack("ffffffffff", force_x, force_y, force_z, energia, integridad, curiosidad, homeostasis, hue, saturation, lightness)
            self.socket.sendall(packet)
        except Exception as e:
            print(f"[TCP] Error enviando accion: {e}")
            self.connected = False

    def _receive_exact(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Conexion perdida")
            data.extend(chunk)
        return bytes(data)

    def receive_sensor_data(self):
        if not self.connected:
            return None
        try:
            img_len_bytes = self._receive_exact(4)
            img_len = struct.unpack("I", img_len_bytes)[0]

            jpeg_data = None
            if img_len > 0:
                jpeg_data = self._receive_exact(img_len)

            pos_data = self._receive_exact(12)
            pos = struct.unpack("fff", pos_data)
            vel_data = self._receive_exact(12)
            vel = struct.unpack("fff", vel_data)

            col_data = self._receive_exact(4)
            impact_force = struct.unpack("f", col_data)[0]
            collided = impact_force > 0.001

            return {
                "jpeg_data": jpeg_data,
                "position": pos,
                "velocity": vel,
                "collided": collided,
                "impact_force": impact_force,
            }
        except socket.timeout:
            return None
        except Exception as e:
            print(f"[TCP] Error recibiendo datos: {e}")
            self.connected = False
            return None

    def decode_image(self, jpeg_data):
        if jpeg_data is None:
            return None
        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def close(self):
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
