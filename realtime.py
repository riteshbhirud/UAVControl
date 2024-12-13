import serial
from djitellopy import Tello
from collections import deque
import time
import os
import numpy as np
import pickle


class ReadError(Exception):
    pass


labels = {0: 'up', 1: 'down', 2: 'right', 3: 'left', 4: 'front', 5: 'back'}

SAMPLE_FREQUENCY = 100
SAMPLE_DURATION = 2
SAMPLE_TOTAL = SAMPLE_FREQUENCY * SAMPLE_DURATION
WINDOW_SIZE = SAMPLE_TOTAL

# ss = pickle.load(open("processing/scaler.pkl", "rb"))
# model = pickle.load(open("processing/svm.pkl", "rb"))
with open('rfc_project.pkl', 'rb') as f:
    rfc = pickle.load(f)

motion_labels = ["up", "down", "left", "right", "undetected"]

device_names = os.listdir("/dev/")
esp_name = [dev for dev in device_names if dev.startswith("cu.usb")][0]
print("Device Name:", esp_name)

# buffer = [None] * WINDOW_SIZE

acc_info = []
start_value = 0

sliding_window_size = 250
movement_checker = 100

threshold = 4


internal_memory = deque(maxlen=200)


def detect_movement(window):
    if len(window) >= movement_checker:
        energy = np.sum((np.array(list(
            window)[-movement_checker:]) - np.mean(list(window)[-movement_checker:], axis=0)) ** 2)
        if energy > threshold:
            print("started a movement")
            return True
    return False


tello = Tello()

tello.connect()
print("Battery:", tello.get_battery())
tello.takeoff()


def tello_move(label):
    if label == 'up':
        tello.move_up(100)
    if label == 'down':
        tello.move_down(100)
    if label == 'left':
        tello.move_left(100)
    if label == 'right':
        tello.move_right(100)
    if label == 'front':
        tello.move_forward(100)
    if label == 'back':
        tello.move_back(100)
    else:
        raise Exception(f"this should not happpen: {label}")


with serial.Serial("/dev/" + esp_name, baudrate=115200, timeout=1) as esp_serial:
    i = 0
    window = deque(maxlen=sliding_window_size)
    is_movement_start = False
    start_index = 0
    count = 0
    while True:
        try:
            try:
                response = esp_serial.readline().decode()
                values = list(map(float, response.strip().split(", ")))
                assert len(values) == 6
            except Exception:
                raise ReadError("Read Error")

            # buffer[i] = values
            i += 1

            acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z = values

            window.append([acce_x, acce_y, acce_z])

            if not is_movement_start:
                is_movement_start = detect_movement(window)
            else:
                start_index += 1
                if start_index == 100:
                    print("movement end")
                    start_index = 0
                    is_movement_start = False
                    to_measure = np.array(internal_memory).flatten()[
                        :1116].reshape(1, -1)
                    # plot_stuff(np.array(list(window)[-200:]), count)
                    count += 1
                    window.clear()
                    internal_memory.clear()
                    y_pred = rfc.predict(to_measure)
                    print(labels[y_pred[0]])
                    tello_move(labels[y_pred[0]])
                    print("Sleeping for 2 seconds")
                    time.sleep(2)
                    print("Woke up")

            # csv_writer.writerow([timestamp, acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z])
            internal_memory.append(
                [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z])

            # if i >= WINDOW_SIZE:
            #     instance = np.array(buffer)
            #     instance = ss.transform(instance.reshape(1, -1))
            #
            #     pred = model.predict(instance)
            #     print("Prediction:", motion_labels[pred[0]])
            #
            #     buffer = [None] * WINDOW_SIZE
            #     i = 0

        except ReadError:
            pass
