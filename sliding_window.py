from djitellopy import Tello
import pickle
from collections import deque
import numpy as np
import os
import serial
import time
from scipy.signal import butter, lfilter

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
TIMEOUT = 20

tello = Tello()

tello.connect()
print("Battery:", tello.get_battery())
tello.takeoff()


MOVEMENT_SIZE = 50


def tello_move(label):
    if label == "up":
        tello.move_up(MOVEMENT_SIZE)
    elif label == "down":
        tello.move_down(MOVEMENT_SIZE)
    elif label == "left":
        tello.move_left(MOVEMENT_SIZE)
    elif label == "right":
        tello.move_right(MOVEMENT_SIZE)
    elif label == "front":
        tello.move_forward(MOVEMENT_SIZE)
    elif label == "back":
        tello.move_back(MOVEMENT_SIZE)
    else:
        raise Exception(f"this should not happpen: {label}")


# Open serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
print(ser)

folder = "output"

print(os.curdir)

labels = {0: 'up', 1: 'down', 2: 'right', 3: 'left', 4: 'front', 5: 'back'}

new_file = ""
new_file = os.path.join(folder, new_file)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def apply_lowpass_filter(data, cutoff=3, fs=50):
    # Create an array to store filtered results
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):      # Loop over each column (feature)
        filtered_data[:, i] = butter_lowpass_filter(
            data[:, i], cutoff=cutoff, fs=fs)
    return filtered_data


print("Listening to serial port...")

acc_info = []
start_value = 0

sliding_window_size = 250
movement_checker = 100

curr = time.time()
threshold = 3.5

with open('svm_project.pkl', 'rb') as f:
    svm = pickle.load(f)


def detect_movement(window):
    if len(window) >= movement_checker:
        energy = np.sum((np.array(list(
            window)[-movement_checker:]) - np.mean(list(window)[-movement_checker:], axis=0)) ** 2)
        if energy > threshold:
            print("started a movement")
            return True
    return False


internal_memory = deque(maxlen=200)


feature_columns = ['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']


def normalize_test_data(test_data):
    # Minimum value for each column (feature)
    min_values = np.min(test_data, axis=0)
    # Maximum value for each column (feature)
    max_values = np.max(test_data, axis=0)
    normalized_data = (test_data - min_values) / (max_values - min_values)
    return normalized_data


try:
    window = deque(maxlen=sliding_window_size)
    is_movement_start = False
    start_index = 0
    count = 0
    while True:
        if ser.in_waiting > 0:
            serial_data = ser.readline().decode('utf-8').strip()

            if "acce_x" not in serial_data:
                continue

            def process_number(serial_data):
                new_thing = []
                for i in serial_data:
                    new_thing.append(float(i[i.find(":") + 1:]))
                return new_thing

            acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z = process_number(
                serial_data.split(","))

            timestamp = start_value
            start_value += 0.01
            acc_info.append([timestamp, acce_x, acce_y, acce_z])
            window.append([acce_x, acce_y, acce_z])

            if not is_movement_start:
                is_movement_start = detect_movement(window)
            else:
                start_index += 1

                if start_index == 100:
                    print("movement end")
                    start_index = 0
                    is_movement_start = False
                    to_measure = apply_lowpass_filter(normalize_test_data(
                        np.array(internal_memory))).flatten()[:1116].reshape(1, -1)
                    count += 1
                    window.clear()
                    internal_memory.clear()

                    y_pred = svm.predict(to_measure)
                    print(labels[y_pred[0]])
                    tello_move(labels[y_pred[0]])

            internal_memory.append(
                [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z])

except KeyboardInterrupt:
    print("Terminating script...")
finally:
    ser.close()
    timestamps = [row[0] for row in acc_info]
    acce_x = [row[1] for row in acc_info]
    acce_y = [row[2] for row in acc_info]
    acce_z = [row[3] for row in acc_info]

    cutoff = 10
    fs = 100

    print("Serial port and file closed.")
