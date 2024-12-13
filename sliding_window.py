import pickle
from collections import deque
import numpy as np
import os
import serial
import time
from scipy.signal import butter, lfilter

BAUD_RATE = 115200
TIMEOUT = 1

device_names = os.listdir("/dev/")
esp_name = [dev for dev in device_names if dev.startswith("cu.usb")][0]

print("Device Name:", esp_name)
SERIAL_PORT = esp_name

# Open serial connection
ser = serial.Serial("/dev/" + SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

folder = "output"
type = "stationary_result"

print(os.curdir)

labels = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}

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


print("Listening to serial port...")

acc_info = []
start_value = 0

sliding_window_size = 250
movement_checker = 100

curr = time.time()
threshold = 4

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


def get_type(window):
    if len(window) >= movement_checker:
        energy = np.sum((np.array(list(
            window)[-movement_checker:]) - np.mean(list(window)[-movement_checker:])) ** 2)
        if energy > threshold:
            print("started a movement")
            return True
    return False


internal_memory = deque(maxlen=200)


def plot_stuff(data, count):
    acce_x = data[:, 0]
    acce_y = data[:, 1]
    acce_z = data[:, 2]
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(200), acce_x, label='acce_x', marker='o')
    # plt.plot(range(200), acce_y, label='acce_y', marker='o')
    # plt.plot(range(200), acce_z, label='acce_z', marker='o')

    # # Add labels and title
    # plt.xlabel('Time (or index)')
    # plt.ylabel('Acceleration')
    # plt.title('Acceleration Components Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{count}.png")


try:
    window = deque(maxlen=sliding_window_size)
    is_movement_start = False
    start_index = 0
    count = 0
    while True:
        if ser.in_waiting > 0:
            serial_data = ser.readline().decode('utf-8').strip()
            print(serial_data)

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
                    to_measure = np.array(internal_memory).flatten()[
                        :1116].reshape(1, -1)
                    plot_stuff(np.array(list(window)[-200:]), count)
                    count += 1
                    window.clear()
                    internal_memory.clear()
                    y_pred = svm.predict(to_measure)
                    print(labels[y_pred[0]])

            # csv_writer.writerow([timestamp, acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z])
            internal_memory.append(
                [acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z])

except KeyboardInterrupt:
    print("Terminating script...")
finally:
    ser.close()
    # csv_file.close()
    timestamps = [row[0] for row in acc_info]
    acce_x = [row[1] for row in acc_info]
    acce_y = [row[2] for row in acc_info]
    acce_z = [row[3] for row in acc_info]

    cutoff = 10
    fs = 100

    print("Serial port and file closed.")
