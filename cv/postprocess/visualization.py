import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_trajectory_on_frame(video_path, centroids, states, frames, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir {video_path}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[ERROR] No se pudo leer el primer frame.")
        return
    traj_img = frame.copy()
    cap.release()

    for i in range(1, len(centroids)):
        pt1, pt2 = tuple(map(int, centroids[i-1])), tuple(map(int, centroids[i]))
        color = (0,255,0) if states[i] == "activo" else (0,0,255)
        cv2.line(traj_img, pt1, pt2, color, 3)
        cv2.circle(traj_img, pt2, 4, color, -1)
    cv2.imwrite(output_path, traj_img)

def plot_velocity_over_time(velocities, states, frames, output_path):
    plt.figure(figsize=(10,3))
    plt.plot(frames, velocities, label="Velocidad (px/frame)")
    plt.xlabel("Frame")
    plt.ylabel("Velocidad")
    plt.title("Velocidad por frame")
    for i in range(1, len(frames)):
        color = 'green' if states[i] == "activo" else 'red'
        plt.axvspan(frames[i-1], frames[i], color=color, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
