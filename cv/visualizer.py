import cv2

def draw_boxes(frame, tracks, color=(0,255,0)):
    for t in tracks:
        x1, y1, x2, y2, tid = map(int, t)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
