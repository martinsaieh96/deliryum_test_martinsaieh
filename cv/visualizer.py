import cv2

def draw_boxes(frame, tracks, person_data, frame_idx, color_map=None):
    import cv2
    out = frame.copy()
    color_map = color_map or {}

    for t in tracks:
        x1, y1, x2, y2, tid = map(int, t)
        try:
            state = person_data[tid]["states"][person_data[tid]["frames"].index(frame_idx)]
            color = (0,255,0) if state == "activo" else (0,0,255)
        except:
            color = (255,255,0)
            state = "N/A"
        try:
            speed = person_data[tid]["velocities"][person_data[tid]["frames"].index(frame_idx)]
        except:
            speed = 0.0

        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"ID:{tid} {state} v={speed:.1f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        pts = [c for i,c in enumerate(person_data[tid]["centroids"]) if person_data[tid]["frames"][i] <= frame_idx]
        for i in range(1, len(pts)):
            cv2.line(out, pts[i-1], pts[i], color, 2)

    return out
