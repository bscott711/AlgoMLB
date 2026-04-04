import numpy as np

MOUND_Y = 60.5

def get_mound_arc_pts(radius, start_ang_deg, end_ang_deg, num=5):
    pts = []
    for i in range(num):
        a_deg = start_ang_deg + (end_ang_deg - start_ang_deg) * i / (num - 1)
        rad = np.radians(a_deg)
        pts.append((radius * np.sin(rad), MOUND_Y + radius * np.cos(rad)))
    return pts

p_f = [(-233.3, 233.3), (233.3, 233.3)]
skin_pts = get_mound_arc_pts(145, -62.3, 62.3)
hub_pts = get_mound_arc_pts(95, -73.0, 73.0)

print("Skin pts x-range:", skin_pts[0][0], "to", skin_pts[-1][0])
print("Hub pts x-range:", hub_pts[0][0], "to", hub_pts[-1][0])

# Just check the start/end points
print(f"Hub Path starts at M 0,15, then L -54,54, then arc starts at {hub_pts[0][0]:.1f}")
