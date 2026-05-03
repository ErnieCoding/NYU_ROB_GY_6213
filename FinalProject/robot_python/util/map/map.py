import matplotlib.pyplot as plt

# =========================
# 1. DEFINE CORNERS
# =========================
corners = {
    "O": (0, 0),
    "P": (48, 0),
    "A": (0, 203),
    "B": (79, 203),
    "C": (79, 263),
    "D": (162, 263),
    "E": (162, 203),
    "F": (245, 203),
    "G": (245, 285),
    "H": (308, 285),
    "I": (308, 122),
    "Y": (274, 122),
    "J": (274, 54),
    "Z": (193, 54),
    "K": (193, 13),
    "L": (130, 13),
    "M": (130, 54),
    "N": (48, 54),
    
}

# LEFT TABLE
corners["Q"] = (48, 105)
corners["S"] = (47, 136)
corners["R"] = (88, 105)
corners["T"] = (87, 136)

# RIGHT TABLE
corners["U"] = (167, 144)
corners["V"] = (167, 108)
corners["X"] = (208, 144)
corners["W"] = (208, 108)

# =========================
# 2. DEFINE WALLS
# =========================
walls = [
    ("O", "P"),
    ("P", "N"),
    ("O", "A"),

    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("E", "F"),
    ("F", "G"),
    ("G", "H"),
    ("H", "I"),

    ("I", "Y"),
    ("Y", "J"),
    ("J", "Z"),
    ("Z", "K"),
    ("K", "L"),
    ("L", "M"),
    ("M", "N"),

    #first inner table:
    ("Q", "S"),
    ("S", "T"),
    ("T", "R"),
    ("R", "Q"),


    #second  inner table:
    ("U", "V"),
    ("V", "W"),
    ("W", "X"),
    ("X", "U"),
]

# =========================
# 3. DEFINE TAGS
# =========================
tags = {
    # From O
    0: (corners["O"][0] + 20, corners["O"][1]),
    1: (corners["O"][0], corners["O"][1] + 94),

    # From A
    3: (corners["A"][0] + 12, corners["A"][1]),
    2: (corners["A"][0], corners["A"][1] - 30),

    # From T
    4: (corners["T"][0], corners["T"][1] - 17),

    # From C
    5: (corners["C"][0] + 48, corners["C"][1]),

    # From D
    6: (corners["D"][0], corners["D"][1] - 31),

    # From U
    7: (corners["U"][0], corners["U"][1] - 18),

    # From G
    8: (corners["G"][0] + 30, corners["G"][1]),

    # From H
    9: (corners["H"][0], corners["H"][1] - 91),

    # From J
    10: (corners["J"][0], corners["J"][1] + 54),
    11: (corners["J"][0] - 32, corners["J"][1]),

    # From K
    12: (corners["K"][0] - 30, corners["K"][1]),
    14: (corners["K"][0], corners["K"][1] + 32),

    # From L
    13: (corners["L"][0], corners["L"][1] + 30),

    # From Q 
    15: (corners["Q"][0] + 20, corners["Q"][1]),
}

# =========================
# 4. PLOT MAP
# =========================
plt.figure(figsize=(8, 10))

# ---- Draw walls ----
for w in walls:
    x1, y1 = corners[w[0]]
    x2, y2 = corners[w[1]]
    plt.plot([x1, x2], [y1, y2], color='black', linewidth=2)

# ---- Draw corners ----
for name, (x, y) in corners.items():
    plt.scatter(x, y, color='blue', s=30)
    
    # Corner label
    plt.text(x + 2, y + 2, name, fontsize=8, color='blue')
    
    # Coordinate label (smaller, slightly offset)
    plt.text(x + 2, y - 6, f"({int(x)}, {int(y)})", fontsize=6, color='gray')

# ---- Draw tags ----
for tag_id, (x, y) in tags.items():
    plt.scatter(x, y, color='red', s=50, marker='x')
    
    # Tag label
    plt.text(x + 2, y + 2, f"T{tag_id}", fontsize=8, color='red')
    
    # Coordinate label (smaller)
    plt.text(x + 2, y - 6, f"({int(x)}, {int(y)})", fontsize=6, color='darkred')

# ---- Legend (clean manual legend) ----
plt.scatter([], [], color='blue', label='Corners')
plt.scatter([], [], color='red', marker='x', label='ArUco Tags')
plt.legend(loc='upper left')

# ---- Styling ----
plt.title("Map Layout with Corners and ArUco Tags")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)

# ---- Auto bounds ----
all_x = [p[0] for p in corners.values()] + [p[0] for p in tags.values()]
all_y = [p[1] for p in corners.values()] + [p[1] for p in tags.values()]

plt.xlim(min(all_x) - 20, max(all_x) + 20)
plt.ylim(min(all_y) - 20, max(all_y) + 20)

plt.show()