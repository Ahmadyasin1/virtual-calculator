# --- 🔰 Library Imports ---
import cv2                  # For webcam access and image display
import mediapipe as mp      # For hand tracking using MediaPipe
import time                 # For time-based delays (like click cooldown)
import math                 # For mathematical operations (e.g., sqrt, sin)
import re                   # For regular expressions (used in expression parsing)
import numpy as np          # For numerical array operations (used to create popups)

# --- ✋ Hand Tracking Setup ---
mp_hands = mp.solutions.hands       # Initializes the hand tracking model.
hands = mp_hands.Hands(
    static_image_mode=False,        # static_image_mode=False: Real-time video.
    max_num_hands=1,                # max_num_hands=1: Track only 1 hand.
    min_detection_confidence=0.8,   # Detection & tracking confidences must be > 0.8 to ensure accuracy.
    min_tracking_confidence=0.8
)

# --- 🔘 Button Class ---
# A custom class to manage buttons for the calculator interface.
class Button:
    def __init__(self, pos, text, size=(100, 100), color=(60, 60, 200)):
        self.pos = pos # pos: Top-left (x, y) position.
        self.size = size # size: (width, height) of the button.
        self.text = text # text: Label (e.g., '1', '+').
        self.color = color # color: RGB color of the button.
        self.last_click = 0 # last_click: Timestamp of the last click.
        self.active = False # active: Indicates if the button is currently pressed.
        self.hover = False # hover: Indicates if the button is currently hovered over.
        self.font = cv2.FONT_HERSHEY_SIMPLEX # font: Font type for button text.

    def draw(self, img):
        x, y = self.pos
        w, h = self.size

        # Button glow effect: adjust brightness based on hover/active state
        base_color = tuple(min(255, c + (40 if self.hover else 0) - (40 if self.active else 0)) for c in self.color)

        # Shadow effect for a 3D look
        shadow_offset = 4
        cv2.rectangle(img, (x + shadow_offset, y + shadow_offset), (x + w + shadow_offset, y + h + shadow_offset), (30, 30, 30), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), base_color, cv2.FILLED)

        # Draw border with a glow effect on hover
        border_color = (255, 255, 255) if self.hover else (120, 120, 120)
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2)

        # Draw text with dynamic font scaling
        font_scale = 0.7 if len(self.text) > 2 else 1
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    def is_clicked(self, cursor_pos, cooldown=0.5):
        x, y = self.pos
        w, h = self.size
        in_bound = x < cursor_pos[0] < x + w and y < cursor_pos[1] < y + h
        ready = (time.time() - self.last_click) > cooldown
        return in_bound and ready

# --- 📜 Instructions Popup ---
def show_instructions():
    instructions = [
        "Welcome to the Futuristic Virtual Calculator!",
        "",
        "Instructions:",
        "- Hover with your index finger.",
        "- Pinch with your middle finger to press.",
        "",
        "Supported functions and constants:",
        "+, -, *, /, ^, sqrt, sin, cos, tan, log, ln, %",
        "pi, e, (, )",
        "",
        "Use 'C' to clear and '<-' to backspace.",
        "",
        "Press any key to start..."
    ]
    popup = 255 * np.ones((600, 800, 3), dtype=np.uint8)
    y0, dy = 40, 40
    for i, line in enumerate(instructions):
        y = y0 + i * dy
        cv2.putText(popup, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imshow("Instructions", popup)
    cv2.waitKey(0)
    cv2.destroyWindow("Instructions")

# --- 🎛️ Initialize Buttons ---
# Updated key layout including constants (pi, e)
keys = [
    ['7', '8', '9', '/', 'sqrt'],
    ['4', '5', '6', '*', '^'],
    ['1', '2', '3', '-', '<-'],
    ['0', '.', '=', '+', 'C'],
    ['(', ')', '%', 'log', 'ln'],
    ['pi', 'e', '', '', ''],
    ['sin', 'cos', 'tan', '', '']
]
buttons = []
sx, sy = 50, 170  # starting x,y position
bw, bh = 100, 100  # button width and height
gap = 15           # gap between buttons

for i, row in enumerate(keys):
    for j, key in enumerate(row):
        if key == '':  # skip empty cells
            continue
        x = sx + j * (bw + gap)
        y = sy + i * (bh + gap)
        # Determine color based on function or number
        is_op = re.match(r'[+\-*/^%()]|log|ln|sqrt|sin|cos|tan|C|<-|=', key)
        color = (0, 120, 215) if is_op else (60, 200, 100)
        buttons.append(Button((x, y), key, (bw, bh), color))

# --- 🔢 Expression Handling ---
expr = ""
result = ""
show_instructions()

# --- 🎥 Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- 🔄 Main Loop ---
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Reset button states for the new frame
    for btn in buttons:
        btn.hover = False
        btn.active = False

    # --- 🖐️ Hand Landmark Processing ---
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        ix = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
        iy = int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
        mx = int(hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w)
        my = int(hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h)
        pinch_dist = math.hypot(ix - mx, iy - my)

        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        # --- 🖲️ Button Interactions ---
        for btn in buttons:
            if btn.is_clicked((ix, iy), 0.5):
                btn.hover = True
                if pinch_dist < 40:
                    btn.last_click = time.time()
                    key = btn.text
                    btn.active = True
                    # --- 🧮 Button Functionalities ---
                    if key == 'C':
                        expr = ""
                        result = ""
                    elif key == '<-':
                        expr = expr[:-1]
                    elif key == '=':
                        try:
                            # Preprocess the expression:
                            safe_expr = expr.replace('^', '**')
                            # Replace textual functions and constants with math functions/constants
                            safe_expr = safe_expr.replace('sqrt', 'math.sqrt')
                            mapping = {
                                'sin': 'math.sin',
                                'cos': 'math.cos',
                                'tan': 'math.tan',
                                'log': 'math.log10',
                                'ln': 'math.log',
                                'pi': 'math.pi',
                                'e': 'math.e'
                            }
                            pattern = r'sin|cos|tan|log|ln|pi|e'
                            safe_expr = re.sub(pattern, lambda m: mapping[m.group()], safe_expr)
                            safe_expr = safe_expr.replace('%', '/100')
                            # Remove any disallowed characters
                            safe_expr = re.sub(r'[^0-9+\-*/().a-zA-Z_]', '', safe_expr)
                            result = str(round(eval(safe_expr, {"math": math, "__builtins__": None}), 10))
                            expr = result
                        except Exception:
                            result = "Error"
                    else:
                        expr += key

    # --- 🖼️ Drawing Display + Buttons ---
    # Draw display area background
    cv2.rectangle(frame, (sx, 30), (sx + 5*(bw + gap) - gap, 150), (20, 20, 30), cv2.FILLED)
    cv2.putText(frame, f"Expr: {expr}", (sx + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 255), 2)
    cv2.putText(frame, f"Result: {result}", (sx + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)

    # Draw all buttons
    for btn in buttons:
        btn.draw(frame)
    # --- 🧭 Exit and Cleanup ---
    cv2.imshow("Futuristic Calculator", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
