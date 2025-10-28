import tkinter as tk
from PIL import Image, ImageTk  # noqa: F401 imported for future use
import pygetwindow as gw
import pyautogui


class OverlayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.wm_attributes('-topmost', True)
        self.root.wm_attributes('-transparentcolor', 'black')
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def update_position(self, target_title="IQ Option"):
        try:
            win = gw.getWindowsWithTitle(target_title)[0]
            self.root.geometry(f"{win.width}x{win.height}+{win.left}+{win.top}")
        except Exception:
            pass

    def draw_arrow(self, x, y, signal_type="CALL"):
        color = "green" if signal_type == "CALL" else "red" if signal_type == "PUT" else "gray"
        direction = -1 if signal_type == "CALL" else 1
        size = 20
        points = [x, y, x - size, y + (direction * size), x + size, y + (direction * size)]
        self.canvas.create_polygon(points, fill=color, outline=color)

    def refresh(self):
        self.canvas.delete("all")
        self.root.update_idletasks()
        self.root.update()
