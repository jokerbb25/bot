import threading
import tkinter as tk

import pygetwindow as gw


class OverlayWindow:
    def __init__(self, target_title="IQ Option"):
        self.target_title = target_title
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", "black")
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.running = True
        self.root.after(50, self.update_position)
        threading.Thread(target=self._start_mainloop, daemon=True).start()

    def _start_mainloop(self):
        try:
            self.root.mainloop()
        except tk.TclError:
            pass

    def update_position(self):
        try:
            window = gw.getWindowsWithTitle(self.target_title)[0]
            self.root.geometry(f"{window.width}x{window.height}+{window.left}+{window.top}")
        except Exception:
            pass
        if self.running:
            self.root.after(100, self.update_position)

    def draw_arrow(self, x, y, signal_type="CALL"):
        color = "green" if signal_type == "CALL" else "red" if signal_type == "PUT" else "gray"
        direction = -1 if signal_type == "CALL" else 1
        size = 20
        points = [x, y, x - size, y + (direction * size), x + size, y + (direction * size)]

        def _draw():
            self.canvas.create_polygon(points, fill=color, outline=color)

        self.root.after(0, _draw)

    def clear(self):
        self.root.after(0, self.canvas.delete, "all")

    def close(self):
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass
