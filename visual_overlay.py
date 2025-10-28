import queue
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
        self.commands = queue.Queue()
        self.running = True

    def pump(self):
        if not self.running:
            return
        try:
            window = gw.getWindowsWithTitle(self.target_title)[0]
            self.root.geometry(f"{window.width}x{window.height}+{window.left}+{window.top}")
        except Exception:
            pass
        while not self.commands.empty():
            command, payload = self.commands.get()
            if command == "draw" and payload is not None:
                x, y, signal_type = payload
                color = "green" if signal_type == "CALL" else "red" if signal_type == "PUT" else "gray"
                direction = -1 if signal_type == "CALL" else 1
                size = 20
                points = [x, y, x - size, y + (direction * size), x + size, y + (direction * size)]
                self.canvas.delete("all")
                self.canvas.create_polygon(points, fill=color, outline=color)
            elif command == "clear":
                self.canvas.delete("all")
            elif command == "close":
                self.running = False
                try:
                    self.root.destroy()
                except Exception:
                    pass
                return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.running = False

    def draw_arrow(self, x, y, signal_type):
        self.commands.put(("draw", (x, y, signal_type)))

    def clear(self):
        self.commands.put(("clear", None))

    def close(self):
        self.commands.put(("close", None))
        self.pump()
