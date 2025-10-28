import queue
import time
import tkinter as tk

import pygetwindow as gw

try:
    import win32gui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for Windows only
    win32gui = None


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
        self.chart_rect = (0, 0, 0, 0)
        self.chart_x_offset = 0
        self.chart_y_offset = 0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.last_drawn_key = None
        self._last_geometry_update = 0.0
        self.window_position = (0, 0)

    def pump(self):
        if not self.running:
            return
        now = time.time()
        if now - self._last_geometry_update > 0.5:
            self._update_geometry()
            self._last_geometry_update = now
        while not self.commands.empty():
            command, payload = self.commands.get()
            try:
                if command == "draw" and payload is not None:
                    rel_x, rel_y, signal_type, cache_key = payload
                    if cache_key and cache_key == self.last_drawn_key:
                        continue
                    position = self._translate(rel_x, rel_y)
                    if position is None:
                        continue
                    x, y = position
                    color = "green" if signal_type == "CALL" else "red" if signal_type == "PUT" else "gray"
                    direction = -1 if signal_type == "CALL" else 1
                    size = 20
                    points = [x, y, x - size, y + (direction * size), x + size, y + (direction * size)]
                    self.canvas.delete("all")
                    self.canvas.create_polygon(points, fill=color, outline=color)
                    self.last_drawn_key = cache_key
                elif command == "clear":
                    self.canvas.delete("all")
                    self.last_drawn_key = None
                elif command == "close":
                    self.running = False
                    try:
                        self.root.destroy()
                    except Exception:
                        pass
                    return
            except Exception:
                continue
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.running = False

    def draw_arrow(self, rel_x, rel_y, signal_type, cache_key=None):
        self.commands.put(("draw", (rel_x, rel_y, signal_type, cache_key)))

    def clear(self):
        self.commands.put(("clear", None))

    def close(self):
        self.commands.put(("close", None))
        self.pump()

    def _update_geometry(self):
        geometry = None
        if win32gui is not None:
            try:
                hwnd = win32gui.FindWindow(None, self.target_title)
                if hwnd:
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    geometry = (left, top, right, bottom)
            except Exception:
                geometry = None
        if geometry is None:
            try:
                window = gw.getWindowsWithTitle(self.target_title)[0]
                geometry = (window.left, window.top, window.right, window.bottom)
            except Exception:
                geometry = None
        if geometry is None:
            return
        left, top, right, bottom = geometry
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return
        chart_left = left + int(width * 0.22)
        chart_top = top + int(height * 0.12)
        chart_width = int(width * 0.58)
        chart_height = int(height * 0.75)
        self.chart_rect = (chart_left, chart_top, chart_width, chart_height)
        self.window_position = (left, top)
        self.root.geometry(f"{width}x{height}+{left}+{top}")

    def _translate(self, rel_x, rel_y):
        chart_left, chart_top, chart_width, chart_height = self.chart_rect
        if chart_width <= 0 or chart_height <= 0:
            return None
        try:
            rel_x = max(0.0, min(1.0, float(rel_x)))
            rel_y = max(0.0, min(1.0, float(rel_y)))
        except (TypeError, ValueError):
            return None
        window_left, window_top = self.window_position
        x = chart_left + int(chart_width * rel_x) - window_left
        y = chart_top + int(chart_height * rel_y) - window_top
        x = int((x + self.chart_x_offset) * self.scale_x)
        y = int((y + self.chart_y_offset) * self.scale_y)
        return x, y
