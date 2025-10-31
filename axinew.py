# axinew.py - BOTAXINEW V2
# GPT-5 – Nano development version

from PyQt5.QtWidgets import QApplication
import sys

from gui_main import BotAxiGUI


def main():
    app = QApplication(sys.argv)

    gui = BotAxiGUI()     # GUI principal
    gui.show()            # Mostrar ventana

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
