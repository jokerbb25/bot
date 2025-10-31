from PyQt5.QtWidgets import QApplication
import sys
from gui_main import BotAxiGUI

def main():
    app = QApplication(sys.argv)
    gui = BotAxiGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
