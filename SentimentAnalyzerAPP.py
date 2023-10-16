import sys

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame, QLabel
from PyQt5.QtCore import Qt

class GrayFrame(QFrame):
    def __init__(self):
        super().__init__()

        self.setObjectName("GrayFrame")
        # Applica il colore RGB(34, 34, 34) come sfondo
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(65, 65, 65))
        self.setPalette(palette)
        self.setStyleSheet("""
            QFrame#GrayFrame {
                border-radius: 20px;
            }
        """)
        # Crea un layout per il frame grigio e centra il suo contenuto
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # Aggiungi un QLabel per la scritta in alto
        text_label = QLabel("Hello, World!", self)
        text_label.setStyleSheet("color: white; background-color: transparent;")
        layout.addWidget(text_label)

def main():
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Two Frames Example")
    window.setGeometry(100, 100, 700, 500)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)

    black_frame = QFrame(window)
    black_frame.setObjectName("BlackFrame")
    layout.addWidget(black_frame)

    # Crea un layout per il frame nero e centra il suo contenuto
    black_layout = QVBoxLayout(black_frame)
    black_layout.setAlignment(Qt.AlignCenter)

    gray_frame = GrayFrame()
    black_layout.addWidget(gray_frame)

    window.setStyleSheet("""
        QWidget {
            background-color: rgb(27,27,27);
        }
        #BlackFrame {
            background-color: rgb(27,27,27);
            z-index: 0;
        }
        #GrayFrame {
            background-color: rgb(65,65,65);
            border-radius: 20px;
            z-index: 1;
        }
         #GrayLabel {
            color: white;
            font-size: 18px;
            font-weight: bold;
        }
    """)

    # Imposta le dimensioni esplicite per entrambi i frame
    black_frame.setFixedSize(900, 800)
    gray_frame.setFixedSize(700, 700)

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
