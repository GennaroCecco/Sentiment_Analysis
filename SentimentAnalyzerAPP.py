import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt  # Aggiungi questa importazione
class RedFrame(QFrame):
    def __init__(self):
        super().__init__()

        self.setObjectName("RedFrame")

        self.setStyleSheet("""
            QFrame {
                background-color: red;
                border-radius: 20px;
            }
        """)

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

    red_frame = RedFrame()
    black_layout.addWidget(red_frame)

    window.setStyleSheet("""
        QWidget {
            background-color: black;
        }
        #BlackFrame {
            background-color: black;
            z-index: 0;
        }
        #RedFrame {
            background-color: red;
            border-radius: 20px;
            z-index: 1;
        }
    """)

    # Imposta le dimensioni esplicite per entrambi i frame
    black_frame.setFixedSize(900, 800)
    red_frame.setFixedSize(700, 700)

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
