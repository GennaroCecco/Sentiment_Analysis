import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextBrowser, \
    QWidget
from RunLSTM import RNNLSTM


class SentimentAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sentiment Analyzer App')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton('Load CSV File')
        self.load_button.clicked.connect(self.load_csv)

        self.load_button2 = QPushButton('Example run')
        self.load_button2.clicked.connect(self.example)

        self.results_browser = QTextBrowser()

        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.load_button2)
        self.layout.addWidget(self.results_browser)
        self.central_widget.setLayout(self.layout)

        self.sentiment_analyzer = RNNLSTM()

    def load_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        csv_file, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)

        if csv_file:
            try:
                self.sentiment_analyzer.load_tweets_from_csv(csv_file)
                results = self.sentiment_analyzer.analyze_sentiments()

                self.results_browser.clear()
                for tweet, sentiment in results:
                    self.results_browser.append(f"Tweet: {tweet}\nSentiment: {sentiment}\n")
            except Exception as e:
                self.results_browser.clear()
                self.results_browser.append("Error loading or analyzing sentiments: " + str(e))

    def example(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            result = self.sentiment_analyzer.analyze_sentiments()
            self.results_browser.clear()
            for tweet, sentiment in result:
                self.results_browser.append(f"Tweet: {tweet}\nSentiment: {sentiment}\n")
        except Exception as e:
            self.results_browser.clear()
            self.results_browser.append("Error analyzing sentiments: " + str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SentimentAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())
