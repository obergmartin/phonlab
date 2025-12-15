from PySide6.QtWidgets import QScrollBar, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel

class MyPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Example plot
        self.ax.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.ax.set_xlim(0, 5) # Initial visible range

        self.scrollbar = QScrollBar(self) # Horizontal scrollbar
        self.scrollbar.setOrientation(Qt.Horizontal) # Or QtCore.Qt.Vertical
        self.scrollbar.setMinimum(0)
        self.scrollbar.setMaximum(self.ax.get_xlim()[1] - (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])) # Adjust max to fit visible range
        self.scrollbar.setValue(0) # Initial position
        self.scrollbar.valueChanged.connect(self.update_plot_scroll)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.scrollbar)

        self.scrollbar.valueChanged.connect(self.update_plot_scroll)

    def update_plot_scroll(self, value):
        current_xlim = self.ax.get_xlim()
        visible_range = current_xlim[1] - current_xlim[0]
        new_min = value
        new_max = value + visible_range
        self.ax.set_xlim(new_min, new_max)
        self.canvas.draw_idle() # Redraw the canvas

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My First Qt App")
        self.setGeometry(100, 100, 400, 200) # x, y, width, height
        pltarea = MyPlotWidget()
        hz_layout = QVBoxLayout()
        hz_layout.addWidget(pltarea)

        # label = QLabel("Hello, Qt with Python!", self)
        # label.move(150, 80) # Position the label

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())
