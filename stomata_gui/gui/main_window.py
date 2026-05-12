import os
from PySide6.QtWidgets import QMainWindow, QTabWidget, QFileDialog
from PySide6.QtGui import QIcon

from gui.home_page import HomePage
from gui.image_detect_page import ImageDetectPage
from core.stomata_detector import StomataDetector


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Stomata detection system")
        self.setWindowIcon(QIcon(r"images\tmp\picture.png"))
        self.resize(1200, 800)

        self.model_path = (
            "E:/2025 Work Priorities/12-stomata/02-data/16-model/"
            "yolo11-crack-GUI/42_demo/runs/exp37/weights/best.pt"
        )

        self.detector = StomataDetector(self.model_path)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.home_page = HomePage(
            self.model_path,
            self.change_model,
            self.open_record
        )

        self.image_page = ImageDetectPage(self.detector)

        self.tabs.addTab(self.home_page, "主页")
        self.tabs.addTab(self.image_page, "图片检测")

    def change_model(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型",
            "",
            "*.pt"
        )

        if not file_path:
            return

        self.detector.switch_model(file_path)
        self.model_path = file_path

        self.home_page.update_model_path(file_path)

    def open_record(self):
        os.startfile(os.path.abspath("record"))