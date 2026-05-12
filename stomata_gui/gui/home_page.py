from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap


class HomePage(QWidget):

    def __init__(self, model_path, on_change_model, on_open_record):
        super().__init__()

        self.model_path = model_path
        self.on_change_model = on_change_model
        self.on_open_record = on_open_record

        self._logo_pix = QPixmap("images/UI/qiushu.png")

        self._intro_pix = QPixmap(r"I:\05-全同胞气孔标记\气孔小论文上传代码\ultralytics-yolo11-main-stomata\stomata_gui\images\tmp\homepage.png")

        self._build_ui()

    def _build_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(10)

        title = QLabel("Catalpa Bungei Stomatal Detection System")
        title.setFont(QFont("KaiTi", 20))
        title.setAlignment(Qt.AlignCenter)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)

        self.intro_label = QLabel()
        self.intro_label.setAlignment(Qt.AlignCenter)

        self.intro_label.setFixedSize(1000, 600)

        bottom_layout = QHBoxLayout()

        self.model_label = QLabel(f"Current model：{self.model_path}")
        self.model_label.setFont(QFont("KaiTi", 13))

        change_btn = QPushButton("Switch Model")
        record_btn = QPushButton("View Results")

        change_btn.clicked.connect(self.on_change_model)
        record_btn.clicked.connect(self.on_open_record)

        bottom_layout.addWidget(self.model_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(change_btn)
        bottom_layout.addWidget(record_btn)

        main_layout.addWidget(title)
        main_layout.addWidget(self.logo_label)
        main_layout.addWidget(self.intro_label, alignment=Qt.AlignCenter)
        main_layout.addLayout(bottom_layout)

        self._init_images()

    def _init_images(self):

        if not self._logo_pix.isNull():
            self.logo_label.setPixmap(
                self._logo_pix.scaledToHeight(
                    100, Qt.SmoothTransformation
                )
            )

        if not self._intro_pix.isNull():
            self.intro_label.setPixmap(
                self._intro_pix.scaled(
                    self.intro_label.width(),
                    self.intro_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def update_model_path(self, new_path):
        self.model_label.setText(f"Current model：{new_path}")