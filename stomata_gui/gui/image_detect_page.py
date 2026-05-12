import os
import shutil
import copy
import cv2

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont

from core.stomata_detector import StomataDetector


class ImageDetectPage(QWidget):

    def __init__(self, detector: StomataDetector, output_size=480):
        super().__init__()

        self.detector = detector
        self.output_size = output_size

        # 当前图像路径
        self.current_image = None

        self.conf = 0.25
        self.iou = 0.45

        self._build_ui()

    def _build_ui(self):

        main_layout = QVBoxLayout(self)

        title = QLabel("Single Image Stomatal Detection")
        title.setFont(QFont("KaiTi", 16))
        title.setAlignment(Qt.AlignCenter)

        img_layout = QHBoxLayout()

        self.src_label = QLabel()
        self.dst_label = QLabel()

        self.src_label.setAlignment(Qt.AlignCenter)
        self.dst_label.setAlignment(Qt.AlignCenter)

        left_box = QGroupBox("Original Image")
        right_box = QGroupBox("Detection Result")

        l = QVBoxLayout(left_box)
        r = QVBoxLayout(right_box)

        l.addWidget(self.src_label)
        r.addWidget(self.dst_label)

        img_layout.addWidget(left_box)
        img_layout.addWidget(right_box)

        # ---------- 结果 ----------
        self.result_label = QLabel("Detection result: not detected")
        self.result_label.setFont(QFont("KaiTi", 13))

        # ---------- 按钮 ----------
        btn_layout = QHBoxLayout()

        upload_btn = QPushButton("Select Image")
        detect_btn = QPushButton("Start Detection")

        upload_btn.clicked.connect(self.load_image)
        detect_btn.clicked.connect(self.run_detect)

        btn_layout.addWidget(upload_btn)
        btn_layout.addWidget(detect_btn)

        main_layout.addWidget(title)
        main_layout.addLayout(img_layout)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(btn_layout)

    # ---------------------------------------------------
    # ---------------------------------------------------
    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "*.jpg *.png *.tif *.jpeg"
        )

        if not file_path:
            return

        os.makedirs("images/tmp", exist_ok=True)

        suffix = os.path.splitext(file_path)[1]
        save_path = os.path.join("images/tmp", "current" + suffix)

        shutil.copy(file_path, save_path)

        img = cv2.imread(save_path)

        scale = self.output_size / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        show_path = "images/tmp/show_src.jpg"
        cv2.imwrite(show_path, img)

        self.current_image = file_path

        self.src_label.setPixmap(QPixmap(show_path))
        self.dst_label.clear()
        self.result_label.setText("Detection result: not detected")

    # ---------------------------------------------------
    # ---------------------------------------------------
    def run_detect(self):

        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return

        results = self.detector.predict_image(
            self.current_image,
            conf=self.conf,
            iou=self.iou
        )

        result = results[0]

        drawn = result.plot()

        save_img = copy.deepcopy(drawn)

        scale = self.output_size / drawn.shape[0]
        show_img = cv2.resize(drawn, (0, 0), fx=scale, fy=scale)

        os.makedirs("images/tmp", exist_ok=True)
        os.makedirs("record/img", exist_ok=True)
        os.makedirs("record/txt", exist_ok=True)

        base_name = os.path.basename(self.current_image)
        name_no_ext, _ = os.path.splitext(base_name)

        show_path = "images/tmp/show_result.jpg"
        cv2.imwrite(show_path, show_img)
        self.dst_label.setPixmap(QPixmap(show_path))

        save_img_path = os.path.join("record/img", base_name)
        cv2.imwrite(save_img_path, save_img)

        txt_path = os.path.join("record/txt", name_no_ext + ".txt")
        self._save_txt(result, txt_path)

        names = result.names

        if result.boxes is None:
            self.result_label.setText("Detection result: no target detected")
        else:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            counter = {}
            for cid in cls_ids:
                counter[cid] = counter.get(cid, 0) + 1

            info = []
            for cid, num in counter.items():
                info.append(f"{names[cid]} : {num}")

            if len(info) == 0:
                self.result_label.setText("Detection result: no target detected")
            else:
                self.result_label.setText("Detection result：\n" + "\n".join(info))

        QMessageBox.information(
            self,
            "Complete",
            f"Detection finished!\nImage saved to: record/img/{base_name}\nResult saved to: record/txt/{name_no_ext}.txt"
        )

    # ---------------------------------------------------
    # ---------------------------------------------------
    def _save_txt(self, result, save_path):


        if result.boxes is None:
            open(save_path, "w", encoding="utf-8").close()
            return

        boxes = result.boxes

        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        xywhn = boxes.xywhn.cpu().numpy()

        with open(save_path, "w", encoding="utf-8") as f:
            for i in range(len(cls)):
                line = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    int(cls[i]),
                    xywhn[i][0],
                    xywhn[i][1],
                    xywhn[i][2],
                    xywhn[i][3],
                    conf[i]
                )
                f.write(line)
