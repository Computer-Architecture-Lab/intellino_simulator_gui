import sys
import os
import shutil
import numpy as np
import cv2

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QTextBrowser, QLineEdit, QFileDialog, QMessageBox, QSizePolicy
)
from PySide2.QtCore import Qt, QSize
from PySide2.QtGui import QPixmap, QIcon, QColor, QFont

from intellino.core.neuron_cell import NeuronCells
from custom_3 import train as intellino_train
from custom_3 import infer as intellino_infer, IMG_EXTS

# ------------------------------------------------------------
# 공통 유틸 (중복 정의 X)
# ------------------------------------------------------------
from utils.resource_utils import resource_path
from utils.ui_common import TitleBar, BUTTON_STYLE
from utils.image_preprocess import preprocess_digit_image

#=======================================================================================================#
#                                                 main                                                  #
#=======================================================================================================#
class ExperimentWindow(QWidget):
    """Experiment 결과 요약 + best_results 구성 + Inference + Finish"""

    def __init__(self, num_categories=0, best_results_root=None):
        super().__init__()

        self.num_categories = num_categories

        # custom_3에서 넘어온 best_results_root
        if best_results_root is not None:
            self.best_results_root = best_results_root
        else:
            self.best_results_root = os.path.join(os.getcwd(), "best_results")

        self.best_results_path = None    # 7번 Output 폴더 경로
        self.test_root = None            # datasets/test 절대경로
        self.selected_param_dir = None   # 선택한 Vxx_Cxx_Txx_MxxK 폴더
        self.neuron_cells = None         # 학습한 모델 저장
        self.param_str = None            # V@/T@/C@/M@ 문자열

        # 기본 UI 초기화
        self._setup_window()
        self._setup_titlebar()
        self._setup_layout_placeholder()

        # custom_4 UI 전체 생성
        self.build_ui()

    # 창 기본 세팅
    def _setup_window(self):
        """창 스타일/크기/배경/그림자 설정"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(800, 800)

        # 메인 배경 컨테이너
        self.container = QWidget(self)
        self.container.setGeometry(0, 0, 800, 800)
        self.container.setObjectName("base_container")
        self.container.setStyleSheet("""
            #base_container {
                background: #ffffff;
                border-radius: 15px;
            }
        """)

    # TitleBar 추가
    def _setup_titlebar(self):
        """상단 TitleBar(로고 + home 버튼)"""
        self.titlebar = TitleBar(self)
        self.titlebar.setParent(self.container)
        self.titlebar.setGeometry(0, 0, 800, 50)

    # 전체 레이아웃 placeholder
    def _setup_layout_placeholder(self):
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 80, 20, 20)
        layout.setSpacing(18)
        self.main_layout = layout


# ============================================================
# custom_4.py — PART 2
# best_results 생성 + Output Folder Section UI
# ============================================================

    def _build_best_results(self):
        """best_results 경로 + test 경로 생성"""
        self.best_results_path = self.best_results_root
        run_root = os.path.dirname(self.best_results_root)
        self.test_root = os.path.join(run_root, "datasets", "test")

    # 절대 경로 표시 편하게
    def _pretty_path(self, path: str) -> str:
        if not path:
            return ""

        lower = path.lower()
        key = os.path.join("intellino_simulator_gui").lower()
        idx = lower.find(key)
        if idx != -1:
            return path[idx:]
        return path

    # Start 버튼 enable/disable 공통 처리
    def _set_start_button_enabled(self, enabled: bool):
        if not hasattr(self, "start_btn"):
            return
        self.start_btn.setEnabled(enabled)
        if enabled:
            self.start_btn.setStyleSheet(BUTTON_STYLE)
        else:
            # 회색으로 비활성화 느낌만 주기
            self.start_btn.setStyleSheet(BUTTON_STYLE + "color: gray;")

    # Output Folder Section 생성 (UI 교체됨)
    def _add_output_section(self):
        """7. Output Folder — 파라미터 폴더 선택 + Apply(학습)"""

        group = QGroupBox("7. Output Folder")
        group.setStyleSheet("""
            QGroupBox {
                font-weight:bold; 
                border:1px solid #ccc; 
                border-radius:10px; 
                margin-top:6px; 
                padding:8px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding:-6px 4px 4px 4px;
            }
        """)

        hl = QHBoxLayout()

        # Label: param folder (Vxx_Cxx_Txx_MxxK) — 테두리 없는 텍스트만
        self.output_label = QLabel("")
        self.output_label.setStyleSheet("font-size:13px;")
        self.output_label.setMinimumHeight(30)

        # "..." 버튼 → 폴더 선택
        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(36, 30)
        browse_btn.setStyleSheet(BUTTON_STYLE)
        browse_btn.clicked.connect(self._browse_param_folder)

        # Apply 버튼 → train 실행
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedSize(80, 30)
        apply_btn.setStyleSheet(BUTTON_STYLE)
        apply_btn.clicked.connect(self._run_train_only)

        hl.addWidget(self.output_label)
        hl.addWidget(browse_btn)
        hl.addWidget(apply_btn)

        group.setLayout(hl)
        self.main_layout.addWidget(group)

    # 파라미터 폴더 선택
    def _browse_param_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select parameter folder",
            self.best_results_path
        )
        if path:
            self.selected_param_dir = path
            self.output_label.setText(self._pretty_path(path))

            # 새 파라미터를 선택했으니, 다시 학습 전까지 Start 비활성화
            self.neuron_cells = None
            self.param_str = None
            self._set_start_button_enabled(False)

    # 학습만 실행
    def _run_train_only(self):
        """7번 Output Folder → Apply 버튼 동작"""

        param_dir = self.selected_param_dir
        if not param_dir:
            QMessageBox.warning(self, "No folder", "먼저 '...' 버튼으로 폴더를 선택하세요.")
            return
        if not os.path.isdir(param_dir):
            QMessageBox.warning(self, "Invalid folder", "선택한 폴더가 존재하지 않습니다.")
            return

        # 폴더명 파싱 (예: V256_C3_T2_M8K)
        folder = os.path.basename(param_dir)
        parts = folder.split("_")
        try:
            v = int(parts[0][1:])
            c = int(parts[1][1:])
            t = int(parts[2][1:])
        except Exception:
            QMessageBox.warning(self, "Format error",
                                "폴더 이름이 V@_C@_T@ 형식이 아닙니다.")
            return

        m_str = parts[3] if len(parts) >= 4 else None
        # parameter : V@/T@/C@/M@ 형식 문자열 저장
        self.param_str = f"V{v}/S{c}/C{t}"
        if m_str:
            self.param_str += f"/{m_str}"

        # train_samples 만들기
        train_samples = []
        for label_dir in sorted(os.listdir(param_dir)):
            label_path = os.path.join(param_dir, label_dir)
            if not os.path.isdir(label_path):
                continue

            for fname in sorted(os.listdir(label_path)):
                if fname.lower().endswith(IMG_EXTS):
                    train_samples.append(
                        (os.path.join(label_path, fname), label_dir)
                    )

        if not train_samples:
            QMessageBox.warning(self, "No data", "선택 폴더에 이미지가 없습니다.")
            return

        # 모델 생성 + 학습
        length = v
        num_cells = c * t

        self.neuron_cells = NeuronCells(
            number_of_neuron_cells=num_cells,
            length_of_input_vector=length,
            measure="manhattan"
        )

        intellino_train(
            neuron_cells=self.neuron_cells,
            train_samples=train_samples,
            number_of_neuron_cells=num_cells,
            length_of_input_vector=length,
            progress_callback=None
        )

        # 학습 완료 후 Start 버튼 활성화
        self._set_start_button_enabled(True)

        print("TRAINED LENGTH =", self.neuron_cells.length_of_input_vector)
        print("TRAINED CELL COUNT =", self.neuron_cells.number_of_neuron_cells)

        print("Param DIR =", param_dir)
        print("Train samples count =", len(train_samples))
        for i in range(3):
            print(train_samples[i])

# ============================================================
# custom_4.py — PART 3
# Test Section (UI 교체) + Test 실행 함수
# ============================================================

    def _add_inference_section(self):
        """8. Test — 경로 표시 + Start 버튼"""

        group = QGroupBox("8. Test")
        group.setStyleSheet("""
            QGroupBox {
                font-weight:bold; 
                border:1px solid #ccc; 
                border-radius:10px; 
                margin-top:6px; 
                padding:8px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding:-6px 4px 4px 4px;
            }
        """)

        hl = QHBoxLayout()

        # Label: test_root 표시
        self.test_label = QLabel(self._pretty_path(self.test_root))
        self.test_label.setStyleSheet("font-size:13px;")
        self.test_label.setMinimumHeight(30)

        # Start 버튼 (test only)
        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedSize(80, 30)
        self._set_start_button_enabled(False)   # 초기에는 비활성화
        self.start_btn.clicked.connect(self._run_test_only)

        hl.addWidget(self.test_label)
        hl.addStretch()
        hl.addWidget(self.start_btn)

        group.setLayout(hl)
        self.main_layout.addWidget(group)

    # Test만 수행
    def _run_test_only(self):
        """8번 Test → Start 버튼"""

        if self.neuron_cells is None:
            QMessageBox.warning(self, "No Model",
                                "먼저 7번 Output Folder에서 Apply로 학습을 완료하세요.")
            return

        if not self.test_root or not os.path.isdir(self.test_root):
            QMessageBox.warning(self, "No test set",
                                "datasets/test 폴더가 없습니다.")
            return

        per_correct = {}
        per_total = {}

        # 전체 class 평가
        for label_dir in sorted(os.listdir(self.test_root)):
            label_path = os.path.join(self.test_root, label_dir)
            if not os.path.isdir(label_path):
                continue

            per_correct.setdefault(label_dir, 0)
            per_total.setdefault(label_dir, 0)

            for fname in sorted(os.listdir(label_path)):
                if not fname.lower().endswith(IMG_EXTS):
                    continue

                img_path = os.path.join(label_path, fname)

                pred = intellino_infer(
                    neuron_cells=self.neuron_cells,
                    image_path=img_path,
                    length_of_input_vector=self.neuron_cells.length_of_input_vector
                )

                per_total[label_dir] += 1
                if str(pred) == str(label_dir):
                    per_correct[label_dir] += 1

        # ====== 여기서부터 Result 포맷 출력 ======
        class_list = sorted(per_total.keys(), key=lambda x: int(x))
        classes_str = "/".join(class_list)

        # 한 번 실행할 때마다 아래로 누적
        self._append_result("")  # 빈 줄로 구분
        self._append_result(f"class         : {classes_str}")

        # param_str 이 있으면 그대로, 없으면 '-' 표시
        param_to_show = self.param_str if self.param_str else "-"
        self._append_result(f"parameter : {param_to_show}")
        self._append_result(f"  ")
        self._append_result(f"correct/total :")

        for cls in class_list:
            c = per_correct.get(cls, 0)
            t = per_total.get(cls, 0)
            self._append_result(f"{cls} : ({c}/{t})")
        self._append_result("-" *32)

        vec = preprocess_digit_image(...)
        print("TEST VECTOR LENGTH =", len(vec))

# ============================================================
# custom_4.py — PART 4
# Result Section (QTextBrowser) + 출력 헬퍼 함수들
# ============================================================

    def _add_result_section(self):
        """9. Result"""

        group = QGroupBox("9. Result")
        group.setStyleSheet("""
            QGroupBox {
                font-weight:bold; 
                border:1px solid #ccc; 
                border-radius:10px; 
                margin-top:6px; 
                padding:8px 10px 10px 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding:-6px 4px 4px 4px;
            }
        """)

        layout = QVBoxLayout()

        self.result_box = QTextBrowser()
        self.result_box.setStyleSheet("""
            QTextBrowser {
                background:#f8f9fa;
                border:1px solid #ccc;
                border-radius:8px;
                font-size:14px;
                padding:10px;
            }
        """)

        layout.addWidget(self.result_box)
        group.setLayout(layout)
        self.main_layout.addWidget(group)

    def _append_result(self, html: str):
        self.result_box.append(html)
        self.result_box.verticalScrollBar().setValue(
            self.result_box.verticalScrollBar().maximum()
        )

    def _append_hr(self):
        # 현재는 사용하지 않지만, 혹시 나중을 위해 남겨둠
        self.result_box.append("<hr style='margin:8px 0;'>")
        self.result_box.verticalScrollBar().setValue(
            self.result_box.verticalScrollBar().maximum()
        )

# ============================================================
# custom_4.py — PART 5
# Finish 버튼 + 전체 레이아웃 연결
# ============================================================
    def _add_finish_button(self):
        """Finish 버튼"""
        btn = QPushButton("Finish")
        btn.setFixedSize(130, 40)
        btn.setStyleSheet(BUTTON_STYLE)
        btn.clicked.connect(self.close)

        hl = QHBoxLayout()
        hl.addStretch()
        hl.addWidget(btn)
        self.main_layout.addLayout(hl)

    def build_ui(self):
        """전체 UI 배치"""

        self._build_best_results()
        self._add_output_section()
        self._add_inference_section()
        self._add_result_section()
        self._add_finish_button()
