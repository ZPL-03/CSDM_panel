"""交互式 PyVista 几何与有限元结果视口。"""

from __future__ import annotations

from typing import Dict

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from gui.render_utils import build_candidate_scene, build_mode_shape_scene
from gui.theme import BACKGROUND, TEXT

try:
    from pyvistaqt import QtInteractor
except Exception:  # pragma: no cover
    QtInteractor = None


class InteractivePlotWidget(QWidget):
    def __init__(self, empty_message: str) -> None:
        super().__init__()
        self.empty_message = empty_message
        self._interactive = QtInteractor is not None
        self.plotter = None
        self._initial_camera_position = None

        self.message_label = QLabel(empty_message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)
        self.message_label.setMinimumHeight(180)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(6)

        self.stack = QStackedLayout()
        self.stack.addWidget(self.message_label)
        self.stack.addWidget(self.plot_container)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addLayout(self.stack)

        self.clear_scene(empty_message)

    def _ensure_plotter(self) -> bool:
        if self.plotter is not None:
            return True
        if not self._interactive:
            self.clear_scene("\u5f53\u524d\u73af\u5883\u672a\u5b89\u88c5 pyvistaqt\uff0c\u65e0\u6cd5\u63d0\u4f9b\u4ea4\u4e92\u5f0f\u4e09\u7ef4\u89c6\u56fe\u3002")
            return False
        try:
            self.plotter = QtInteractor(self.plot_container)
        except Exception:
            self.plotter = None
            self.clear_scene("\u5f53\u524d\u73af\u5883\u65e0\u6cd5\u521d\u59cb\u5316\u4ea4\u4e92\u5f0f OpenGL \u89c6\u56fe\uff0c\u53ef\u5728\u672c\u5730\u56fe\u5f62\u754c\u9762\u4e2d\u91cd\u8bd5\u3002")
            return False

        self.plot_layout.addWidget(self.plotter.interactor)
        self.plotter.interactor.installEventFilter(self)
        self.plotter.set_background(BACKGROUND)
        self.plotter.enable_trackball_style()
        return True

    def _dispose_plotter(self) -> None:
        if self.plotter is None:
            return
        interactor = getattr(self.plotter, "interactor", None)
        try:
            if interactor is not None:
                self.plot_layout.removeWidget(interactor)
                interactor.hide()
                interactor.close()
                interactor.setParent(None)
                interactor.deleteLater()
        except Exception:
            pass
        try:
            ren_win = getattr(self.plotter, "ren_win", None)
            if ren_win is not None:
                ren_win.Finalize()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        self.plotter = None
        self._initial_camera_position = None

    def reset_plotter(self, message: str | None = None) -> None:
        self._dispose_plotter()
        self.clear_scene(message)

    def clear_scene(self, message: str | None = None) -> None:
        self.message_label.setText(message or self.empty_message)
        if self.plotter is not None:
            try:
                self.plotter.clear()
                self.plotter.set_background(BACKGROUND)
            except Exception:
                self._dispose_plotter()
        self._initial_camera_position = None
        self.stack.setCurrentWidget(self.message_label)

    def _store_initial_camera(self) -> None:
        if self.plotter is not None:
            self._initial_camera_position = self.plotter.camera_position

    def _apply_default_camera(self, zoom: float) -> None:
        assert self.plotter is not None
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(zoom)
        self._store_initial_camera()
        self.plotter.render()

    def _restore_initial_camera(self) -> None:
        if self.plotter is None or self._initial_camera_position is None:
            return
        self.plotter.camera_position = self._initial_camera_position
        self.plotter.reset_camera()
        self.plotter.render()

    def eventFilter(self, watched, event):
        if self.plotter is not None and watched is getattr(self.plotter, "interactor", None):
            if event.type() == QEvent.Type.MouseButtonDblClick:
                self._restore_initial_camera()
                return True
        return super().eventFilter(watched, event)

    def _activate_plotter(self, title: str) -> bool:
        if not self._ensure_plotter():
            return False
        assert self.plotter is not None
        self.plotter.clear()
        self.plotter.set_background(BACKGROUND)
        self.plotter.add_text(title, position="upper_left", font_size=11, color=TEXT)
        self.plotter.add_axes(line_width=2, color=TEXT)
        self.stack.setCurrentWidget(self.plot_container)
        return True

    def closeEvent(self, event) -> None:
        self.reset_plotter()
        super().closeEvent(event)

    def show_candidate(self, candidate: Dict) -> None:
        scene = build_candidate_scene(candidate)
        if scene is None:
            self.clear_scene("\u5f53\u524d\u5019\u9009\u65b9\u6848\u7f3a\u5c11\u51e0\u4f55\u53c2\u6570\uff0c\u65e0\u6cd5\u663e\u793a\u4e09\u7ef4\u6a21\u578b\u3002")
            return
        meshes, title = scene
        if not self._activate_plotter(title):
            return
        assert self.plotter is not None
        for mesh, kwargs in meshes:
            self.plotter.add_mesh(mesh, **kwargs)
        self._apply_default_camera(0.94)

    def show_mode_shape(self, result: Dict) -> None:
        scene = build_mode_shape_scene(result)
        if scene is None:
            self.clear_scene("\u5f53\u524d\u7ed3\u679c\u8fd8\u6ca1\u6709\u53ef\u663e\u793a\u7684\u6a21\u6001\u4e91\u56fe\u6570\u636e\u3002")
            return
        mesh, scalar_name, title = scene
        if not self._activate_plotter(title):
            return
        assert self.plotter is not None
        self.plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="viridis",
            show_edges=False,
            smooth_shading=True,
            scalar_bar_args={
                "title": "Mode amplitude",
                "color": TEXT,
                "vertical": False,
                "position_x": 0.24,
                "position_y": 0.04,
                "height": 0.12,
                "width": 0.64,
                "n_labels": 3,
                "title_font_size": 10,
                "label_font_size": 9,
            },
        )
        self._apply_default_camera(0.86)
