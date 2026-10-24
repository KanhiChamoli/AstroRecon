"""Small Qt window for browsing rendered PNG artifacts."""

from pathlib import Path
import sys

from PyQt5 import QtCore, QtGui, QtWidgets


class RenderGallery(QtWidgets.QMainWindow):
	def __init__(self, image_paths):
		super().__init__()
		self.image_paths = [Path(path).resolve() for path in image_paths if Path(path).is_file()]
		if not self.image_paths:
			raise FileNotFoundError("No render images were found to display")
		self.index = 0
		self.setWindowTitle("AstroRecon Render Viewer")
		self.resize(1000, 780)
		self.setMinimumSize(640, 480)

		central = QtWidgets.QWidget(self)
		layout = QtWidgets.QVBoxLayout(central)
		layout.setContentsMargins(12, 12, 12, 12)
		self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
		self.image_label.setMinimumSize(320, 320)
		self.image_label.setStyleSheet("background-color: #171a1f; color: #e6e8eb;")
		self.caption = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
		self.caption.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

		controls = QtWidgets.QHBoxLayout()
		self.previous_button = QtWidgets.QPushButton("Previous")
		self.counter = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
		self.next_button = QtWidgets.QPushButton("Next")
		self.previous_button.clicked.connect(lambda: self.show_index(self.index - 1))
		self.next_button.clicked.connect(lambda: self.show_index(self.index + 1))
		controls.addWidget(self.previous_button)
		controls.addStretch(1)
		controls.addWidget(self.counter)
		controls.addStretch(1)
		controls.addWidget(self.next_button)
		layout.addWidget(self.image_label, stretch=1)
		layout.addWidget(self.caption)
		layout.addLayout(controls)
		self.setCentralWidget(central)
		self.show_index(0)

	def show_index(self, index):
		self.index = index % len(self.image_paths)
		path = self.image_paths[self.index]
		pixmap = QtGui.QPixmap(str(path))
		if pixmap.isNull():
			self.image_label.setText(f"Could not load image:\n{path}")
		else:
			self.image_label.setPixmap(
				pixmap.scaled(
					self.image_label.size(),
					QtCore.Qt.KeepAspectRatio,
					QtCore.Qt.SmoothTransformation,
				)
			)
		self.setWindowTitle(f"AstroRecon Render Viewer | {path.name}")
		self.caption.setText(str(path))
		self.counter.setText(f"{self.index + 1} / {len(self.image_paths)}")

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self.show_index(self.index)

	def keyPressEvent(self, event):
		if event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Space):
			self.show_index(self.index + 1)
		elif event.key() == QtCore.Qt.Key_Left:
			self.show_index(self.index - 1)
		else:
			super().keyPressEvent(event)


def show_render_gallery(image_paths):
	"""Open rendered artifacts in a separate, navigable application window."""
	app = QtWidgets.QApplication.instance()
	created_app = app is None
	if created_app:
		app = QtWidgets.QApplication(sys.argv[:1])
	window = RenderGallery(image_paths)
	window.show()
	window.raise_()
	window.activateWindow()
	if created_app:
		app.exec_()
	return window