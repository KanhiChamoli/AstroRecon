"""Windowed Simulation Engine for real-time interactive rendering and convergence monitoring."""

from pathlib import Path
import sys
import logging
import time
import urllib.parse
from typing import Optional, Callable, Tuple, Dict, Any
import webbrowser

import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers --

def _array_to_png_bytes(array: np.ndarray, colormap: str = "magma") -> bytes:
    """Encode a 2D float array in [0,1] to PNG bytes using the same color ramps
    as the Qt viewport, so exported files match what the user sees."""
    import io
    from PIL import Image

    array = np.clip(np.nan_to_num(array), 0.0, 1.0)
    if colormap == "viridis":
        r = np.uint8(np.clip(255 * (array ** 0.8), 0, 255))
        g = np.uint8(np.clip(255 * (array ** 0.5), 0, 255))
        b = np.uint8(np.clip(255 * (1.0 - array), 0, 255))
    else:
        r = np.uint8(np.clip(255 * (array ** 0.7), 0, 255))
        g = np.uint8(np.clip(255 * (array ** 1.8), 0, 255))
        b = np.uint8(np.clip(255 * (array ** 3.0), 0, 255))
    rgb = np.stack([r, g, b], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _export_volume_obj(path, shape, scale_note: str = "") -> None:
    """Write a simple OBJ cube bounding-box representing the volumetric data.

    A dense 64^3 voxel mesh would be enormous, so we export a light placeholder
    cube that external tools can use as a reference bound; the real density
    volume ships alongside as density.vtk / .fits.
    """
    sx, sy, sz = shape
    lines = ["# AstroRecon volumetric export placeholder", "o astrorecon_volume"]
    # 8 corners of a unit cube scaled by data extents
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    for i, (a, b, c) in enumerate(corners, start=1):
        lines.append(f"v {a * sx:.4f} {b * sy:.4f} {c * sz:.4f}")
    faces = [(1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5),
             (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6)]
    for f in faces:
        lines.append("f " + " ".join(str(i) for i in f))
    if scale_note:
        lines.append(f"# {scale_note}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_engine_scripts(target_name: str, recon_path, volume_path, meta) -> dict:
    """Return a dict of copy-pasteable scripts for external 3D packages."""
    recon_rel = str(recon_path).replace("\\", "/") if recon_path else "outputs/renders/reconstruction.png"
    vol_rel = str(volume_path).replace("\\", "/") if volume_path else "outputs/volumes/density.vtk"

    blender = f'''# Blender 3.x/4.x/5.x — AstroRecon volumetric import
# Paste into Blender's Scripting tab (or run: blender --python this_file.py)
#
# NOTE: this script only runs INSIDE Blender. Running it with a plain Python
# interpreter raises "ModuleNotFoundError: No module named 'bpy'" because
# `bpy` ships with Blender itself, not with pip.
import bpy

# --- version-proof socket lookup -----------------------------------------
# Blender 4.x renamed the Principled BSDF "Emission" input to
# "Emission Color" (and split out "Emission Strength"). Blender 5.x keeps the
# new names. Resolve the socket by name with a fallback so the same script
# works across versions instead of raising KeyError on the wrong build.
def _principled_emission_socket(node):
    for name in ("Emission Color", "Emission"):
        if name in node.inputs:
            return node.inputs[name]
    raise KeyError(
        "Principled BSDF has no emission socket; inputs are: "
        + ", ".join(s.name for s in node.inputs)
    )

# Clean default scene objects
for ob in list(bpy.data.objects):
    if ob.name.startswith(("Cube", "Light", "Camera")):
        bpy.data.objects.remove(ob, do_unlink=True)

# Load the reconstructed density volume as a volumetric shader cube
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
vol_obj = bpy.context.active_object
vol_obj.name = "AstroRecon_Density"
mat = bpy.data.materials.new("AstroRecon_Volume")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
principled_emission = _principled_emission_socket(principled)
principled_emission.default_value = (1.0, 0.55, 0.15, 1.0)
# Blender 4.x+ also needs emission strength > 0 for the color to show up.
if "Emission Strength" in principled.inputs:
    principled.inputs["Emission Strength"].default_value = 1.0
nt.links.new(principled.outputs[0], out.inputs[0])
vol_obj.data.materials.append(mat)

# Import the reconstructed 2D image as an emission texture plane (quick preview)
img = bpy.data.images.load("{recon_rel}")
plane_data = bpy.data.meshes.new("ReconPlane")
import bmesh, mathutils
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
bm.to_mesh(plane_data)
bm.free()
plane = bpy.data.objects.new("AstroRecon_ReconPlane", plane_data)
bpy.context.collection.objects.link(plane)
tex_mat = bpy.data.materials.new("AstroRecon_Tex")
tex_mat.use_nodes = True
nt2 = tex_mat.node_tree
img_node = nt2.nodes.new("ShaderNodeTexImage")
img_node.image = img
emis = nt2.nodes.new("ShaderNodeEmission")
nt2.links.new(img_node.outputs[0], emis.inputs[0])
out2 = nt2.nodes.new("ShaderNodeOutputMaterial")
nt2.links.new(emis.outputs[0], out2.inputs[0])
plane.data.materials.append(tex_mat)

print("AstroRecon volume placeholder + reconstruction plane loaded.")
print("Density volume file (import via Volume Import add-on): {vol_rel}")
'''

    maya = f'''# Maya (Python / MEL-compatible) — AstroRecon reconstruction import
# Paste into Maya's Script Editor (Python tab)
import maya.cmds as cmds

# Reference plane textured with the reconstructed image
plane = cmds.polyPlane(w=10, h=10, sx=1, sy=1, name="AstroRecon_ReconPlane")
shader = cmds.shadingNode("lambert", asShader=True, name="AstroRecon_Tex")
cmds.setAttr(shader + ".color", 0.8, 0.45, 0.12, type="double3")
cmds.select(plane)
cmds.hyperShade(assign=shader)

# Volumetric bounding box placeholder for the density volume
bbox = cmds.polyCube(w=10, h=10, d=10, name="AstroRecon_VolumeBox")
cmds.setAttr(bbox[0] + ".displayHandle", 1)
print("AstroRecon data loaded. Density volume for volumetric import: {vol_rel}")
'''

    houdini = f'''# Houdini (Python source editor) — AstroRecon import helper
# Paste into Houdini's Python source editor pane.
import hou

obj = hou.node("/obj")
box = obj.createNode("geo", "AstroRecon_Volume")
print("Created /obj/AstroRecon_Volume — import density volume:")
print("  File > Import > Volume, then point it at:", "{vol_rel}")
print("Reference reconstruction image:", "{recon_rel}")
'''

    paraview = f'''# ParaView (Python trace) — AstroRecon volumetric data import
# Run from ParaView's Python shell (Tools > Python Shell)
from paraview.simple import *
from paraview import servermanager

# Load the AstroRecon density volume (legacy VTK)
volume = LegacyVTKReader(FileNames=[r"{vol_rel}"])
RenameSource("AstroRecon_Density", volume)
Show(volume)
Render()
print("AstroRecon volume loaded — apply Volume Rendering in the display panel.")
'''

    return {
        "blender": blender,
        "maya": maya,
        "houdini": houdini,
        "paraview": paraview,
        "meta": meta or {},
    }


class SimulationEngineWindow(QtWidgets.QMainWindow):
    """Interactive desktop application window displaying live reconstruction, telemetry, and astrophysical metadata."""

    def __init__(
        self,
        title: str = "AstroRecon Simulation Engine | Live Viewport",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1300, 840)
        self.setMinimumSize(960, 600)
        self.is_paused = False
        self.metadata = metadata or {}
        self.source_url = self.metadata.get("source_url", "https://skyview.gsfc.nasa.gov/")
        self.start_time = time.time()
        self._output_dir = Path("outputs")
        self._latest_recon_path = None
        self._latest_volume_path = None
        self._recon_2d = None  # cached 2D recon for exports

        self._build_menu_bar()

        # Central Widget & Main Vertical Layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        # ---------------------------------------------------------
        # 1. Header & Live Telemetry HUD Bar
        # ---------------------------------------------------------
        hud_frame = QtWidgets.QFrame()
        hud_frame.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        hud_layout = QtWidgets.QHBoxLayout(hud_frame)
        hud_layout.setContentsMargins(14, 6, 14, 6)

        self.label_status = QtWidgets.QLabel("Status: OPTIMIZING")
        self.label_status.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 13px;")

        self.label_epoch = QtWidgets.QLabel("Epoch: 0 / 0")
        self.label_epoch.setStyleSheet("color: #f0f6fc; font-size: 13px;")

        self.label_loss = QtWidgets.QLabel("Total Loss: --")
        self.label_loss.setStyleSheet("color: #7ee787; font-weight: bold; font-size: 13px;")

        self.label_fps = QtWidgets.QLabel("Speed: -- it/s")
        self.label_fps.setStyleSheet("color: #d2a8ff; font-size: 13px;")

        self.label_time = QtWidgets.QLabel("Elapsed: 0.0s")
        self.label_time.setStyleSheet("color: #ffa657; font-size: 13px;")

        hud_layout.addWidget(self.label_status)
        hud_layout.addStretch(1)
        hud_layout.addWidget(self.label_epoch)
        hud_layout.addStretch(1)
        hud_layout.addWidget(self.label_loss)
        hud_layout.addStretch(1)
        hud_layout.addWidget(self.label_fps)
        hud_layout.addStretch(1)
        hud_layout.addWidget(self.label_time)

        # Y2K-style "Fetch Information" button beside the epoch/feed/loss stats
        self.btn_fetch_info = QtWidgets.QPushButton("📡 Fetch Information")
        self.btn_fetch_info.setToolTip(
            "Look up the latest scientific background about the currently loaded object"
        )
        self.btn_fetch_info.setStyleSheet("""
            QPushButton {
                background-color: #1f6feb; color: #ffffff;
                padding: 4px 12px; border-radius: 5px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #388bfd; }
        """)
        self.btn_fetch_info.clicked.connect(self._fetch_information)
        hud_layout.addSpacing(10)
        hud_layout.addWidget(self.btn_fetch_info)
        main_layout.addWidget(hud_frame)

        # ---------------------------------------------------------
        # 2. Main Content Splitter: [Viewports (Left)] + [Metadata Inspector (Right)]
        # ---------------------------------------------------------
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(12)

        # Left: Three Synchronized Viewports
        viewports_layout = QtWidgets.QHBoxLayout()
        viewports_layout.setSpacing(8)

        self.target_layout, self.target_caption = self._create_viewport_panel("Target Observation")
        self.recon_layout, self.recon_caption = self._create_viewport_panel("Model Reconstruction (Live)")
        self.residual_layout, self.residual_caption = self._create_viewport_panel("Absolute Residual Error")

        viewports_layout.addLayout(self.target_layout)
        viewports_layout.addLayout(self.recon_layout)
        viewports_layout.addLayout(self.residual_layout)
        content_layout.addLayout(viewports_layout, stretch=3)

        # Right: Astrophysical Metadata & Processing Inspector
        inspector_frame = self._build_metadata_inspector()
        content_layout.addWidget(inspector_frame, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

        # ---------------------------------------------------------
        # 3. Footer Bar: Branding (Bottom-Left) + Source & Controls (Bottom-Right)
        # ---------------------------------------------------------
        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setContentsMargins(4, 4, 4, 4)

        # Bottom Left: User Ownership & Copyright
        self.label_branding = QtWidgets.QLabel(
            "Made and owned by Aaradhya Chamoli  |  © AstroRecon Project (All Rights Reserved)"
        )
        self.label_branding.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 500;")
        footer_layout.addWidget(self.label_branding)

        footer_layout.addStretch(1)

        # Clickable Source Button
        self.btn_source = QtWidgets.QPushButton("🌐 View Data Source")
        self.btn_source.setStyleSheet("""
            QPushButton {
                background-color: #1f6feb;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #388bfd; }
        """)
        self.btn_source.clicked.connect(self._open_source_url)
        footer_layout.addWidget(self.btn_source)

        # Pause / Resume Button
        self.btn_pause = QtWidgets.QPushButton("Pause Simulation")
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #f0f6fc;
                padding: 6px 14px;
                border-radius: 5px;
                border: 1px solid #30363d;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #30363d; }
        """)
        self.btn_pause.clicked.connect(self._toggle_pause)
        footer_layout.addWidget(self.btn_pause)

        # Save Snapshot Button
        self.btn_save = QtWidgets.QPushButton("📸 Save Snapshot")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #238636;
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2ea043; }
        """)
        self.btn_save.clicked.connect(self._save_snapshot)
        footer_layout.addWidget(self.btn_save)

        main_layout.addLayout(footer_layout)

        # State storage
        self.latest_recon = None
        self.latest_target = None

        # Dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; color: #c9d1d9; }
            QLabel { color: #c9d1d9; }
        """)

    def _build_menu_bar(self):
        """Build the top menu bar: File (Save/Export), View, Help."""
        menu_bar = self.menuBar()

        # ---- File menu ----
        file_menu = menu_bar.addMenu("&File")
        act_snapshot = QtWidgets.QAction("📸 Save Snapshot…", self)
        act_snapshot.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        act_snapshot.triggered.connect(self._save_snapshot)
        file_menu.addAction(act_snapshot)

        export_menu = file_menu.addMenu("Export for 3D Packages…")
        export_menu.addAction("Blender script", lambda: self._export_script("blender"))
        export_menu.addAction("Maya script", lambda: self._export_script("maya"))
        export_menu.addAction("Houdini script", lambda: self._export_script("houdini"))
        export_menu.addAction("ParaView script", lambda: self._export_script("paraview"))
        export_menu.addSeparator()
        export_menu.addAction("Export volume as OBJ…", self._export_obj)

        file_menu.addSeparator()
        act_quit = QtWidgets.QAction("Exit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ---- View menu ----
        view_menu = menu_bar.addMenu("&View")
        self._act_pause = QtWidgets.QAction("⏸ Pause Simulation", self)
        self._act_pause.setShortcut(QtGui.QKeySequence("Space"))
        self._act_pause.setCheckable(True)
        self._act_pause.toggled.connect(self._toggle_pause)
        view_menu.addAction(self._act_pause)
        view_menu.addSeparator()
        act_source = QtWidgets.QAction("🌐 Open Data Source", self)
        act_source.triggered.connect(self._open_source_url)
        view_menu.addAction(act_source)

        # ---- Help menu ----
        help_menu = menu_bar.addMenu("&Help")
        act_about = QtWidgets.QAction("About AstroRecon", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About AstroRecon",
            "<b>AstroRecon Simulation Engine</b><br>"
            "Live reconstruction viewport with telemetry, residual analysis, "
            "and volumetric export for 3D packages.<br><br>"
            "Made and owned by Aaradhya Chamoli — © AstroRecon Project.",
        )

    def _fetch_information(self):
        """Open a lookup of scientific background for the current target."""
        target = self.metadata.get("target", "Sagittarius A*")
        query = urllib.parse.quote_plus(f"{target} radio astronomy observations")
        url = f"https://ui.adsabs.harvard.edu/search/q={query}"
        try:
            webbrowser.open(url)
            logger.info("Fetched information for '%s' -> %s", target, url)
        except Exception as e:
            logger.error("Failed to fetch information: %s", e)
            QtWidgets.QMessageBox.warning(self, "Fetch failed", f"Could not open the lookup page:\n{e}")

    def _export_script(self, package: str):
        """Write a copy-pasteable import script for an external 3D package."""
        out_dir = self._output_dir / "engine_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        target_name = self.metadata.get("target", "observation")
        scripts = _generate_engine_scripts(
            target_name, self._latest_recon_path, self._latest_volume_path, self.metadata
        )
        out_path = out_dir / f"{package}_import.py"
        out_path.write_text(scripts[package], encoding="utf-8")
        QtWidgets.QMessageBox.information(
            self,
            f"{package.capitalize()} export ready",
            f"Import script saved to:\n<b>{out_path}</b><br><br>"
            f"Open it in {package.capitalize()}'s scripting console or run it with "
            f"{package}'s Python interpreter.",
        )
        logger.info("Exported %s import script: %s", package, out_path)

    def _export_obj(self):
        """Export a lightweight OBJ placeholder for the volumetric data."""
        if self._recon_2d is None and self.latest_recon is None:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to export", "Run a reconstruction first, then export."
            )
            return
        out_dir = self._output_dir / "engine_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        shape = self._recon_2d.shape if self._recon_2d is not None else self.latest_recon.shape
        out_path = out_dir / "astrorecon_volume.obj"
        _export_volume_obj(out_path, (shape[1], shape[0], 64))
        QtWidgets.QMessageBox.information(
            self, "OBJ exported", f"Volumetric placeholder written to:<br><b>{out_path}</b>"
        )
        logger.info("Exported OBJ placeholder: %s", out_path)

    def _create_viewport_panel(self, title_text: str):
        layout = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(title_text)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #8b949e;")

        view = QtWidgets.QLabel()
        view.setAlignment(QtCore.Qt.AlignCenter)
        view.setMinimumSize(220, 220)
        view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        view.setStyleSheet("background-color: #030407; border: 1px solid #30363d; border-radius: 6px;")

        layout.addWidget(title)
        layout.addWidget(view, stretch=1)
        return layout, view

    def _build_metadata_inspector(self) -> QtWidgets.QFrame:
        """Construct the right-hand panel detailing dataset origin, processing, and voxel physics."""
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("ASTROPHYSICAL METADATA")
        title.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 12px; letter-spacing: 1px;")
        layout.addWidget(title)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color: #30363d;")
        layout.addWidget(line)

        # Source / Dataset info
        source_name = self.metadata.get("source_name", "NASA SkyView Radio Survey")
        target_name = self.metadata.get("target", "Sagittarius A*")
        dim_info = self.metadata.get("dimensions", "300 x 300 px")
        self.label_meta_source = QtWidgets.QLabel(f"<b>Dataset:</b> {source_name}")
        self.label_meta_target = QtWidgets.QLabel(f"<b>Target:</b> {target_name}")
        self.label_meta_dims = QtWidgets.QLabel(f"<b>Grid Size:</b> {dim_info}")
        layout.addWidget(self.label_meta_source)
        layout.addWidget(self.label_meta_target)
        layout.addWidget(self.label_meta_dims)

        # Processing info
        proc_text = self.metadata.get("processing", "asinh stretch, median bg subtraction, TV smoothing")
        self.label_meta_proc = QtWidgets.QLabel(f"<b>Processing:</b><br><span style='color: #8b949e;'>{proc_text}</span>")
        self.label_meta_proc.setWordWrap(True)
        layout.addWidget(self.label_meta_proc)

        # Physics & Density Telemetry
        phys_title = QtWidgets.QLabel("VOXEL DENSITY & PLASMA")
        phys_title.setStyleSheet("color: #7ee787; font-weight: bold; font-size: 11px; margin-top: 8px;")
        layout.addWidget(phys_title)

        self.label_density_mean = QtWidgets.QLabel("<b>Mean Density (ρ̄):</b> --")
        self.label_density_peak = QtWidgets.QLabel("<b>Peak Density (ρ_max):</b> --")
        self.label_temp_avg = QtWidgets.QLabel("<b>Avg Voxel Temp (T_b):</b> -- K")
        self.label_temp_peak = QtWidgets.QLabel("<b>Peak Plasma Temp:</b> -- K")

        layout.addWidget(self.label_density_mean)
        layout.addWidget(self.label_density_peak)
        layout.addWidget(self.label_temp_avg)
        layout.addWidget(self.label_temp_peak)

        layout.addStretch(1)
        return frame

    def set_metadata(self, metadata: Dict[str, Any]):
        """Dynamically update metadata in the inspector."""
        self.metadata = metadata
        self.source_url = metadata.get("source_url", self.source_url)
        self.label_meta_source.setText(f"<b>Dataset:</b> {metadata.get('source_name', 'Observation')}")
        self.label_meta_target.setText(f"<b>Target:</b> {metadata.get('target', 'Astronomical Source')}")
        self.label_meta_dims.setText(f"<b>Grid Size:</b> {metadata.get('dimensions', '--')}")
        if "processing" in metadata:
            self.label_meta_proc.setText(f"<b>Processing:</b><br><span style='color: #8b949e;'>{metadata['processing']}</span>")

    def _open_source_url(self):
        """Open the original dataset source URL in the web browser safely."""
        if self.source_url:
            try:
                logger.info("Opening data source: %s", self.source_url)
                webbrowser.open(self.source_url)
            except Exception as e:
                logger.error("Failed to open URL %s: %s", self.source_url, e)
                # Keep the application running


    def _tensor_to_qpixmap(self, array: np.ndarray, colormap: str = "magma") -> QtGui.QPixmap:
        """Convert a 2D float numpy array in [0, 1] into a formatted QPixmap."""
        array = np.clip(np.nan_to_num(array), 0.0, 1.0)
        h, w = array.shape

        if colormap == "viridis":
            r = np.uint8(np.clip(255 * (array ** 0.8), 0, 255))
            g = np.uint8(np.clip(255 * (array ** 0.5), 0, 255))
            b = np.uint8(np.clip(255 * (1.0 - array), 0, 255))
        else:
            r = np.uint8(np.clip(255 * (array ** 0.7), 0, 255))
            g = np.uint8(np.clip(255 * (array ** 1.8), 0, 255))
            b = np.uint8(np.clip(255 * (array ** 3.0), 0, 255))

        rgb = np.stack([r, g, b], axis=-1)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg)

    def set_target(self, target_image: np.ndarray):
        """Set the static ground truth observation."""
        self.latest_target = np.asarray(target_image, dtype=np.float32)
        pixmap = self._tensor_to_qpixmap(self.latest_target)
        self.target_caption.setPixmap(pixmap.scaled(
            self.target_caption.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
        self.target_caption.repaint()  # force immediate paint so target is visible

    def update_simulation_frame(
        self,
        epoch: int,
        total_epochs: int,
        recon_image: np.ndarray,
        loss_val: float,
        step_fps: float = 0.0,
    ):
        """Update live viewports, timing telemetry, and physical voxel statistics."""
        # NOTE: pause is handled in train.py (blocked with a cooperative wait).
        # We deliberately do NOT spin `while self.is_paused: processEvents()`
        # here — that busy-wait re-entered the Qt event loop and froze the UI
        # at 100% completion. Instead, when paused we simply skip the frame.
        if self.is_paused:
            return

        self.latest_recon = np.asarray(recon_image, dtype=np.float32)
        self._recon_2d = self.latest_recon
        elapsed_sec = time.time() - self.start_time

        # Update HUD Telemetry
        self.label_status.setText("Status: OPTIMIZING")
        self.label_epoch.setText(f"Epoch: {epoch} / {total_epochs}")
        self.label_loss.setText(f"Total Loss: {loss_val:.6f}")
        self.label_fps.setText(f"Speed: {step_fps:.1f} it/s")
        self.label_time.setText(f"Elapsed: {elapsed_sec:.1f}s")

        # Compute Voxel Statistics & Physical Plasma Temperature
        mean_rho = float(np.mean(self.latest_recon))
        peak_rho = float(np.max(self.latest_recon))
        # Relativistic synchrotron brightness temperature proxy (Sgr A* plasma ~ 10^10 to 10^11 K)
        t_base = 5.0e10  # Kelvin base synchrotron brightness temp scale
        avg_temp = mean_rho * t_base
        peak_temp = peak_rho * (t_base * 2.5)

        self.label_density_mean.setText(f"<b>Mean Density (ρ̄):</b> {mean_rho:.4f}")
        self.label_density_peak.setText(f"<b>Peak Density (ρ_max):</b> {peak_rho:.4f}")
        self.label_temp_avg.setText(f"<b>Avg Plasma Temp:</b> {avg_temp:.2e} K")
        self.label_temp_peak.setText(f"<b>Peak Core Temp:</b> {peak_temp:.2e} K")

        # Update Reconstruction Viewport
        recon_pixmap = self._tensor_to_qpixmap(self.latest_recon)
        self.recon_caption.setPixmap(recon_pixmap.scaled(
            self.recon_caption.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

        # Update Residual Viewport
        if self.latest_target is not None and self.latest_target.shape == self.latest_recon.shape:
            residual = np.abs(self.latest_target - self.latest_recon)
            res_pixmap = self._tensor_to_qpixmap(residual, colormap="viridis")
            self.residual_caption.setPixmap(res_pixmap.scaled(
                self.residual_caption.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            ))

        # Force synchronous repaints so panels never appear frozen/stale
        for cap in (self.target_caption, self.recon_caption, self.residual_caption):
            cap.repaint()
        QtWidgets.QApplication.processEvents()

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.setText("Resume Simulation")
            self.label_status.setText("Status: PAUSED")
            self.label_status.setStyleSheet("color: #f0883e; font-weight: bold; font-size: 13px;")
        else:
            self.btn_pause.setText("Pause Simulation")
            self.label_status.setText("Status: OPTIMIZING")
            self.label_status.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 13px;")
        # Keep the View menu checkbox in sync with the footer button
        if hasattr(self, "_act_pause") and self._act_pause.isChecked() != self.is_paused:
            self._act_pause.blockSignals(True)
            self._act_pause.setChecked(self.is_paused)
            self._act_pause.blockSignals(False)

    def _save_snapshot(self):
        if self.latest_recon is not None:
            out_dir = self._output_dir / "renders"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            out_path = out_dir / f"live_snapshot_{timestamp}.png"
            pixmap = self._tensor_to_qpixmap(self.latest_recon)
            pixmap.save(str(out_path), "PNG")
            self._latest_recon_path = out_path
            logger.info("Saved live snapshot to: %s", out_path)
            self.statusBar().showMessage(f"Snapshot saved: {out_path}", 5000)


def launch_engine_app(metadata: Optional[Dict[str, Any]] = None) -> Tuple[QtWidgets.QApplication, SimulationEngineWindow]:
    """Factory function to instantiate and launch the windowed engine with metadata."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    window = SimulationEngineWindow(metadata=metadata)
    window.show()
    window.raise_()
    window.activateWindow()
    QtWidgets.QApplication.processEvents()
    return app, window
