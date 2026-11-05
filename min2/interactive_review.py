"""Single-window Tkinter UI for interactive solar-disk review."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from tkinter import font as tkfont
from tkinter import ttk
from typing import cast

import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent, PickEvent
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.patches import Circle
from matplotlib.text import Annotation
from matplotlib.widgets import LassoSelector, RectangleSelector

PointList = list[list[float]]
CircleStatus = tuple[float, float, float]
DetectionResult = tuple[tuple[float, float], float]
EdgeProfile = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class _ReviewRequest:
    """Data shared between the detection worker and the UI thread."""

    image: np.ndarray
    spots: PointList
    circle: CircleStatus
    title: str
    result: PointList | None = None
    completed: threading.Event = field(default_factory=threading.Event)


@dataclass
class _UiMessage:
    """A message sent from the detection worker to the Tkinter thread."""

    kind: str
    payload: object = None


def _indices_in_rectangle(
    spots: PointList,
    start: tuple[float, float],
    end: tuple[float, float],
) -> set[int]:
    """Return point indices located inside a rectangular data-coordinate region."""
    if not spots:
        return set()
    points = np.asarray(spots, dtype=float)[:, :2]
    x_min, x_max = sorted((start[0], end[0]))
    y_min, y_max = sorted((start[1], end[1]))
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    return set(np.flatnonzero(mask).tolist())


def _indices_in_polygon(
    spots: PointList,
    vertices: Sequence[tuple[float, float]],
) -> set[int]:
    """Return point indices located inside a free-form polygon."""
    if len(spots) == 0 or len(vertices) < 3:
        return set()
    points = np.asarray(spots, dtype=float)[:, :2]
    mask = Path(vertices).contains_points(points, radius=1e-9)
    return set(np.flatnonzero(mask).tolist())


def _extract_edge_profiles(
    image: np.ndarray,
    spots: PointList,
    n: int,
    window_size: int = 15,
) -> list[EdgeProfile]:
    """Extract brightness and first-derivative profiles around every edge point."""
    height, width = image.shape[:2]
    profiles: list[EdgeProfile] = []

    for spot in spots:
        x = int(np.clip(round(float(spot[0])), 0, width - 1))
        y = int(np.clip(round(float(spot[1])), 0, height - 1))

        if len(spot) > 3:
            is_x_line = int(spot[3]) == 0
        else:
            is_x_line = any(y == height * index // n for index in range(1, n))

        if is_x_line:
            line_data = image[y, :].astype(float)
            center_index = x
        else:
            line_data = image[:, x].astype(float)
            center_index = y

        start = max(0, center_index - window_size)
        end = min(len(line_data), center_index + window_size + 1)
        values = line_data[start:end]
        gradients = np.append(np.diff(line_data), 0)[start:end]
        coordinates = np.arange(start, end) - center_index
        profiles.append((coordinates, values, gradients))

    return profiles


class InteractiveReviewWindow:
    """Own one Tkinter window and reuse its embedded Matplotlib canvases."""

    def __init__(self) -> None:
        """Create the window, image toolbar, graph panel, and review controls."""
        self.root = tk.Tk()
        self.root.title("Solar Disk Detection - Interactive Review")
        self.root.geometry("1360x900")
        self.root.minsize(900, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(family="Yu Gothic UI", size=10)
        except tk.TclError:
            pass

        self._messages: queue.Queue[_UiMessage] = queue.Queue()
        self._active_request: _ReviewRequest | None = None
        self._last_image: np.ndarray | None = None
        self._excluded_indices: set[int] = set()
        self._scatter: PathCollection | None = None
        self._image_artist: AxesImage | None = None
        self._circle_artist: Circle | None = None
        self._point_annotations: list[Annotation] = []
        self._rectangle_selector: RectangleSelector | None = None
        self._lasso_selector: LassoSelector | None = None
        self._selection_mode = tk.StringVar(value="point")
        self._show_profiles = tk.BooleanVar(value=False)
        self._finished = False
        self._result: DetectionResult | None = None
        self._error: Exception | None = None
        self._worker: threading.Thread | None = None

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        self._status = tk.StringVar(value="処理を準備しています...")
        ttk.Label(
            outer,
            textvariable=self._status,
            font=("Yu Gothic UI", 12, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, sticky=tk.EW, pady=(0, 5))

        self._detail = tk.StringVar(value="画像を読み込んでいます。")
        ttk.Label(
            outer,
            textvariable=self._detail,
            anchor=tk.W,
            wraplength=1250,
        ).grid(row=1, column=0, sticky=tk.EW, pady=(0, 8))

        self._progress = ttk.Progressbar(outer, mode="indeterminate", length=260)
        self._progress.grid(row=2, column=0, sticky=tk.EW, pady=(0, 8))
        self._progress.start(12)

        options = ttk.Frame(outer)
        options.grid(row=3, column=0, sticky=tk.EW, pady=(0, 6))

        self._profile_toggle = ttk.Checkbutton(
            options,
            text="縁近傍の輝度グラフを表示",
            variable=self._show_profiles,
            command=self._toggle_profiles,
        )
        self._profile_toggle.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(options, text="範囲選択:").pack(side=tk.LEFT, padx=(0, 5))
        for label, value in (
            ("点をクリック", "point"),
            ("矩形範囲", "rectangle"),
            ("自由範囲", "lasso"),
        ):
            ttk.Radiobutton(
                options,
                text=label,
                value=value,
                variable=self._selection_mode,
                command=self._update_selection_mode,
            ).pack(side=tk.LEFT, padx=3)

        self._plot_frame = ttk.Frame(outer)
        self._plot_frame.grid(row=4, column=0, sticky=tk.NSEW)
        self._plot_frame.columnconfigure(0, weight=3)
        self._plot_frame.columnconfigure(1, weight=2)
        self._plot_frame.rowconfigure(0, weight=1)
        self._plot_frame.rowconfigure(0, weight=1)

        self._image_frame = ttk.Frame(self._plot_frame)
        self._image_frame.grid(row=0, column=0, sticky=tk.NSEW)
        self._image_frame.rowconfigure(0, weight=1)
        self._image_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self._image_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)

        toolbar_frame = ttk.Frame(self._image_frame)
        toolbar_frame.grid(row=1, column=0, sticky=tk.EW)
        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            toolbar_frame,
            pack_toolbar=False,
        )
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._profile_frame = ttk.LabelFrame(
            self._plot_frame,
            text="縁近傍の輝度プロファイル",
            padding=4,
        )
        self._profile_frame.rowconfigure(0, weight=1)
        self._profile_frame.columnconfigure(0, weight=1)
        self._profile_scroll = tk.Canvas(
            self._profile_frame,
            highlightthickness=0,
            width=500,
        )
        profile_scrollbar = ttk.Scrollbar(
            self._profile_frame,
            orient=tk.VERTICAL,
            command=self._profile_scroll.yview,
        )
        self._profile_scroll.configure(yscrollcommand=profile_scrollbar.set)
        self._profile_scroll.grid(row=0, column=0, sticky=tk.NSEW)
        profile_scrollbar.grid(row=0, column=1, sticky=tk.NS)

        self._profile_inner = ttk.Frame(self._profile_scroll)
        self._profile_window = self._profile_scroll.create_window(
            (0, 0),
            window=self._profile_inner,
            anchor=tk.NW,
        )
        self._profile_inner.bind("<Configure>", self._on_profile_inner_configure)
        self._profile_scroll.bind("<Configure>", self._on_profile_scroll_configure)

        self.profile_figure = Figure(figsize=(5.1, 3), dpi=90)
        self.profile_canvas = FigureCanvasTkAgg(
            self.profile_figure,
            master=self._profile_inner,
        )
        self.profile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._profile_frame.grid_remove()

        footer = ttk.Frame(outer)
        footer.grid(row=5, column=0, sticky=tk.EW, pady=(8, 0))

        self._review_count = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self._review_count).pack(side=tk.LEFT)

        self._review_controls = ttk.Frame(footer)
        self._apply_button = ttk.Button(
            self._review_controls,
            text="選択を適用",
            command=self._apply_selection,
        )
        self._apply_button.pack(side=tk.LEFT, padx=(0, 6))
        self._cancel_button = ttk.Button(
            self._review_controls,
            text="変更を取り消す",
            command=self._cancel_selection,
        )
        self._cancel_button.pack(side=tk.LEFT)
        self._review_controls.pack_forget()

        self._close_button = ttk.Button(
            footer,
            text="閉じる",
            command=self.root.destroy,
        )
        self._close_button.pack(side=tk.RIGHT)
        self._close_button.pack_forget()

        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self._show_processing_canvas()

    def run(self, target: Callable[[], DetectionResult]) -> DetectionResult:
        """Run detection on a worker while keeping Tkinter on the main thread."""
        self._worker = threading.Thread(
            target=self._run_target,
            args=(target,),
            name="solar-disk-detection",
            daemon=True,
        )
        self._worker.start()
        self.root.after(40, self._process_messages)
        self.root.mainloop()

        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("検出処理が結果を返さずに終了しました。")
        return self._result

    def update_processing(self, message: str) -> None:
        """Queue a processing-status update from the worker thread."""
        self._messages.put(_UiMessage("status", message))

    def review(
        self,
        image: np.ndarray,
        spots: PointList,
        circle: CircleStatus,
        title: str,
    ) -> PointList:
        """Display points in the existing window and wait for a user decision."""
        request = _ReviewRequest(
            image=image,
            spots=[list(point) for point in spots],
            circle=circle,
            title=title,
        )
        self._messages.put(_UiMessage("review", request))
        request.completed.wait()
        return request.result if request.result is not None else spots

    def _run_target(self, target: Callable[[], DetectionResult]) -> None:
        """Execute the detector without blocking the Tk event loop."""
        try:
            result = target()
        except Exception as error:
            self._messages.put(_UiMessage("error", error))
        else:
            self._messages.put(_UiMessage("finished", result))

    def _process_messages(self) -> None:
        """Apply queued worker updates on the Tkinter main thread."""
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break

            if message.kind == "status":
                self._show_processing(cast(str, message.payload))
            elif message.kind == "review":
                self._show_review(cast(_ReviewRequest, message.payload))
            elif message.kind == "finished":
                self._show_finished(cast(DetectionResult, message.payload))
            elif message.kind == "error":
                self._show_error(cast(Exception, message.payload))

        if self.root.winfo_exists():
            self.root.after(40, self._process_messages)

    def _on_profile_inner_configure(self, _event: tk.Event[tk.Misc]) -> None:
        """Update the scrollable brightness-graph region."""
        self._profile_scroll.configure(
            scrollregion=self._profile_scroll.bbox("all")
        )

    def _on_profile_scroll_configure(self, event: tk.Event[tk.Misc]) -> None:
        """Keep the embedded graph canvas as wide as its viewport."""
        self._profile_scroll.itemconfigure(
            self._profile_window,
            width=event.width,
        )

    def _disconnect_selectors(self) -> None:
        """Disable range selectors before changing the reviewed point set."""
        for selector in (self._rectangle_selector, self._lasso_selector):
            if selector is not None:
                selector.set_active(False)
                selector.disconnect_events()
        self._rectangle_selector = None
        self._lasso_selector = None

    def _show_processing_canvas(self) -> None:
        """Clear the image area while preserving its Figure and toolbar."""
        self._disconnect_selectors()
        self.axes.clear()
        self.axes.set_axis_off()
        self._image_artist = None
        self._circle_artist = None
        self._scatter = None
        self._point_annotations.clear()
        self.canvas.draw_idle()

    def _show_processing(self, message: str) -> None:
        """Update the processing state without creating another window."""
        self._active_request = None
        self._disconnect_selectors()
        self._review_controls.pack_forget()
        self._profile_frame.grid_remove()
        self._progress.start(12)
        self._status.set("処理中")
        self._detail.set(message)
        self._review_count.set("")
        self._show_processing_canvas()

    def _show_review(self, request: _ReviewRequest) -> None:
        """Display a review stage in the persistent plot canvases."""
        self._progress.stop()
        self._active_request = request
        self._last_image = request.image
        self._excluded_indices.clear()
        self._status.set(request.title)
        self._detail.set(
            "点をクリックして個別に切り替えるか、矩形／自由範囲で囲んだ点を除外できます。"
        )
        self._review_controls.pack(side=tk.RIGHT)
        self._close_button.pack_forget()
        self._draw_review_points()
        self._toggle_profiles()
        self._update_selection_mode()

    def _draw_review_points(self) -> None:
        """Update the image, circle, and point artists without resetting the toolbar."""
        request = self._active_request
        if request is None:
            return

        if self._image_artist is None:
            self.axes.clear()
            self._image_artist = self.axes.imshow(request.image, cmap="magma")
            self._circle_artist = Circle(
                (request.circle[0], request.circle[1]),
                request.circle[2],
                fill=False,
                color="orange",
                linewidth=2,
            )
            self.axes.add_patch(self._circle_artist)
            self.axes.set_aspect("equal", adjustable="box")
        else:
            self._image_artist.set_data(request.image)
            if self._circle_artist is not None:
                self._circle_artist.center = (request.circle[0], request.circle[1])
                self._circle_artist.set_radius(request.circle[2])

        if self._scatter is not None:
            self._scatter.remove()
        for annotation in self._point_annotations:
            annotation.remove()
        self._point_annotations.clear()

        points = np.asarray(request.spots, dtype=float)
        colors = [
            "tab:orange" if index in self._excluded_indices else "tab:green"
            for index in range(len(request.spots))
        ]
        self._scatter = self.axes.scatter(
            points[:, 0],
            points[:, 1],
            c=colors,
            s=85,
            edgecolors="white",
            linewidths=1.2,
            picker=True,
        )

        for index, point in enumerate(request.spots):
            self._point_annotations.append(
                self.axes.annotate(
                    f"#{index + 1}",
                    (float(point[0]), float(point[1])),
                    color="yellow",
                    fontsize=9,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
            )

        included = len(request.spots) - len(self._excluded_indices)
        self._review_count.set(
            f"採用点: {included} / {len(request.spots)}    "
            f"除外点: {len(self._excluded_indices)}"
        )
        self.canvas.draw_idle()

    def _toggle_profiles(self) -> None:
        """Show or hide the edge-neighborhood brightness graphs."""
        request = self._active_request
        if not self._show_profiles.get() or request is None:
            self._profile_frame.grid_remove()
            return
        self._profile_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(6, 0))
        self._draw_brightness_profiles()

    def _draw_brightness_profiles(self) -> None:
        """Render the original brightness/gradient mini-graphs in a scrollable pane."""
        request = self._active_request
        if request is None:
            return

        profiles = _extract_edge_profiles(
            request.image,
            request.spots,
            n=10,
            window_size=15,
        )
        # Give every edge point a full-width row.  A two-column layout
        # makes titles, tick labels, and traces unreadable on a narrow pane.
        columns = 1
        rows = max(1, len(profiles))
        row_height_inches = 1.9
        figure_width_inches = 6.0
        figure_height_inches = max(2.4, rows * row_height_inches)
        self.profile_figure.clear()
        self.profile_figure.set_size_inches(
            figure_width_inches,
            figure_height_inches,
            forward=True,
        )
        # TkAgg does not reliably resize its Tk widget when only the Figure
        # size changes, so explicitly resize the widget to match the scrollable
        # figure's requested height.
        self.profile_canvas.get_tk_widget().configure(
            height=round(figure_height_inches * self.profile_figure.dpi),
        )
        grid = GridSpec(rows, columns, figure=self.profile_figure, hspace=0.9)

        for index, (coordinates, brightness, gradient) in enumerate(profiles):
            axis = self.profile_figure.add_subplot(grid[index, 0])
            axis.set_title(f"#{index + 1}", fontsize=9, fontweight="bold")
            axis.plot(
                coordinates,
                brightness,
                color="tab:orange",
                linewidth=1.3,
            )
            axis.tick_params(axis="y", labelcolor="tab:orange", labelsize=7)
            axis.tick_params(axis="x", labelsize=7)
            axis.grid(alpha=0.3)

            gradient_axis = axis.twinx()
            gradient_axis.plot(
                coordinates,
                gradient,
                color="tab:cyan",
                linewidth=1.1,
                linestyle="--",
            )
            gradient_axis.tick_params(
                axis="y",
                labelcolor="tab:cyan",
                labelsize=7,
            )
            axis.axvline(0, color="red", linewidth=1, alpha=0.5)

        self.profile_canvas.draw_idle()
        self._profile_scroll.configure(
            scrollregion=self._profile_scroll.bbox("all")
        )

    def _update_selection_mode(self) -> None:
        """Switch between point, rectangle, and free-form area selection."""
        self._disconnect_selectors()
        request = self._active_request
        if request is None or self._selection_mode.get() == "point":
            return

        if self.toolbar.mode:
            self._detail.set(
                "範囲選択の前に、ツールバーのパン／ズームを解除してください。"
            )
            return

        if self._selection_mode.get() == "rectangle":
            self._rectangle_selector = RectangleSelector(
                self.axes,
                self._on_rectangle_select,
                useblit=True,
                button=[MouseButton.LEFT],
                minspanx=1,
                minspany=1,
                spancoords="data",
                interactive=False,
                props={"facecolor": "tab:blue", "edgecolor": "tab:blue", "alpha": 0.2},
            )
            self._detail.set("画像上をドラッグして矩形範囲を指定します。範囲内の点を除外します。")
        elif self._selection_mode.get() == "lasso":
            self._lasso_selector = LassoSelector(
                self.axes,
                self._on_lasso_select,
                useblit=True,
                button=MouseButton.LEFT,
                props={"color": "tab:blue", "linewidth": 1.5},
            )
            self._detail.set("画像上をドラッグして自由範囲を描きます。範囲内の点を除外します。")

    def _on_rectangle_select(self, start: MouseEvent, end: MouseEvent) -> None:
        """Exclude all points inside a rectangular selection."""
        if start.xdata is None or start.ydata is None:
            return
        if end.xdata is None or end.ydata is None:
            return
        indices = _indices_in_rectangle(
            self._active_request.spots if self._active_request else [],
            (float(start.xdata), float(start.ydata)),
            (float(end.xdata), float(end.ydata)),
        )
        self._exclude_selected_indices(indices)

    def _on_lasso_select(self, vertices: list[tuple[float, float]]) -> None:
        """Exclude all points inside a free-form selection."""
        indices = _indices_in_polygon(
            self._active_request.spots if self._active_request else [],
            vertices,
        )
        self._exclude_selected_indices(indices)

    def _exclude_selected_indices(self, indices: set[int]) -> None:
        """Add selected points to the exclusion set and refresh the preview."""
        if not indices:
            self._detail.set("範囲内に検出点がありませんでした。")
            return
        self._excluded_indices.update(indices)
        self._draw_review_points()
        self._detail.set(f"範囲選択で {len(indices)} 点を除外しました。")

    def _on_pick(self, event: PickEvent) -> None:
        """Toggle a single point when point-selection mode is active."""
        indices = cast(
            np.ndarray,
            getattr(event, "ind", np.array([], dtype=int)),
        )
        if (
            self._active_request is None
            or self._selection_mode.get() != "point"
            or event.artist is not self._scatter
            or indices.size == 0
        ):
            return

        index = int(indices[0])
        if index in self._excluded_indices:
            self._excluded_indices.remove(index)
        else:
            self._excluded_indices.add(index)
        self._draw_review_points()

    def _on_key(self, event: KeyEvent) -> None:
        """Support Enter and Escape while the plot canvas has focus."""
        if event.key in {"enter", "return"}:
            self._apply_selection()
        elif event.key == "escape":
            self._cancel_selection()

    def _apply_selection(self) -> None:
        """Accept the selected points and release the worker thread."""
        request = self._active_request
        if request is None:
            return

        reviewed = [
            point
            for index, point in enumerate(request.spots)
            if index not in self._excluded_indices
        ]
        if len(reviewed) < 3:
            self._detail.set("円の計算には3点以上必要です。除外点を戻してください。")
            return

        request.result = reviewed
        self._complete_review(request)

    def _cancel_selection(self) -> None:
        """Discard manual edits and keep all points."""
        request = self._active_request
        if request is None:
            return
        request.result = request.spots
        self._complete_review(request)

    def _complete_review(self, request: _ReviewRequest) -> None:
        """Return from the review stage and resume detection."""
        self._active_request = None
        self._disconnect_selectors()
        self._review_controls.pack_forget()
        self._profile_frame.grid_remove()
        self._status.set("処理中")
        self._detail.set("選択内容を反映して検出を続けています。")
        self._progress.start(12)
        self._show_processing_canvas()
        request.completed.set()

    def _show_finished(self, result: DetectionResult) -> None:
        """Render the final fitted circle in the same window."""
        self._finished = True
        self._result = result
        self._progress.stop()
        self._review_controls.pack_forget()
        self._profile_frame.grid_remove()
        self._status.set("検出が完了しました")
        (cx, cy), radius = result
        self._detail.set(
            f"中心座標: ({cx:.2f}, {cy:.2f})    半径: {radius:.2f} pix"
        )
        self.axes.clear()
        if self._last_image is not None:
            self.axes.imshow(self._last_image, cmap="magma")
        self.axes.add_patch(
            Circle(
                (cx, cy),
                radius,
                fill=False,
                color="orange",
                linewidth=2,
            )
        )
        self.axes.set_aspect("equal", adjustable="box")
        self.canvas.draw_idle()
        self.toolbar.update()
        self._close_button.pack(side=tk.RIGHT)

    def _show_error(self, error: Exception) -> None:
        """Show worker errors in the window and allow the user to close it."""
        self._finished = True
        self._error = error
        self._progress.stop()
        self._review_controls.pack_forget()
        self._profile_frame.grid_remove()
        self._status.set("処理中にエラーが発生しました")
        self._detail.set(str(error))
        self._show_processing_canvas()
        self._close_button.pack(side=tk.RIGHT)

    def _on_window_close(self) -> None:
        """Prevent closing while the worker is waiting or still processing."""
        if not self._finished:
            self._detail.set("処理中です。完了後に閉じることができます。")
            return
        self.root.destroy()
