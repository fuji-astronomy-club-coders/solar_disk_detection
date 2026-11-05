"""Single-window Tkinter UI for interactive solar-disk review."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from tkinter import font as tkfont
from tkinter import ttk
from typing import cast

import numpy as np
from matplotlib.backend_bases import KeyEvent, PickEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle

PointList = list[list[float]]
CircleStatus = tuple[float, float, float]
DetectionResult = tuple[tuple[float, float], float]


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


class InteractiveReviewWindow:
    """Own one Tkinter window and reuse its embedded Matplotlib canvas.

    The Tkinter main loop runs on the caller thread. Detection runs on a worker
    thread and communicates with the UI through a thread-safe message queue.
    """

    def __init__(self) -> None:
        """Create the window and its single, reusable plot canvas."""
        self.root = tk.Tk()
        self.root.title("Solar Disk Detection - Interactive Review")
        self.root.geometry("1100x850")
        self.root.minsize(760, 560)
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
        self._finished = False
        self._result: DetectionResult | None = None
        self._error: Exception | None = None
        self._worker: threading.Thread | None = None

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        self._status = tk.StringVar(value="処理を準備しています...")
        ttk.Label(
            outer,
            textvariable=self._status,
            font=("Yu Gothic UI", 12, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 5))

        self._detail = tk.StringVar(value="画像を読み込んでいます。")
        ttk.Label(
            outer,
            textvariable=self._detail,
            anchor=tk.W,
            wraplength=1000,
        ).pack(fill=tk.X, pady=(0, 8))

        self._progress = ttk.Progressbar(outer, mode="indeterminate", length=260)
        self._progress.pack(fill=tk.X, pady=(0, 8))
        self._progress.start(12)

        plot_frame = ttk.Frame(outer)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(8, 0))

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

    def _show_processing_canvas(self) -> None:
        """Clear the plot while retaining the existing figure and canvas."""
        self.axes.clear()
        self.axes.set_axis_off()
        self.canvas.draw_idle()

    def _show_processing(self, message: str) -> None:
        """Update the progress message without creating another window."""
        self._active_request = None
        self._review_controls.pack_forget()
        self._progress.start(12)
        self._status.set("処理中")
        self._detail.set(message)
        self._review_count.set("")
        self._show_processing_canvas()

    def _show_review(self, request: _ReviewRequest) -> None:
        """Display a review stage in the persistent plot canvas."""
        self._progress.stop()
        self._active_request = request
        self._last_image = request.image
        self._excluded_indices.clear()
        self._status.set(request.title)
        self._detail.set(
            "画像上の点をクリックして除外・復帰できます。選択後に「選択を適用」を押してください。"
        )
        self._review_controls.pack(side=tk.RIGHT)
        self._close_button.pack_forget()
        self._draw_review_points()

    def _draw_review_points(self) -> None:
        """Redraw only the axes content; the Figure and Tk window are reused."""
        request = self._active_request
        if request is None:
            return

        points = np.asarray(request.spots, dtype=float)
        self.axes.clear()
        self.axes.imshow(request.image, cmap="magma")
        self.axes.add_patch(
            Circle(
                (request.circle[0], request.circle[1]),
                request.circle[2],
                fill=False,
                color="orange",
                linewidth=2,
            )
        )

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
            self.axes.annotate(
                f"#{index + 1}",
                (float(point[0]), float(point[1])),
                color="yellow",
                fontsize=9,
                xytext=(4, 4),
                textcoords="offset points",
            )

        self.axes.axis("equal")
        self.axes.set_axis_off()
        included = len(request.spots) - len(self._excluded_indices)
        self._review_count.set(
            f"採用点: {included} / {len(request.spots)}    "
            f"除外点: {len(self._excluded_indices)}"
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_pick(self, event: PickEvent) -> None:
        """Toggle a point when its marker is clicked."""
        indices = cast(
            np.ndarray,
            getattr(event, "ind", np.array([], dtype=int)),
        )
        if (
            self._active_request is None
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
        self._review_controls.pack_forget()
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
        self.axes.set_title("Final fitted circle")
        self.axes.axis("equal")
        self.canvas.draw_idle()
        self._close_button.pack(side=tk.RIGHT)

    def _show_error(self, error: Exception) -> None:
        """Show worker errors in the window and allow the user to close it."""
        self._finished = True
        self._error = error
        self._progress.stop()
        self._review_controls.pack_forget()
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
