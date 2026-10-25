"""Optional ipywidgets GUI (parity with the legacy notebook's last cell).

Importing this module does not require ipywidgets; only ``build_ui`` does, so
the package stays usable on headless machines.
"""

from __future__ import annotations

from typing import Any

from .predict import NewsClassifier, format_prediction


def build_ui(classifier: NewsClassifier, default_text: str = "") -> Any:
    """Create the widget tree; the caller decides where to display it."""

    try:
        import ipywidgets as widgets
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "ipywidgets is required for the notebook UI. "
            "Install it with `pip install ipywidgets`."
        ) from exc

    text_input = widgets.Text(
        value=default_text,
        placeholder="테스트할 뉴스 제목을 입력하세요...",
        description="뉴스 제목:",
        layout=widgets.Layout(width="75%"),
    )
    run_button = widgets.Button(
        description="전체 점수 측정",
        button_style="success",
        layout=widgets.Layout(width="18%"),
    )
    output = widgets.Output(
        layout=widgets.Layout(
            border="2px solid #28A745",
            border_radius="6px",
            padding="15px",
            margin="10px 0px",
        )
    )

    def run(_: Any = None) -> None:
        with output:
            output.clear_output()
            title = text_input.value.strip()
            if not title:
                print("⚠️ 뉴스 제목을 입력해 주세요.")
                return
            try:
                prediction = classifier.predict(title)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                print(f"❌ 점수 측정 오류 발생: {exc}")
                return
            print(format_prediction(prediction))

    run_button.on_click(run)
    text_input.on_submit(run)
    return widgets.VBox([widgets.HBox([text_input, run_button]), output])


def launch_ui(classifier: NewsClassifier, default_text: str = "") -> Any:
    """Build and display the GUI inside a Jupyter/Colab session."""

    try:
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The GUI needs IPython/Jupyter. Use `news-classifier predict` instead."
        ) from exc

    ui = build_ui(classifier, default_text=default_text)
    display(ui)
    return ui
