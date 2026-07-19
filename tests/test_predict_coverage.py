"""V11 y V5 sobre la misma ventana, y la rejilla que deja huecos.

Nace de un fallo observado: en `dirty-paragraphs-80ancho` las esquinas de V11 no
coincidían con las de V5, y la sospecha razonable era que uno de los dos calculaba
mal la posición. **No era eso.** Los dos hacen `win.x0 + frac * n` y coinciden
hasta el último decimal; lo que no coincidía era *qué ventanas* miraba cada uno.
La pantalla mandaba `stride = 20`, un número escrito para patches de 40 px, y el
run tenía patches de **10**: la rejilla saltaba de 20 en 20 con una ventana de 10,
así que **30 de 80 columnas no las veía nadie**.

Ese fallo no lanza ninguna excepción. Devuelve una predicción bien formada a la
que le faltan las esquinas de las franjas que nadie miró, y en pantalla eso es
indistinguible de un modelo que no las detectó. Así que hay dos invariantes que
fijar, y son de la misma familia que el contrato ⑤: no «¿es correcta la fórmula?»
sino «¿ven lo mismo los dos que tienen que ver lo mismo?».
"""

from __future__ import annotations

import numpy as np
import pytest

from itf.geometry import windows
from itf.inference.predict import (
    detect_corners,
    grid_coverage,
    predict_image,
    window_prediction,
)
from itf.models import ConfigurableCNN, build_model


@pytest.fixture
def model() -> ConfigurableCNN:
    """Una red pequeña de 10 px: el tamaño que destapó el fallo."""
    return build_model(
        {
            "input_size": 10,
            "in_channels": 1,
            "border_features": True,
            "backbone": [{"filters": 4, "kernel": 3, "stride": 1, "padding": 1, "pool": 2}],
            "head": {"hidden": [8]},
        }
    )


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(60, 80), dtype=np.uint8)


def test_v5_and_v11_agree_on_a_window_they_share(model, image):
    """**La pregunta del usuario, aislada.**

    Para una ventana que está en la rejilla de V11, la detección cruda y lo que
    V5 dice de esa misma ventana tienen que ser el mismo número. Si divergen, uno
    de los dos convierte mal de coordenadas del patch a píxeles de la imagen, y
    la pantalla dibuja dos verdades distintas del mismo forward.
    """
    n, stride = model.config.input_size, 5
    grid = windows(80, 60, n, stride)
    # Umbral 0 para que la comparación no dependa de qué detecta el modelo: lo
    # que se afirma es la GEOMETRÍA, no la calidad de la red.
    raw = detect_corners(model, image, stride=stride, threshold=0.0, device="cpu")

    for gi in (0, 7, 40, len(grid) - 1):
        g = grid[gi]
        v5 = window_prediction(model, image, g.x0, g.y0, device="cpu")
        assert (v5["x0"], v5["y0"]) == (g.x0, g.y0)
        for c, corner in enumerate(("TL", "TR", "BR", "BL")):
            here = [
                d
                for d in raw
                if d.corner == corner
                and abs(d.x - v5["corners"][c]["image_x"]) < 1e-2
                and abs(d.y - v5["corners"][c]["image_y"]) < 1e-2
            ]
            assert here, (
                f"V11 no tiene ninguna detección {corner} donde V5 la pone "
                f"({v5['corners'][c]['image_x']}, {v5['corners'][c]['image_y']}) "
                f"para la ventana {(g.x0, g.y0)}"
            )
            assert here[0].score == pytest.approx(v5["corners"][c]["score"], abs=1e-3)


def test_a_stride_above_the_patch_leaves_gaps_and_says_so(model, image):
    """El fallo silencioso, hecho ruidoso.

    Con ventana de 10 y paso 20 hay franjas entre ventanas que ninguna ve. La
    predicción sale igualmente — es F, un stride grosero es legítimo — pero el
    payload tiene que declararlo, porque «no hay esquina ahí» y «nadie miró ahí»
    se dibujan idénticos.
    """
    payload = predict_image(model, image, stride=20, threshold=0.5, device="cpu")

    assert payload["patch_size"] == 10
    cov = payload["coverage"]
    assert cov["has_gaps"] is True
    assert cov["unseen_columns"] == 30  # 80 px, ventanas en 0/20/40/60/70
    assert cov["unseen_rows"] > 0


def test_the_default_stride_covers_every_pixel(model, image):
    """Y el default no los deja: sin `stride`, F elige `n / 2`.

    Es lo que hace que «no mandar stride» sea la respuesta correcta del cliente,
    en vez de una constante que significa algo distinto en cada run.
    """
    payload = predict_image(model, image, threshold=0.5, device="cpu")

    assert payload["knobs"]["stride"] == 5  # la mitad del patch
    cov = payload["coverage"]
    assert cov["has_gaps"] is False
    assert (cov["unseen_columns"], cov["unseen_rows"]) == (0, 0)


@pytest.mark.parametrize(
    "width,height,n,stride,gaps",
    [
        (80, 60, 10, 20, True),
        (80, 60, 10, 10, False),  # justo a tope: se tocan, no se solapan
        (80, 60, 10, 5, False),
        (80, 60, 40, 20, False),  # el caso para el que se escribió el 20
    ],
)
def test_grid_coverage_is_the_one_definition_of_a_gap(width, height, n, stride, gaps):
    """`has_gaps` es exactamente `stride > n`, y las cuentas lo respaldan."""
    cov = grid_coverage(width, height, n, stride)
    assert cov["has_gaps"] is gaps
    assert (cov["unseen_columns"] > 0 or cov["unseen_rows"] > 0) is gaps
