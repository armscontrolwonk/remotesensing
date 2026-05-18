"""Planet metadata sidecar parsing.

Planet's PSScene products ship a ``<stem>_metadata.xml`` next to each scene
GeoTIFF. The XML contains per-band radiometric coefficients used to convert
raw DN to TOA radiance and TOA reflectance. For Analytic Radiance (TOAR)
products we want ``reflectanceCoefficient`` (DN × coeff → TOA reflectance,
roughly 0-1). For Surface Reflectance products the DN→reflectance scaling
is the well-known 0-10000 convention and no XML lookup is required.

The parser uses local-name matching so it is robust to namespace prefix
changes between Planet schema versions.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SceneCalibration:
    """Per-band coefficients to convert DN to TOA reflectance in [0, 1]."""

    reflectance_coefficients: dict[int, float]  # 1-indexed band number → coefficient

    def coefficient(self, band_index: int) -> float:
        try:
            return self.reflectance_coefficients[band_index]
        except KeyError as e:
            raise KeyError(
                f"no reflectanceCoefficient for band {band_index}; "
                f"available bands: {sorted(self.reflectance_coefficients)}"
            ) from e

    @classmethod
    def surface_reflectance(cls, n_bands: int = 8) -> "SceneCalibration":
        """Constant 1/10000 coefficient for Planet SR products."""
        return cls(reflectance_coefficients={i: 1.0 / 10000.0 for i in range(1, n_bands + 1)})

    @classmethod
    def from_planet_xml(cls, xml_path: str | Path) -> "SceneCalibration":
        """Parse per-band ``reflectanceCoefficient`` values from a Planet XML sidecar."""
        tree = ET.parse(xml_path)
        coefficients: dict[int, float] = {}
        for elem in tree.iter():
            if _local(elem.tag) != "bandSpecificMetadata":
                continue
            band_num: int | None = None
            coeff: float | None = None
            for child in elem:
                name = _local(child.tag)
                text = (child.text or "").strip()
                if not text:
                    continue
                if name == "bandNumber":
                    band_num = int(text)
                elif name == "reflectanceCoefficient":
                    coeff = float(text)
            if band_num is not None and coeff is not None:
                coefficients[band_num] = coeff
        if not coefficients:
            raise ValueError(
                f"no bandSpecificMetadata with reflectanceCoefficient in {xml_path}"
            )
        return cls(reflectance_coefficients=coefficients)


def _local(tag: str) -> str:
    """Strip the XML namespace prefix from an ElementTree tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_planet_metadata(scene_path: str | Path) -> Path | None:
    """Locate the ``<stem>_metadata.xml`` sidecar for a Planet scene GeoTIFF.

    Falls back to any ``*_metadata.xml`` in the same directory whose stem is
    a prefix of the scene stem — handles cases where the scene file has an
    extra suffix not present in the metadata filename.
    """
    p = Path(scene_path)
    direct = p.with_name(f"{p.stem}_metadata.xml")
    if direct.exists():
        return direct
    suffix_len = len("_metadata.xml")
    candidates = sorted(
        x for x in p.parent.glob("*_metadata.xml")
        if p.stem.startswith(x.name[:-suffix_len])
    )
    return candidates[0] if candidates else None


def resolve_calibration(
    scene_path: str | Path,
    product: str = "toar",
    n_bands: int = 8,
) -> SceneCalibration:
    """Return the right calibration for a scene given its product type.

    - ``product="toar"`` — parse the Planet XML sidecar; raise if missing.
    - ``product="sr"``   — return the constant 1/10000 SR-scale calibration.
    """
    if product == "sr":
        return SceneCalibration.surface_reflectance(n_bands=n_bands)
    if product == "toar":
        xml_path = find_planet_metadata(scene_path)
        if xml_path is None:
            raise FileNotFoundError(
                f"no _metadata.xml sidecar found next to {scene_path}; "
                "if this is a Surface Reflectance product, set product='sr'"
            )
        return SceneCalibration.from_planet_xml(xml_path)
    raise ValueError(f"unknown product {product!r}; expected 'toar' or 'sr'")
