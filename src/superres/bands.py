"""SuperDove band definitions and RGB pair specifications.

Planet's 8-band PSScene products ship bands in this order (1-indexed, as
rasterio returns them):

    1: Coastal Blue   (431-452 nm)
    2: Blue           (465-515 nm)
    3: Green I        (513-549 nm)
    4: Green          (547-583 nm)
    5: Yellow         (600-620 nm)
    6: Red            (650-680 nm)
    7: Red Edge       (697-713 nm)
    8: NIR            (845-885 nm)

If you receive a product with a different band order, override the index map
when constructing pairs.
"""
from __future__ import annotations

from dataclasses import dataclass


SUPERDOVE_BAND_INDEX: dict[str, int] = {
    "coastal_blue": 1,
    "blue": 2,
    "green_i": 3,
    "green": 4,
    "yellow": 5,
    "red": 6,
    "red_edge": 7,
    "nir": 8,
}


@dataclass(frozen=True)
class BandPair:
    """A pair of source bands that fuse into one super-resolved RGB channel."""

    name: str  # "blue" | "green" | "red"
    band_a: str
    band_b: str

    def indices(self, index_map: dict[str, int] = SUPERDOVE_BAND_INDEX) -> tuple[int, int]:
        return index_map[self.band_a], index_map[self.band_b]


# Default pairings — Yellow chosen over Red Edge for the red channel to keep
# vegetation chromatically natural. Swap to Red Edge if you want NDVI-flavored
# false-color output.
DEFAULT_PAIRS: tuple[BandPair, BandPair, BandPair] = (
    BandPair(name="blue", band_a="coastal_blue", band_b="blue"),
    BandPair(name="green", band_a="green_i", band_b="green"),
    BandPair(name="red", band_a="red", band_b="yellow"),
)


RED_EDGE_PAIRS: tuple[BandPair, BandPair, BandPair] = (
    BandPair(name="blue", band_a="coastal_blue", band_b="blue"),
    BandPair(name="green", band_a="green_i", band_b="green"),
    BandPair(name="red", band_a="red", band_b="red_edge"),
)
