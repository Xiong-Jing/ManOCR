from typing import List

from manchu_ocr.models.detection.necks.db_fpn import DBFPN
from manchu_ocr.models.registry import NECKS


@NECKS.register("DBFPNVSAA")
class DBFPNVSAA(DBFPN):
    """Compatibility wrapper for DBFPN with VSAA enabled."""

    def __init__(
        self,
        in_channels: List[int],
        inner_channels: int = 128,
        out_channels: int = 256,
        use_asf: bool = True,
        vsaa_reduction: int = 4,
        vsaa_vertical_kernel: int = 15,
        vsaa_horizontal_kernel: int = 5,
        vsaa_dropout: float = 0.0,
    ):
        super().__init__(
            in_channels=in_channels,
            inner_channels=inner_channels,
            out_channels=out_channels,
            use_asf=use_asf,
            use_vsaa=True,
            vsaa_reduction=vsaa_reduction,
            vsaa_vertical_kernel=vsaa_vertical_kernel,
            vsaa_horizontal_kernel=vsaa_horizontal_kernel,
            vsaa_dropout=vsaa_dropout,
        )
