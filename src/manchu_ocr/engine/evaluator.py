from collections import defaultdict
from typing import Dict, Iterable


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def average_dicts(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    meters = defaultdict(AverageMeter)
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                meters[key].update(float(value))
    return {key: meter.avg for key, meter in meters.items()}
