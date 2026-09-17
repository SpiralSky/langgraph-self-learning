"""Online (Welford) statistics for per-node run telemetry.

``OnlineStats`` accumulates count / mean / variance in a single pass with no
history retention, so it stays cheap for unbounded node collections. Only
stdlib ``math`` is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class OnlineStats:
    """Welford online mean/std accumulator (single-pass, no history)."""

    count: int = 0
    mean: float = 0.0
    _m2: float = field(default=0.0, repr=False)

    def update(self, value: float) -> None:
        """Incorporate ``value`` using the stable Welford recurrence."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    @property
    def std(self) -> float | None:
        """Sample standard deviation, or ``None`` with fewer than 2 values."""
        if self.count < 2:
            return None
        return sqrt(self._m2 / (self.count - 1))

    def to_dict(self) -> dict[str, float | None]:
        """Serialize to the portable ``{"count", "mean", "std"}`` form."""
        return {"count": self.count, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict) -> OnlineStats:
        """Rebuild from :meth:`to_dict`, restoring ``_m2`` for continuation."""
        count = int(data["count"])
        mean = float(data["mean"]) if data["mean"] is not None else 0.0
        std = data.get("std")
        m2 = std * std * (count - 1) if count > 1 and std is not None else 0.0
        return cls(count=count, mean=mean, _m2=m2)