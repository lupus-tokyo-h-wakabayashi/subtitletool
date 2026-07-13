#!/usr/bin/env python3
from dataclasses import dataclass, field


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class ProgressTracker:
    total_chunks: int
    durations: list[float] = field(default_factory=list)

    def add(self, duration: float) -> None:
        self.durations.append(duration)

    @property
    def completed_chunks(self) -> int:
        return len(self.durations)

    @property
    def average_seconds(self) -> float:
        if not self.durations:
            return 0.0

        return sum(self.durations) / len(self.durations)

    @property
    def remaining_chunks(self) -> int:
        return max(0, self.total_chunks - self.completed_chunks)

    @property
    def eta_seconds(self) -> float:
        return self.average_seconds * self.remaining_chunks

    @property
    def progress_percent(self) -> float:
        if self.total_chunks == 0:
            return 100.0

        return self.completed_chunks / self.total_chunks * 100

    @property
    def fastest_seconds(self) -> float:
        return min(self.durations) if self.durations else 0.0

    @property
    def slowest_seconds(self) -> float:
        return max(self.durations) if self.durations else 0.0