from dataclasses import dataclass


@dataclass(frozen=True)
class ShelfLifeEstimate:
    minimum_days: int
    maximum_days: int
    method: str

    def display(self) -> str:
        if self.minimum_days == self.maximum_days:
            return f"{self.minimum_days} day(s)"
        return f"{self.minimum_days}-{self.maximum_days} days"


# This is deliberately a PROTOTYPE fallback.
#
# It is NOT trained from your current dataset because your dataset
# contains ripeness classes, not "days remaining" labels.
#
# Later, replace this with a real regression/time-to-event model
# trained on longitudinal observations of bananas.
STAGE_RANGES = {
    "unripe": (3, 7),
    "ripe": (2, 4),
    "overripe": (0, 2),
    "rotten": (0, 0),
}


def prototype_estimate(stage: str) -> str:
    low, high = STAGE_RANGES.get(stage, (0, 0))
    return ShelfLifeEstimate(
        minimum_days=low,
        maximum_days=high,
        method="prototype_stage_rule",
    ).display()
