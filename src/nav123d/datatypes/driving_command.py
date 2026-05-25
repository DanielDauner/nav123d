from py123d.common.utils.enums import SerialIntEnum


class DrivingCommand(SerialIntEnum):
    """Enum for high-level driving commands."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3
