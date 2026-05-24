from py123d.datatypes import DefaultBoxDetectionLabel

DYNAMIC_OBJECT_LABELS = {
    DefaultBoxDetectionLabel.VEHICLE,
    DefaultBoxDetectionLabel.PERSON,
    DefaultBoxDetectionLabel.TWO_WHEELER,
    DefaultBoxDetectionLabel.ANIMAL,
    DefaultBoxDetectionLabel.TRAIN,
    DefaultBoxDetectionLabel.OTHER,
}

STATIC_OBJECT_LABELS = {
    DefaultBoxDetectionLabel.TRAFFIC_SIGN,
    DefaultBoxDetectionLabel.TRAFFIC_CONE,
    DefaultBoxDetectionLabel.TRAFFIC_LIGHT,
    DefaultBoxDetectionLabel.BARRIER,
    DefaultBoxDetectionLabel.GENERIC_OBJECT,
}
