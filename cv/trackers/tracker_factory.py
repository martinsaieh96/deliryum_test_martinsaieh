def create_tracker(tracker_type, tracker_params):
    tracker_type = tracker_type.lower()
    if tracker_type == "sort":
        from .sort_wrapper import SortTracker
        return SortTracker(**tracker_params)
    elif tracker_type == "bytetrack":
        from .bytetrack_wrapper import ByteTrackTracker
        return ByteTrackTracker(**tracker_params)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
