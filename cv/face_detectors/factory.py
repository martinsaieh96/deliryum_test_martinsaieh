def create_face_detector(config):
    fd_type = config['face_detector_active']
    params = config['face_detectors'][fd_type]
    if fd_type == "fr_dlib":
        from .fr_dlib import FaceRecogDetector
        return FaceRecogDetector(**params)
    elif fd_type == "opencv_haar":
        from .haar import HaarCascadeDetector
        return HaarCascadeDetector(**params)
    elif fd_type == "retinaface":
        from .retinaface import RetinaFaceDetector
        return RetinaFaceDetector(**params)
    else:
        raise ValueError(f"Face detector {fd_type} not implemented")
