class PropDetector:
    def __init__(self):
        # Stub for prop detection model
        pass

    def detect_props(self, crop_img):
        """
        Takes a BGR image crop of a person and detects props like covered_face and large_bag.
        Returns a list of prop strings.
        """
        props = []
        # TODO: Implement actual model inference. 
        # For MVP integration testing, we can return empty or mock values.
        # This is meant to be a placeholder for the logic required by Person B's side.
        
        # Example logic structure:
        # if detect_large_bag(crop_img):
        #     props.append("large_bag")
        # if detect_covered_face(crop_img):
        #     props.append("covered_face")
        
        return props
