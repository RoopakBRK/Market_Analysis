class ChangeDetectionService:

    def detect_change(
        self,
        previous_sentiment,
        current_sentiment,
    ):

        if previous_sentiment is None:
            return None

        if previous_sentiment == current_sentiment:
            return None

        return {
            "previous": previous_sentiment,
            "current": current_sentiment,
            "changed": True,
        }