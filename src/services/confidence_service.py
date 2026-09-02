class ConfidenceService:
    """
    Calculates an explainable confidence score.
    """

    def calculate(
        self,
        source_agreement: float,
        historical_similarity: float,
        data_freshness: float,
        official_filing: float,
        model_certainty: float,
    ) -> int:

        confidence = (
            source_agreement * 0.35
            + historical_similarity * 0.20
            + data_freshness * 0.15
            + official_filing * 0.20
            + model_certainty * 0.10
        )

        return round(confidence)