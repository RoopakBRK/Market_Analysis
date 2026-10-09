import pytest

from config.settings import settings


@pytest.fixture(autouse=True)
def firecrawl_offline(request, monkeypatch):
    """
    Blank the Firecrawl key for every test not marked `live`, so an offline
    test cannot make a paid call with the real key in .env.
    """
    if "live" not in request.keywords:
        monkeypatch.setattr(settings, "FIRECRAWL_API_KEY", "")
