"""
Parsing for the publication dates that news sources report.

These are deterministic Python functions — no LLM involvement.
"""

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# Search results date their articles relative to now: "3 hours ago", "1 day ago".
_RELATIVE = re.compile(
    r"\b(?P<count>\d+|an?)\s*(?P<unit>sec(?:ond)?|min(?:ute)?|hour|hr|day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)

_UNIT = {
    "sec": timedelta(seconds=1),
    "second": timedelta(seconds=1),
    "min": timedelta(minutes=1),
    "minute": timedelta(minutes=1),
    "hr": timedelta(hours=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    # Calendar units are approximated: the result only has to place an
    # article inside or outside a window of days.
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}

# "Oct 7, 2026", "7 Oct 2026" and their full-month forms.
_ABSOLUTE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y")


def parse_published_date(text: str, now: datetime | None = None) -> datetime | None:
    """
    Parse a publication date as a timezone-aware datetime.

    Understands relative dates ("3 hours ago", "yesterday"), ISO-8601,
    RFC 2822 and plain calendar dates ("Oct 7, 2026"). Returns None if the
    text is empty or in none of those forms.
    """
    text = (text or "").strip()
    if not text:
        return None
    now = now or datetime.now(timezone.utc)

    relative = _RELATIVE.search(text)
    if relative:
        count = relative.group("count")
        return now - _UNIT[relative.group("unit").lower()] * (int(count) if count.isdigit() else 1)
    if text.lower() == "yesterday":
        return now - timedelta(days=1)
    if text.lower() in ("today", "just now"):
        return now

    published = None
    try:
        published = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for date_format in _ABSOLUTE_FORMATS:
            try:
                published = datetime.strptime(text, date_format)
                break
            except ValueError:
                continue
    if published is None:
        try:
            published = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None

    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return published


def is_within_days(published: datetime | None, days: int, now: datetime | None = None) -> bool:
    """
    True if a publication date falls inside the last `days` days.

    Undated articles count as inside: there is nothing to judge them by.
    """
    if published is None:
        return True
    now = now or datetime.now(timezone.utc)
    # One day of slack for timezone differences around the cutoff.
    return now - published <= timedelta(days=days + 1)
