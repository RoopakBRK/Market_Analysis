"""
Reddit API client.

Uses PRAW (Python Reddit API Wrapper) for OAuth2 authentication.
Credentials are read from settings — never hardcoded, never logged.

Graceful failure policy:
- Missing credentials → return []
- OAuth failure → return []
- Rate limit → return []
- Network error → return []
- Malformed post → skip, continue
"""

import sys
from typing import Any, Optional

from config.settings import settings
from src.tools.common.normalization import trim_content, utc_now_iso

_MAX_BODY_CHARS = 400
_DEFAULT_LIMIT = 10


def _is_configured() -> bool:
    return bool(
        settings.REDDIT_CLIENT_ID
        and settings.REDDIT_CLIENT_SECRET
    )


def _build_reddit() -> Optional[Any]:
    """
    Return an authenticated PRAW Reddit instance, or None if unavailable.
    """
    if not _is_configured():
        print("[Reddit] Skipped — REDDIT_CLIENT_ID/SECRET not configured.", file=sys.stderr)
        return None

    try:
        import praw  # type: ignore
        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT,
            # Read-only mode — no username/password required for public posts.
        )
        # Quick validation
        _ = reddit.read_only
        return reddit
    except ImportError:
        print("[Reddit] praw not installed. Install with: pip install praw", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[Reddit] Client init failed: {type(exc).__name__}", file=sys.stderr)
        return None


def _normalize_post(submission: Any) -> Optional[dict[str, Any]]:
    """
    Normalise a PRAW Submission into the project's internal post dict.
    Returns None if the post is unusable (deleted, no title, etc.).
    """
    try:
        title = getattr(submission, "title", "") or ""
        if not title or title == "[deleted]":
            return None

        body = getattr(submission, "selftext", "") or ""
        if body == "[deleted]" or body == "[removed]":
            body = ""

        url = f"https://www.reddit.com{getattr(submission, 'permalink', '')}"
        score = int(getattr(submission, "score", 0) or 0)
        created_utc = getattr(submission, "created_utc", 0)
        subreddit = str(getattr(submission, "subreddit", "")) or ""

        # Convert UNIX timestamp to ISO-8601
        published_at = ""
        if created_utc:
            from datetime import datetime, timezone
            published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

        return {
            "title": title.strip(),
            "subreddit": subreddit,
            "url": url,
            "score": score,
            "published_at": published_at,
            "body": trim_content(body, max_chars=_MAX_BODY_CHARS),
            "source_type": "reddit",
        }
    except Exception:
        return None


def search_subreddit_posts(
    subreddits: list[str],
    query: str,
    limit: int = _DEFAULT_LIMIT,
    time_filter: str = "week",
) -> list[dict[str, Any]]:
    """
    Search specific subreddits for posts matching a query.

    Returns a list of normalised post dicts.
    Returns [] gracefully on any failure.
    """
    reddit = _build_reddit()
    if reddit is None:
        return []

    results: list[dict[str, Any]] = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.search(query, limit=limit, time_filter=time_filter):
                post = _normalize_post(submission)
                if post:
                    results.append(post)
        except Exception as exc:
            print(f"[Reddit] Search failed in r/{sub_name}: {type(exc).__name__}", file=sys.stderr)
            continue

    print(f"[Reddit] Query '{query[:60]}' → {len(results)} posts", file=sys.stderr)
    return results
