"""
fetch_reddit.py — Download Reddit threads via the JSON API (no credentials needed).
Saves each thread as a plain .txt file in documents/.

Usage:
    python fetch_reddit.py
"""

import json
import re
import time
import requests
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (unofficial-guide-rag/1.0; educational project)"
}

URLS = [
    "https://www.reddit.com/r/BinghamtonUniversity/comments/1rvqciw/whats_finding_off_campus_housing_like_now/",
    "https://www.reddit.com/r/BinghamtonUniversity/comments/17j9ml5/is_it_better_to_live_offcampus/",
    "https://www.reddit.com/r/BinghamtonUniversity/comments/1phlycj/off_campus_terrible_roommate/",
    "https://www.reddit.com/r/BinghamtonUniversity/comments/16f7af6/living_offcampus/",
    "https://www.reddit.com/r/BinghamtonUniversity/comments/1nc6xkk/off_campus_housing_for_2627/",
    "https://www.reddit.com/r/BinghamtonUniversity/comments/11tfz1t/how_much_cheaper_is_it_to_live_off_campus/",
    "https://www.reddit.com/r/Binghamton/comments/1rjup0u/my_off_campus_experience/",
    "https://www.reddit.com/r/college/comments/gayjbc/off_campus_or_on_campus_housing_for_first_year/",
    "https://www.reddit.com/r/college/comments/1n90nti/why_is_the_difference_between_on_campus_and_off/",
    "https://www.reddit.com/r/college/comments/11yt9kx/off_campus_housing/",
]


def slug_from_url(url: str) -> str:
    """Turn a Reddit URL into a safe filename stem."""
    parts = url.rstrip("/").split("/")
    # grab the post ID and slug
    try:
        comments_idx = parts.index("comments")
        post_id = parts[comments_idx + 1]
        slug = parts[comments_idx + 2] if len(parts) > comments_idx + 2 else post_id
        slug = re.sub(r"[^a-z0-9_]", "_", slug.lower())[:60]
        return f"{post_id}_{slug}"
    except (ValueError, IndexError):
        return re.sub(r"[^a-z0-9_]", "_", url)[-60:]


def extract_comments(children, depth=0, max_depth=8):
    """Recursively extract comment text from a Reddit JSON listing."""
    texts = []
    for child in children:
        if not isinstance(child, dict):
            continue
        kind = child.get("kind", "")
        data = child.get("data", {})
        if kind == "t1":  # comment
            body = data.get("body", "").strip()
            if body and body != "[deleted]" and body != "[removed]":
                indent = "  " * depth
                texts.append(f"{indent}- {body}")
            # recurse into replies
            replies = data.get("replies", "")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                if depth < max_depth:
                    texts.extend(extract_comments(reply_children, depth + 1, max_depth))
        elif kind == "more":
            pass  # skip "load more" stubs
    return texts


def fetch_thread(url: str) -> str | None:
    """Fetch a Reddit thread and return it as plain text."""
    json_url = url.rstrip("/") + ".json"
    try:
        resp = requests.get(json_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return None

    # data is a list: [post_listing, comments_listing]
    if not isinstance(data, list) or len(data) < 2:
        print(f"  Unexpected response shape for {url}")
        return None

    post_data = data[0]["data"]["children"][0]["data"]
    title = post_data.get("title", "").strip()
    selftext = post_data.get("selftext", "").strip()
    subreddit = post_data.get("subreddit_name_prefixed", "")

    comment_children = data[1]["data"]["children"]
    comment_lines = extract_comments(comment_children)

    lines = [
        f"Source: {url}",
        f"Subreddit: {subreddit}",
        f"Title: {title}",
        "",
    ]
    if selftext:
        lines += [selftext, ""]
    lines += ["Comments:", ""] + comment_lines

    return "\n".join(lines)


def main():
    print(f"Saving documents to: {DOCUMENTS_DIR}\n")
    success = 0
    for url in URLS:
        slug = slug_from_url(url)
        out_path = DOCUMENTS_DIR / f"{slug}.txt"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists")
            success += 1
            continue
        print(f"  Fetching {slug} ...", end=" ", flush=True)
        text = fetch_thread(url)
        if text:
            out_path.write_text(text, encoding="utf-8")
            words = len(text.split())
            print(f"OK ({words} words)")
            success += 1
        else:
            print("FAILED")
        time.sleep(1.5)  # be polite to Reddit

    print(f"\nDone: {success}/{len(URLS)} threads saved.")


if __name__ == "__main__":
    main()
