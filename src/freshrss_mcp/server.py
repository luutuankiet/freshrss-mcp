"""FreshRSS MCP Server implementation."""

import asyncio
import logging
import os
import random
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .client import FreshRSSClient, FreshRSSError, AuthenticationError

from mcp.server.transport_security import TransportSecuritySettings


# Load environment variables
load_dotenv()

# --- Configurable short ID length ---
# Article IDs look like: tag:google.com,2005:reader/item/00064efa1097fab5
# The hex suffix is the unique part. We expose shortened IDs to save tokens.
# Set to 0 to disable shortening (return full IDs).
SHORT_ID_LENGTH = 16

ARTICLE_ID_PREFIX = "tag:google.com,2005:reader/item/"

def shorten_id(full_id: str) -> str:
    """Shorten a Google Reader article ID to save tokens."""
    if SHORT_ID_LENGTH <= 0:
        return full_id
    if full_id.startswith(ARTICLE_ID_PREFIX):
        return full_id[len(ARTICLE_ID_PREFIX):][:SHORT_ID_LENGTH]
    return full_id

def expand_id(short_id: str) -> str:
    """Expand a shortened article ID back to full form."""
    if short_id.startswith(ARTICLE_ID_PREFIX):
        return short_id  # Already full
    return f"{ARTICLE_ID_PREFIX}{short_id}"

def expand_ids(ids: List[str]) -> List[str]:
    """Expand a list of potentially shortened IDs."""
    return [expand_id(i) for i in ids]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security_settings = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)
# Initialize MCP server
mcp = FastMCP("FreshRSS MCP Server", transport_security=security_settings)


class AuthenticateParams(BaseModel):
    """Parameters for authentication."""
    base_url: Optional[str] = Field(None, description="FreshRSS instance URL (uses env if not provided)")
    email: Optional[str] = Field(None, description="User email (uses env if not provided)")
    api_password: Optional[str] = Field(None, description="API password (uses env if not provided)")


class GetArticlesParams(BaseModel):
    """Parameters for fetching articles."""
    folder: Optional[str] = Field(None, description="Folder/label name to filter by")
    feed_url: Optional[str] = Field(None, description="Feed URL to filter by")
    show_read: bool = Field(False, description="Include read articles")
    starred_only: bool = Field(False, description="Show only starred articles")
    count: int = Field(50, description="Number of articles to fetch (max ~1000)")
    order: str = Field("newest", description="Sort order: 'newest' or 'oldest'")
    continuation: Optional[str] = Field(None, description="Continuation token for pagination")
    trim_content: bool = Field(True, description="trim default article body and content to first 300 characters to not overwhelm models context.")


class GetHeadlinesParams(BaseModel):
    """Parameters for fetching article headlines (token-efficient)."""
    folder: Optional[str] = Field(None, description="Folder/label name to filter by")
    feed_url: Optional[str] = Field(None, description="Feed URL to filter by")
    show_read: bool = Field(False, description="Include read articles")
    starred_only: bool = Field(False, description="Show only starred articles")
    count: int = Field(100, description="Number of headlines to fetch (max ~1000)")
    order: str = Field("newest", description="Sort order: 'newest' or 'oldest'")
    continuation: Optional[str] = Field(None, description="Continuation token for pagination")


class GetArticleDetailParams(BaseModel):
    """Parameters for fetching full article content by ID."""
    article_ids: List[str] = Field(..., description="List of article IDs to fetch full content for (max 10)")


class GetDiverseDigestParams(BaseModel):
    """Parameters for getting a category-balanced article digest."""
    per_category: int = Field(5, description="Number of articles per category (default 5)")
    categories: Optional[List[str]] = Field(None, description="Categories to include (None = all)")
    include_uncategorized: bool = Field(True, description="Include uncategorized feeds")
    show_read: bool = Field(False, description="Include read articles")


class MarkArticlesParams(BaseModel):
    """Parameters for marking articles."""
    article_ids: List[str] = Field(..., description="List of article IDs")


class AddLabelParams(BaseModel):
    """Parameters for adding labels."""
    article_ids: List[str] = Field(..., description="List of article IDs")
    label: str = Field(..., description="Label name to add")


class SubscribeParams(BaseModel):
    """Parameters for subscribing to a feed."""
    feed_url: str = Field(..., description="URL of the feed to subscribe to")
    title: Optional[str] = Field(None, description="Custom title for the feed")
    folder: Optional[str] = Field(None, description="Folder to add the feed to")


class UnsubscribeParams(BaseModel):
    """Parameters for unsubscribing from a feed."""
    feed_url: str = Field(..., description="URL of the feed to unsubscribe from")


class MarkStreamReadParams(BaseModel):
    """Parameters for marking all articles in a stream as read."""
    stream: str = Field(
        "all",
        description="What to mark as read: 'all' for everything, or a folder name like 'tech', 'ML', 'security'"
    )
    older_than_hours: Optional[float] = Field(
        None,
        description="Only mark articles older than this many hours. None = mark everything."
    )


class GetDigestCompactParams(BaseModel):
    """Parameters for getting an ultra-compact category-balanced digest."""
    per_category: int = Field(5, description="Number of articles per category (default 5)")
    categories: Optional[List[str]] = Field(None, description="Categories to include (None = all)")
    include_uncategorized: bool = Field(True, description="Include uncategorized feeds")
    show_read: bool = Field(False, description="Include read articles")


async def ensure_authenticated() -> FreshRSSClient:
    """Create and authenticate a new client instance."""
    base_url = os.getenv("FRESHRSS_URL")
    email = os.getenv("FRESHRSS_EMAIL")
    api_password = os.getenv("FRESHRSS_API_PASSWORD")

    if not all([base_url, email, api_password]):
        raise AuthenticationError(
            "Authentication details not found in environment variables. "
            "Please set FRESHRSS_URL, FRESHRSS_EMAIL, and FRESHRSS_API_PASSWORD."
        )

    assert base_url is not None
    assert email is not None
    assert api_password is not None
    client = FreshRSSClient(base_url, email, api_password)
    await client.authenticate()
    return client


@mcp.tool()
async def freshrss_authenticate(params: AuthenticateParams) -> Dict[str, Any]:
    """Authenticate with FreshRSS instance and save credentials to environment for subsequent calls.
    
    Uses provided parameters or falls back to environment variables:
    - FRESHRSS_URL
    - FRESHRSS_EMAIL
    - FRESHRSS_API_PASSWORD
    """
    base_url = params.base_url or os.getenv("FRESHRSS_URL")
    email = params.email or os.getenv("FRESHRSS_EMAIL")
    api_password = params.api_password or os.getenv("FRESHRSS_API_PASSWORD")
    
    if not all([base_url, email, api_password]):
        return {
            "success": False,
            "error": "Missing required credentials. Please provide base_url, email, and api_password."
        }
    
    try:
        # Store credentials in environment for this session
        assert base_url is not None
        assert email is not None
        assert api_password is not None
        os.environ["FRESHRSS_URL"] = base_url
        os.environ["FRESHRSS_EMAIL"] = email
        os.environ["FRESHRSS_API_PASSWORD"] = api_password
        
        # Test authentication
        async with FreshRSSClient(base_url, email, api_password) as client:
            await client.authenticate()

        return {
            "success": True,
            "message": "Successfully authenticated with FreshRSS",
            "user": email,
            "instance": base_url
        }
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_get_token() -> Dict[str, Any]:
    """Get edit token for write operations. Usually called automatically when needed."""
    try:
        async with await ensure_authenticated() as client:
            token = await client.get_token()
            return {
                "success": True,
            "token": token,
            "message": "Token retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Failed to get token: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_list_folders() -> Dict[str, Any]:
    """List all folders/categories/tags in FreshRSS."""
    try:
        async with await ensure_authenticated() as client:
            tag_list = await client.get_tag_list()
            
            folders = []
            for tag in tag_list.folders:
                folders.append({
                    "name": tag.label,
                "id": tag.id,
                "type": "folder"
            })
        
        return {
            "success": True,
            "folders": folders,
            "count": len(folders)
        }
    except Exception as e:
        logger.error(f"Failed to list folders: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_list_subscriptions() -> Dict[str, Any]:
    """List all subscribed feeds with their folders."""
    try:
        async with await ensure_authenticated() as client:
            subscription_list = await client.get_subscription_list()
            
            subscriptions = []
            for sub in subscription_list.subscriptions:
                folders = [cat.label for cat in sub.categories if cat.is_label]
                subscriptions.append({
                    "title": sub.title,
                "feed_url": sub.feed_id,
                "id": sub.id,
                "folders": folders,
                "url": sub.url,
                "html_url": sub.htmlUrl,
                "icon_url": sub.iconUrl
            })
        
        return {
            "success": True,
            "subscriptions": subscriptions,
            "count": len(subscriptions)
        }
    except Exception as e:
        logger.error(f"Failed to list subscriptions: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_get_unread_count() -> Dict[str, Any]:
    """Get unread counts by feed and folder."""
    try:
        async with await ensure_authenticated() as client:
            unread_counts = await client.get_unread_counts()
            
            # Organize counts by type
            feeds = []
            folders = []
            total_unread = 0
            
            for count in unread_counts:
                if count.id.startswith("feed/"):
                    feeds.append({
                        "feed_url": count.id[5:],
                    "count": count.count
                })
                elif count.id.startswith("user/-/label/"):
                    folders.append({
                        "folder": count.id.split("/")[-1],
                        "count": count.count
                    })
                elif count.id == "user/-/state/com.google/reading-list":
                    total_unread = count.count
        
        return {
            "success": True,
            "total_unread": total_unread,
            "feeds": feeds,
            "folders": folders
        }
    except Exception as e:
        logger.error(f"Failed to get unread counts: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_get_articles(params: GetArticlesParams) -> Dict[str, Any]:
    """Fetch articles with various filters."""
    try:
        async with await ensure_authenticated() as client:
            # Determine stream ID
            if params.starred_only:
                stream_id = "user/-/state/com.google/starred"
            elif params.folder:
                stream_id = f"user/-/label/{params.folder}"
            elif params.feed_url:
                stream_id = f"feed/{params.feed_url}"
            else:
                stream_id = "user/-/state/com.google/reading-list"
            
            # Set exclude target for unread only
            exclude_target = None if params.show_read else "user/-/state/com.google/read"
            
            # Fetch articles
            stream = await client.get_stream_contents(
                stream_id=stream_id,
                count=params.count,
                order="d" if params.order == "newest" else "o",
                exclude_target=exclude_target,
                continuation=params.continuation
            )


            
            # Format articles
            articles = []
            for article in stream.items:
                content = article.content or ""
                summary = article.summary or ""
                articles.append({
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "content": content[:300] if params.trim_content else content,
                "summary": summary[:300] if params.trim_content else summary,
                "author": article.author,
                "published": article.published.isoformat() if article.published else None,
                "feed_title": article.feed_title,
                "feed_url": article.feed_url,
                "is_read": article.is_read,
                "is_starred": article.is_starred,
                "labels": [cat for cat in article.categories if cat.startswith("user/-/label/")]
            })
        
        return {
            "success": True,
            "articles": articles,
            "count": len(articles),
            "has_more": stream.has_more,
            "continuation": stream.continuation
        }
    except Exception as e:
        logger.error(f"Failed to get articles: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_mark_read(params: MarkArticlesParams) -> Dict[str, Any]:
    """Mark articles as read."""
    try:
        async with await ensure_authenticated() as client:
            expanded = expand_ids(params.article_ids)
            response = await client.mark_as_read(expanded)
            
            return {
                "success": True,
            "message": f"Marked {len(params.article_ids)} article(s) as read",
            "status": response.status
        }
    except Exception as e:
        logger.error(f"Failed to mark articles as read: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_mark_unread(params: MarkArticlesParams) -> Dict[str, Any]:
    """Mark articles as unread."""
    try:
        async with await ensure_authenticated() as client:
            expanded = expand_ids(params.article_ids)
            response = await client.mark_as_unread(expanded)
            
            return {
                "success": True,
            "message": f"Marked {len(params.article_ids)} article(s) as unread",
            "status": response.status
        }
    except Exception as e:
        logger.error(f"Failed to mark articles as unread: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_mark_stream_read(params: MarkStreamReadParams) -> Dict[str, Any]:
    """Mark all articles in a stream/folder as read in one call.

    MUCH more efficient than marking individual articles.
    Use 'all' to mark everything, or a folder name like 'tech', 'ML', 'security'.

    Common pattern: mark_stream_read(all) then mark_unread([5 keeper IDs]).
    """
    try:
        async with await ensure_authenticated() as client:
            # Resolve stream ID
            if params.stream == "all":
                stream_id = "user/-/state/com.google/reading-list"
            else:
                stream_id = f"user/-/label/{params.stream}"

            # Calculate timestamp if hours specified
            older_than_usec = None
            if params.older_than_hours is not None:
                import time
                cutoff = time.time() - (params.older_than_hours * 3600)
                older_than_usec = int(cutoff * 1_000_000)

            response = await client.mark_all_as_read(stream_id, older_than_usec)

            return {
                "success": True,
                "message": f"Marked all articles in '{params.stream}' as read",
                "status": response.status,
            }
    except Exception as e:
        logger.error(f"Failed to mark stream as read: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def freshrss_star_article(params: MarkArticlesParams) -> Dict[str, Any]:
    """Star articles."""
    try:
        async with await ensure_authenticated() as client:
            expanded = expand_ids(params.article_ids)
            response = await client.star_article(expanded)
            
            return {
                "success": True,
            "message": f"Starred {len(params.article_ids)} article(s)",
            "status": response.status
        }
    except Exception as e:
        logger.error(f"Failed to star articles: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_unstar_article(params: MarkArticlesParams) -> Dict[str, Any]:
    """Unstar articles."""
    try:
        async with await ensure_authenticated() as client:
            expanded = expand_ids(params.article_ids)
            response = await client.unstar_article(expanded)
            
            return {
                "success": True,
            "message": f"Unstarred {len(params.article_ids)} article(s)",
            "status": response.status
        }
    except Exception as e:
        logger.error(f"Failed to unstar articles: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_add_label(params: AddLabelParams) -> Dict[str, Any]:
    """Add label to articles."""
    try:
        async with await ensure_authenticated() as client:
            response = await client.add_label(params.article_ids, params.label)
            
            return {
                "success": True,
            "message": f"Added label '{params.label}' to {len(params.article_ids)} article(s)",
            "status": response.status
        }
    except Exception as e:
        logger.error(f"Failed to add label: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_subscribe(params: SubscribeParams) -> Dict[str, Any]:
    """Subscribe to a new feed."""
    try:
        async with await ensure_authenticated() as client:
            response = await client.subscribe(
                feed_url=params.feed_url,
                title=params.title,
            folder=params.folder
        )
        
        return {
            "success": True,
            "message": f"Successfully subscribed to {params.feed_url}",
            "status": response.status
        }
    except Exception as e:
        logger.error(f"Failed to subscribe: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def freshrss_unsubscribe(params: UnsubscribeParams) -> Dict[str, Any]:
    """Unsubscribe from a feed."""
    try:
        async with await ensure_authenticated() as client:
            response = await client.unsubscribe(params.feed_url)

            return {
                "success": True,
                "message": f"Successfully unsubscribed from {params.feed_url}",
                "status": response.status,
            }
    except Exception as e:
        logger.error(f"Failed to unsubscribe: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def freshrss_get_headlines(params: GetHeadlinesParams) -> Dict[str, Any]:
    """Get article headlines only — ultra token-efficient.

    Returns only: id, title, feed_title, folder, published, url.
    ~10x fewer tokens than get_articles. Use this for scanning/triage,
    then call get_article_detail for articles you want to read in full.
    """
    try:
        async with await ensure_authenticated() as client:
            # Determine stream ID
            if params.starred_only:
                stream_id = "user/-/state/com.google/starred"
            elif params.folder:
                stream_id = f"user/-/label/{params.folder}"
            elif params.feed_url:
                stream_id = f"feed/{params.feed_url}"
            else:
                stream_id = "user/-/state/com.google/reading-list"

            exclude_target = None if params.show_read else "user/-/state/com.google/read"

            stream = await client.get_stream_contents(
                stream_id=stream_id,
                count=params.count,
                order="d" if params.order == "newest" else "o",
                exclude_target=exclude_target,
                continuation=params.continuation,
            )

            headlines = []
            for article in stream.items:
                # Extract folder from categories
                folders = [
                    cat.split("/")[-1]
                    for cat in article.categories
                    if cat.startswith("user/-/label/")
                ]
                headline = {
                    "id": shorten_id(article.id),
                    "title": article.title,
                    "feed": article.feed_title or "",
                    "folder": folders[0] if folders else "",
                }
                if article.published:
                    headline["published"] = article.published.strftime("%Y-%m-%d %H:%M")
                if article.url:
                    headline["url"] = article.url
                headlines.append(headline)

            return {
                "success": True,
                "headlines": headlines,
                "count": len(headlines),
                "has_more": stream.has_more,
                "continuation": stream.continuation,
            }
    except Exception as e:
        logger.error(f"Failed to get headlines: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def freshrss_get_article_detail(params: GetArticleDetailParams) -> Dict[str, Any]:
    """Get full content for specific articles by ID.

    Use after scanning headlines to deep-dive into interesting articles.
    Max 10 articles per call to control token usage.
    """
    if len(params.article_ids) > 10:
        return {
            "success": False,
            "error": "Max 10 articles per call. Use multiple calls for more.",
        }

    try:
        async with await ensure_authenticated() as client:
            # Fetch all articles at once using item IDs via stream/items/contents
            expanded_ids = expand_ids(params.article_ids)
            items_data = await client.get_items_by_ids(expanded_ids)

            articles = []
            for article in items_data:
                folders = [
                    cat.split("/")[-1]
                    for cat in article.categories
                    if cat.startswith("user/-/label/")
                ]
                articles.append({
                    "id": article.id,
                    "title": article.title,
                    "url": article.url or "",
                    "content": article.content or "",
                    "summary": article.summary or "",
                    "author": article.author or "",
                    "feed": article.feed_title or "",
                    "folder": folders[0] if folders else "",
                    "published": article.published.isoformat() if article.published else "",
                    "is_starred": article.is_starred,
                })

            return {
                "success": True,
                "articles": articles,
                "count": len(articles),
            }
    except Exception as e:
        logger.error(f"Failed to get article details: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def freshrss_get_diverse_digest(params: GetDiverseDigestParams) -> Dict[str, Any]:
    """Get a category-balanced digest of recent unread articles.

    Fetches N articles from EACH category to ensure topic diversity.
    Returns headlines only (token-efficient). Use get_article_detail
    to deep-dive into selected articles.

    This is the primary tool for daily curation — ensures you see
    security, business, world news, etc., not just AI/ML.
    """
    try:
        async with await ensure_authenticated() as client:
            # Get all folders/categories
            tag_list = await client.get_tag_list()
            target_folders = []

            if params.categories:
                target_folders = [f for f in tag_list.folders if f.label in params.categories]
            else:
                target_folders = tag_list.folders

            exclude_target = None if params.show_read else "user/-/state/com.google/read"

            digest: Dict[str, list] = {}
            total = 0

            for folder in target_folders:
                stream_id = folder.id
                stream = await client.get_stream_contents(
                    stream_id=stream_id,
                    count=params.per_category,
                    order="d",
                    exclude_target=exclude_target,
                )

                folder_headlines = []
                for article in stream.items:
                    headline = {
                        "id": shorten_id(article.id),
                        "title": article.title,
                    }
                    if article.feed_title:
                        headline["feed"] = article.feed_title
                    if article.published:
                        headline["published"] = article.published.strftime("%Y-%m-%d %H:%M")
                    folder_headlines.append(headline)

                if folder_headlines:
                    digest[folder.label] = folder_headlines
                    total += len(folder_headlines)

            # Also fetch uncategorized if requested
            if params.include_uncategorized:
                # Fetch from reading-list and exclude articles already in folders
                all_folder_ids = {f.id for f in target_folders}
                stream = await client.get_stream_contents(
                    stream_id="user/-/state/com.google/reading-list",
                    count=params.per_category * 2,  # fetch extra, filter down
                    order="d",
                    exclude_target=exclude_target,
                )
                uncategorized = []
                for article in stream.items:
                    article_folders = [
                        cat for cat in article.categories
                        if cat.startswith("user/-/label/")
                    ]
                    if not article_folders:
                        headline = {
                            "id": shorten_id(article.id),
                            "title": article.title,
                        }
                        if article.feed_title:
                            headline["feed"] = article.feed_title
                        if article.published:
                            headline["published"] = article.published.strftime("%Y-%m-%d %H:%M")
                        uncategorized.append(headline)
                    if len(uncategorized) >= params.per_category:
                        break

                if uncategorized:
                    digest["Uncategorized"] = uncategorized
                    total += len(uncategorized)

            return {
                "success": True,
                "digest": digest,
                "categories_found": list(digest.keys()),
                "total_articles": total,
            }
    except Exception as e:
        logger.error(f"Failed to get diverse digest: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def freshrss_get_digest_compact(params: GetDigestCompactParams) -> Dict[str, Any]:
    """Get an ultra-compact category-balanced digest as plain text.

    Returns ~10 tokens per article vs ~100 for JSON format.
    Each line: [short_id] Title (Feed)
    Grouped by category with counts.

    Designed for agent curation workflows where token efficiency matters.
    After selecting articles, use mark_stream_read + mark_unread to keep only your picks.
    """
    try:
        async with await ensure_authenticated() as client:
            tag_list = await client.get_tag_list()
            target_folders = []

            if params.categories:
                target_folders = [f for f in tag_list.folders if f.label in params.categories]
            else:
                target_folders = tag_list.folders

            exclude_target = None if params.show_read else "user/-/state/com.google/read"

            lines = []
            total = 0
            categories_found = []

            for folder in target_folders:
                stream = await client.get_stream_contents(
                    stream_id=folder.id,
                    count=params.per_category,
                    order="d",
                    exclude_target=exclude_target,
                )

                if not stream.items:
                    continue

                categories_found.append(folder.label)
                lines.append(f"\n## {folder.label} ({len(stream.items)})")
                for article in stream.items:
                    sid = shorten_id(article.id)
                    feed = article.feed_title or ""
                    # Truncate feed name to save tokens
                    if len(feed) > 25:
                        feed = feed[:22] + "..."
                    lines.append(f"[{sid}] {article.title} ({feed})")
                    total += 1

            # Uncategorized
            if params.include_uncategorized:
                stream = await client.get_stream_contents(
                    stream_id="user/-/state/com.google/reading-list",
                    count=params.per_category * 2,
                    order="d",
                    exclude_target=exclude_target,
                )
                uncategorized = []
                for article in stream.items:
                    article_folders = [
                        cat for cat in article.categories
                        if cat.startswith("user/-/label/")
                    ]
                    if not article_folders:
                        sid = shorten_id(article.id)
                        feed = article.feed_title or ""
                        if len(feed) > 25:
                            feed = feed[:22] + "..."
                        uncategorized.append(f"[{sid}] {article.title} ({feed})")
                    if len(uncategorized) >= params.per_category:
                        break

                if uncategorized:
                    categories_found.append("Uncategorized")
                    lines.append(f"\n## Uncategorized ({len(uncategorized)})")
                    lines.extend(uncategorized)
                    total += len(uncategorized)

            digest_text = "\n".join(lines)

            return {
                "success": True,
                "digest": digest_text,
                "categories": categories_found,
                "total": total,
            }
    except Exception as e:
        logger.error(f"Failed to get compact digest: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Run the MCP server."""
    import sys
    
    # Check for transport argument
    transport = "stdio"  # default
    port = 8000  # default port for HTTP
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--http":
            transport = "streamable-http"
            print(f"🚀 Starting FreshRSS MCP Server on http://localhost:{port}")
            print("📡 Available endpoints:")
            print(f"   • WebSocket: ws://localhost:{port}/ws")
            print(f"   • Health: http://localhost:{port}/health")
            print("📋 MCP Tools: 13 tools available for FreshRSS management")
            print("🔧 Configure in Claude Desktop with WebSocket URL")
        elif sys.argv[1] == "--sse":
            transport = "sse"
            print(f"🚀 Starting FreshRSS MCP Server with SSE on port {port}")
        elif sys.argv[1] == "--stdio":
            transport = "stdio"
            print("🚀 Starting FreshRSS MCP Server with stdio transport")
        elif sys.argv[1] in ["-h", "--help"]:
            print("FreshRSS MCP Server")
            print("Usage:")
            print("  freshrss-mcp [--http|--sse|--stdio]")
            print("  freshrss-mcp --http    # HTTP transport on port 8000")
            print("  freshrss-mcp --sse     # Server-Sent Events transport")
            print("  freshrss-mcp --stdio   # Standard I/O transport (default)")
            return
    
    # Configure logging level
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    
    if transport == "streamable-http":
        # For HTTP transport, we need to set up the server differently
        import uvicorn
        
        # Create the HTTP app
        app = mcp.streamable_http_app()
        
        # Show startup logs
        logger.info(f"🚀 FreshRSS MCP Server starting on http://localhost:{port}")
        logger.info("📋 13 MCP tools loaded for FreshRSS management")
        
        # Use 0.0.0.0 when running in Docker to allow external connections
        host = "0.0.0.0" if os.getenv("DOCKER_CONTAINER", "").lower() == "true" else "localhost"
        
        uvicorn.run(
            app, 
            host=host, 
            port=port, 
            log_level=log_level.lower(),
            access_log=True
        )
    elif transport == "stdio":
        if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
            logger.info("🚀 FreshRSS MCP Server starting with stdio transport")
        mcp.run(transport=transport)
    else:
        logger.info(f"🚀 FreshRSS MCP Server starting with {transport} transport")
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()