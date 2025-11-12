"""FreshRSS API client implementation."""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urljoin, quote
from datetime import datetime

import aiohttp
from aiohttp import ClientResponse

from .models import (
    Article,
    AuthResponse,
    Category,
    EditResponse,
    StreamContents,
    Subscription,
    SubscriptionList,
    TagList,
    UnreadCount,
)

logger = logging.getLogger(__name__)


class FreshRSSError(Exception):
    """Base exception for FreshRSS API errors."""
    pass


class AuthenticationError(FreshRSSError):
    """Authentication failed."""
    pass


class APIError(FreshRSSError):
    """API request failed."""
    pass


class FreshRSSClient:
    """Client for interacting with FreshRSS Google Reader API."""
    
    def __init__(
        self,
        base_url: str,
        email: str,
        api_password: str,
        timeout: float = 30.0,
    ):
        """Initialize FreshRSS client."""
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/greader.php"
        self.email = email
        self.api_password = api_password
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.auth_token: Optional[str] = None
        self.edit_token: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp client session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(self, endpoint: str) -> str:
        """Build full API URL."""
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        return f"{self.api_url}/{endpoint}"
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        auth_required: bool = True,
    ) -> ClientResponse:
        """Make an API request."""
        headers = {}
        if auth_required:
            if not self.auth_token:
                raise AuthenticationError("Not authenticated. Call authenticate() first.")
            headers["Authorization"] = f"GoogleLogin auth={self.auth_token}"
        
        url = self._build_url(endpoint)
        session = await self._get_session()
        
        try:
            response = await session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=headers,
            )
            response.raise_for_status()
            return response
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error {e.status}: {e.message}")
            raise APIError(f"HTTP {e.status}: {e.message}") from e
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Request failed: {e}") from e
    
    async def authenticate(self) -> AuthResponse:
        """Authenticate with FreshRSS."""
        try:
            response = await self._request(
                method="POST",
                endpoint="accounts/ClientLogin",
                data={
                    "Email": self.email,
                    "Passwd": self.api_password,
                },
                auth_required=False,
            )
            
            text = await response.text()
            lines = text.strip().split("\n")
            auth_data = {}
            for line in lines:
                if "=" in line:
                    key, value = line.split("=", 1)
                    auth_data[key] = value
            
            if "Auth" not in auth_data:
                raise AuthenticationError("No auth token in response")
            
            auth_response = AuthResponse(
                SID=auth_data.get("SID", ""),
                Auth=auth_data["Auth"],
            )
            
            self.auth_token = auth_response.token
            logger.info("Successfully authenticated with FreshRSS")
            return auth_response
            
        except APIError as e:
            raise AuthenticationError(f"Authentication failed: {e}") from e
    
    async def get_token(self) -> str:
        """Get edit token for write operations."""
        response = await self._request("GET", "reader/api/0/token")
        self.edit_token = (await response.text()).strip()
        assert self.edit_token is not None
        return self.edit_token
    
    async def get_subscription_list(self) -> SubscriptionList:
        """Get list of subscribed feeds."""
        response = await self._request("GET", "reader/api/0/subscription/list", params={"output": "json"})
        data = await response.json()
        return SubscriptionList(subscriptions=[
            Subscription(**sub) for sub in data.get("subscriptions", [])
        ])
    
    async def get_tag_list(self) -> TagList:
        """Get list of tags/labels/folders."""
        response = await self._request("GET", "reader/api/0/tag/list", params={"output": "json"})
        data = await response.json()
        tags = []
        for tag in data.get("tags", []):
            tags.append(Category(
                id=tag["id"],
                label=tag.get("sortid", tag["id"].split("/")[-1]),
                type="tag" if "label" in tag["id"] else "state"
            ))
        return TagList(tags=tags)
    
    async def get_unread_counts(self) -> List[UnreadCount]:
        """Get unread counts for all feeds and categories."""
        response = await self._request("GET", "reader/api/0/unread-count", params={"output": "json"})
        data = await response.json()
        return [
            UnreadCount(**item) for item in data.get("unreadcounts", [])
        ]
    
    async def get_stream_contents(
        self,
        stream_id: str = "user/-/state/com.google/reading-list",
        count: int = 50,
        order: str = "d",
        start_time: Optional[int] = None,
        continuation: Optional[str] = None,
        exclude_target: Optional[str] = None,
        include_target: Optional[str] = None,
    ) -> StreamContents:
        """Get articles from a stream."""
        params = {
            "output": "json",
            "n": count,
            "r": order,
        }
        
        if start_time:
            params["ot"] = start_time
        if continuation:
            params["c"] = continuation
        if exclude_target:
            params["xt"] = exclude_target
        if include_target:
            params["it"] = include_target
        
        encoded_stream = quote(stream_id, safe="")
        endpoint = f"reader/api/0/stream/contents/{encoded_stream}"
        
        response = await self._request("GET", endpoint, params=params)
        data = await response.json()
        
        articles = []
        for item in data.get("items", []):
            published = None
            if "published" in item:
                published = datetime.fromtimestamp(item["published"])
            elif "crawlTimeMsec" in item:
                published = datetime.fromtimestamp(int(item["crawlTimeMsec"]) / 1000)
            
            article = Article(
                id=item["id"],
                title=item.get("title", ""),
                published=published or datetime.now(),
                updated=datetime.fromtimestamp(item["updated"]) if "updated" in item else None,
                author=item.get("author"),
                content=item.get("content", {}).get("content") if "content" in item else None,
                summary=item.get("summary", {}).get("content") if "summary" in item else None,
                categories=item.get("categories", []),
                origin=item.get("origin"),
                alternate=item.get("alternate", []),
                crawlTimeMsec=item.get("crawlTimeMsec"),
                timestampUsec=item.get("timestampUsec"),
            )
            articles.append(article)
        
        return StreamContents(
            id=data.get("id", stream_id),
            title=data.get("title"),
            items=articles,
            continuation=data.get("continuation"),
            updated=data.get("updated"),
        )
    
    async def _edit_tag(
        self,
        item_ids: List[str],
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
    ) -> EditResponse:
        """Helper to add/remove tags from articles."""
        if not self.edit_token:
            await self.get_token()

        data: List[Tuple[str, Optional[str]]] = [("T", self.edit_token)]

        if add_tags:
            for tag in add_tags:
                data.append(("a", tag))
        if remove_tags:
            for tag in remove_tags:
                data.append(("r", tag))

        for item_id in item_ids:
            data.append(("i", item_id))

        response = await self._request("POST", "reader/api/0/edit-tag", data=data)
        return EditResponse(status=(await response.text()).strip())

    async def mark_as_read(self, item_ids: List[str]) -> EditResponse:
        """Mark articles as read."""
        return await self._edit_tag(
            item_ids, add_tags=["user/-/state/com.google/read"]
        )

    async def mark_as_unread(self, item_ids: List[str]) -> EditResponse:
        """Mark articles as unread."""
        return await self._edit_tag(
            item_ids,
            add_tags=["user/-/state/com.google/kept-unread"],
            remove_tags=["user/-/state/com.google/read"],
        )

    async def star_article(self, item_ids: List[str]) -> EditResponse:
        """Star articles."""
        return await self._edit_tag(
            item_ids, add_tags=["user/-/state/com.google/starred"]
        )

    async def unstar_article(self, item_ids: List[str]) -> EditResponse:
        """Unstar articles."""
        return await self._edit_tag(
            item_ids, remove_tags=["user/-/state/com.google/starred"]
        )

    async def add_label(self, item_ids: List[str], label: str) -> EditResponse:
        """Add label to articles."""
        return await self._edit_tag(item_ids, add_tags=[f"user/-/label/{label}"])
    
    async def subscribe(
        self, feed_url: str, title: Optional[str] = None, folder: Optional[str] = None
    ) -> EditResponse:
        """Subscribe to a new feed."""
        if not self.edit_token:
            await self.get_token()

        data: Dict[str, Any] = {
            "T": self.edit_token,
            "ac": "subscribe",
            "s": f"feed/{feed_url}",
        }
        if title:
            data["t"] = title
        if folder:
            data["a"] = f"user/-/label/{folder}"

        response = await self._request("POST", "reader/api/0/subscription/edit", data=data)
        return EditResponse(status=(await response.text()).strip())

    async def unsubscribe(self, feed_url: str) -> EditResponse:
        """Unsubscribe from a feed."""
        if not self.edit_token:
            await self.get_token()

        data = {
            "T": self.edit_token,
            "ac": "unsubscribe",
            "s": f"feed/{feed_url}",
        }
        response = await self._request("POST", "reader/api/0/subscription/edit", data=data)
        return EditResponse(status=(await response.text()).strip())