"""Google Places API client for finding pubs and restaurants."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from trail_pal.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PlaceResult:
    """Result from Google Places API."""

    place_id: str
    name: str
    latitude: float
    longitude: float
    rating: float
    user_ratings_total: Optional[int]
    types: list[str]
    metadata: dict


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made within rate limits."""
        async with self._lock:
            now = time.time()
            time_since_last = now - self._last_request_time
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self._last_request_time = time.time()


class GooglePlacesClient:
    """Client for the Google Places API."""

    BASE_URL = "https://places.googleapis.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
    ):
        """Initialize the Google Places client.

        Args:
            api_key: Google Places API key.
            requests_per_minute: Rate limit for requests.
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_places_api_key
        self.rate_limiter = RateLimiter(
            requests_per_minute or settings.google_places_requests_per_minute
        )
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key:
            logger.warning(
                "No Google Places API key configured. Set GOOGLE_PLACES_API_KEY environment variable."
            )

    async def __aenter__(self):
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()

    async def search_nearby_pubs(
        self,
        lat: float,
        lon: float,
        radius_m: int = 500,
        min_rating: float = 4.2,
    ) -> list[PlaceResult]:
        """Search for pubs near a location.

        Args:
            lat: Latitude.
            lon: Longitude.
            radius_m: Search radius in meters (max 5000).
            min_rating: Minimum rating (default 4.2).

        Returns:
            List of PlaceResult objects for pubs meeting the criteria.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        if not self.api_key:
            logger.error("Cannot make Google Places request without API key")
            return []

        await self.rate_limiter.acquire()

        # Use the new Places API (New) endpoint
        url = f"{self.BASE_URL}/places:searchNearby"

        # Convert radius to meters (ensure it's within API limits)
        radius_m = min(radius_m, 5000)  # Google Places API max is 5000m

        payload = {
            "includedTypes": ["bar", "restaurant"],
            "maxResultCount": 20,  # API allows up to 20 results
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lon,
                    },
                    "radius": float(radius_m),
                }
            },
        }

        # Use field mask to request only needed fields (reduces API cost)
        headers = {
            "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.rating,places.userRatingCount,places.types"
        }

        try:
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Google Places API returned {len(data.get('places', []))} places near ({lat}, {lon})")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Google Places API rate limit exceeded")
            else:
                logger.error(f"Google Places API request failed: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Failed to search Google Places API: {e}")
            return []

        results = self._parse_nearby_response(data, min_rating)
        logger.info(f"After filtering: {len(results)} pubs with rating >= {min_rating}")
        return results

    def _parse_nearby_response(
        self, data: dict, min_rating: float
    ) -> list[PlaceResult]:
        """Parse Google Places API nearby search response.

        Args:
            data: Raw JSON response from Google Places API.
            min_rating: Minimum rating to filter results.

        Returns:
            List of PlaceResult objects.
        """
        results = []
        places = data.get("places", [])

        for place in places:
            # Extract rating
            rating = place.get("rating", 0.0)
            if rating < min_rating:
                continue

            # Extract location
            location = place.get("location", {})
            lat = location.get("latitude", 0.0)
            lon = location.get("longitude", 0.0)

            if not lat or not lon:
                continue

            # Extract place ID
            place_id = place.get("id", "")
            if not place_id:
                continue

            # Extract name
            display_name = place.get("displayName", {})
            name = display_name.get("text", "")

            # Extract types
            types = place.get("types", [])

            # Check if it's a pub, bar, or restaurant
            # We're searching for bars/restaurants, so any result is potentially valid
            # Google Places types include: bar, restaurant, pub, cafe, etc.
            pub_types = {"bar", "pub", "restaurant", "tavern", "brewery", "wine_bar", "night_club"}
            is_valid = any(t in pub_types for t in types)

            # If no recognized types, still accept if it has a good rating (API returned it for bar/restaurant search)
            if not is_valid and not types:
                is_valid = True  # Accept if types list is empty (rare edge case)

            if not is_valid:
                logger.debug(f"Skipping place '{name}' - types {types} don't match pub criteria")
                continue

            # Extract user ratings total
            user_ratings_total = place.get("userRatingCount")

            results.append(
                PlaceResult(
                    place_id=place_id,
                    name=name,
                    latitude=lat,
                    longitude=lon,
                    rating=rating,
                    user_ratings_total=user_ratings_total,
                    types=types,
                    metadata=place,
                )
            )

        return results

    async def get_place_details(self, place_id: str) -> Optional[dict]:
        """Get detailed information about a place.

        Args:
            place_id: Google Place ID.

        Returns:
            Place details dictionary or None if failed.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        if not self.api_key:
            logger.error("Cannot make Google Places request without API key")
            return None

        await self.rate_limiter.acquire()

        url = f"{self.BASE_URL}/places/{place_id}"

        # Request specific fields to reduce response size
        params = {
            "fields": "id,displayName,location,rating,userRatingCount,types,formattedAddress,websiteUri,internationalPhoneNumber",
        }

        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Google Places API rate limit exceeded")
            else:
                logger.error(f"Google Places API details request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get place details: {e}")
            return None





