"""OpenRouteService API client for hiking route calculations."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from trail_pal.config import get_settings

logger = logging.getLogger(__name__)


# ORS surface type value mappings
SURFACE_TYPES = {
    0: "unknown",
    1: "paved",
    2: "unpaved",
    3: "asphalt",
    4: "concrete",
    5: "cobblestone",
    6: "metal",
    7: "wood",
    8: "compacted_gravel",
    9: "fine_gravel",
    10: "gravel",
    11: "dirt",
    12: "ground",
    13: "ice",
    14: "paving_stones",
    15: "sand",
    16: "woodchips",
    17: "grass",
    18: "grass_paver",
}

# ORS way type value mappings
WAY_TYPES = {
    0: "unknown",
    1: "state_road",
    2: "road",
    3: "street",
    4: "path",
    5: "track",
    6: "cycleway",
    7: "footway",
    8: "steps",
    9: "ferry",
    10: "construction",
}


@dataclass
class SurfaceBreakdown:
    """Breakdown of route by surface and way types."""

    # surface type -> distance in km
    surfaces: dict[str, float] = field(default_factory=dict)
    # way type -> distance in km
    waytypes: dict[str, float] = field(default_factory=dict)
    # Total distance for validation
    total_distance_km: float = 0.0

    def surface_percentages(self) -> dict[str, float]:
        """Get surface type percentages, sorted by percentage descending."""
        if self.total_distance_km == 0:
            return {}
        percentages = {
            surface: (distance / self.total_distance_km) * 100
            for surface, distance in self.surfaces.items()
        }
        return dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

    def waytype_percentages(self) -> dict[str, float]:
        """Get way type percentages, sorted by percentage descending."""
        if self.total_distance_km == 0:
            return {}
        percentages = {
            waytype: (distance / self.total_distance_km) * 100
            for waytype, distance in self.waytypes.items()
        }
        return dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "surfaces": self.surfaces,
            "waytypes": self.waytypes,
            "total_distance_km": self.total_distance_km,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SurfaceBreakdown":
        """Create from dictionary."""
        return cls(
            surfaces=data.get("surfaces", {}),
            waytypes=data.get("waytypes", {}),
            total_distance_km=data.get("total_distance_km", 0.0),
        )


@dataclass
class RouteResult:
    """Result of a route calculation."""

    distance_km: float
    duration_minutes: int
    elevation_gain_m: Optional[float]
    elevation_loss_m: Optional[float]
    geometry: list[tuple[float, float]]  # List of (lon, lat) coordinates
    metadata: dict
    surface_breakdown: Optional[SurfaceBreakdown] = None


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


class OpenRouteServiceClient:
    """Client for the OpenRouteService API."""

    PROFILE_FOOT_HIKING = "foot-hiking"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
    ):
        """Initialize the ORS client.

        Args:
            api_key: OpenRouteService API key.
            base_url: ORS API base URL.
            requests_per_minute: Rate limit for requests.
        """
        settings = get_settings()
        self.api_key = api_key or settings.ors_api_key
        self.base_url = base_url or settings.ors_base_url
        self.rate_limiter = RateLimiter(
            requests_per_minute or settings.ors_requests_per_minute
        )
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key:
            logger.warning(
                "No ORS API key configured. Set ORS_API_KEY environment variable."
            )

    async def __aenter__(self):
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()

    async def get_hiking_route(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
    ) -> Optional[RouteResult]:
        """Calculate a hiking route between two points.

        Args:
            start_lon: Starting longitude.
            start_lat: Starting latitude.
            end_lon: Ending longitude.
            end_lat: Ending latitude.

        Returns:
            RouteResult with distance, duration, and geometry, or None if failed.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        if not self.api_key:
            logger.error("Cannot make ORS request without API key")
            return None

        await self.rate_limiter.acquire()

        url = f"{self.base_url}/v2/directions/{self.PROFILE_FOOT_HIKING}"

        payload = {
            "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
            "elevation": True,
            "instructions": False,
            "geometry_simplify": False,  # Get full geometry
            "preference": "recommended",  # hiking-appropriate routing
            "extra_info": ["surface", "waytype"],  # Request surface and way type data
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("ORS rate limit exceeded")
            else:
                logger.error(f"ORS API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get route from ORS: {e}")
            return None

        return self._parse_route_response(data)

    def _parse_route_response(self, data: dict) -> Optional[RouteResult]:
        """Parse ORS directions response.

        Args:
            data: Raw JSON response from ORS API.

        Returns:
            Parsed RouteResult or None if parsing fails.
        """
        try:
            routes = data.get("routes", [])
            if not routes:
                logger.warning("No routes found in ORS response")
                return None

            route = routes[0]
            summary = route.get("summary", {})

            # Distance in meters -> km
            distance_km = summary.get("distance", 0) / 1000.0

            # Duration in seconds -> minutes
            duration_minutes = int(summary.get("duration", 0) / 60)

            # Elevation (if available)
            elevation_gain_m = summary.get("ascent")
            elevation_loss_m = summary.get("descent")

            # Parse geometry (encoded polyline)
            # ORS returns 3D polylines when elevation=True in the request
            geometry_encoded = route.get("geometry", "")
            geometry = []
            if geometry_encoded and len(geometry_encoded) > 0:
                try:
                    geometry = self._decode_polyline(geometry_encoded, is_3d=True)
                except Exception:
                    # Skip geometry if decoding fails
                    logger.debug("Failed to decode polyline, skipping geometry")

            # Parse surface and waytype extras
            surface_breakdown = self._parse_extras(route, geometry)

            metadata = {
                "way_points": route.get("way_points", []),
            }
            
            # Include surface breakdown in metadata for storage
            if surface_breakdown:
                metadata["surface_breakdown"] = surface_breakdown.to_dict()

            return RouteResult(
                distance_km=distance_km,
                duration_minutes=duration_minutes,
                elevation_gain_m=elevation_gain_m,
                elevation_loss_m=elevation_loss_m,
                geometry=geometry,
                metadata=metadata,
                surface_breakdown=surface_breakdown,
            )

        except Exception as e:
            logger.error(f"Failed to parse ORS response: {e}")
            return None

    def _parse_extras(
        self, route: dict, geometry: list[tuple[float, float]]
    ) -> Optional[SurfaceBreakdown]:
        """Parse extra_info from ORS response to calculate surface breakdown.

        Args:
            route: Route object from ORS response.
            geometry: Decoded geometry coordinates.

        Returns:
            SurfaceBreakdown or None if parsing fails.
        """
        extras = route.get("extras", {})
        if not extras or not geometry:
            return None

        try:
            # Calculate distances between consecutive geometry points
            segment_distances = self._calculate_segment_distances(geometry)
            total_distance = sum(segment_distances)

            surfaces: dict[str, float] = {}
            waytypes: dict[str, float] = {}

            # Parse surface data
            surface_data = extras.get("surface", {}).get("values", [])
            for start_idx, end_idx, value in surface_data:
                surface_name = SURFACE_TYPES.get(value, f"unknown_{value}")
                # Sum distances for segments in this range
                segment_distance = sum(segment_distances[start_idx:end_idx])
                surfaces[surface_name] = surfaces.get(surface_name, 0) + segment_distance

            # Parse waytype data
            waytype_data = extras.get("waytype", {}).get("values", [])
            for start_idx, end_idx, value in waytype_data:
                waytype_name = WAY_TYPES.get(value, f"unknown_{value}")
                segment_distance = sum(segment_distances[start_idx:end_idx])
                waytypes[waytype_name] = waytypes.get(waytype_name, 0) + segment_distance

            # Convert from meters to km
            surfaces_km = {k: v / 1000.0 for k, v in surfaces.items()}
            waytypes_km = {k: v / 1000.0 for k, v in waytypes.items()}
            total_km = total_distance / 1000.0

            return SurfaceBreakdown(
                surfaces=surfaces_km,
                waytypes=waytypes_km,
                total_distance_km=total_km,
            )

        except Exception as e:
            logger.warning(f"Failed to parse surface extras: {e}")
            return None

    def _calculate_segment_distances(
        self, geometry: list[tuple[float, float]]
    ) -> list[float]:
        """Calculate distances between consecutive geometry points.

        Args:
            geometry: List of (lon, lat) coordinate tuples.

        Returns:
            List of distances in meters between consecutive points.
        """
        distances = []
        for i in range(len(geometry) - 1):
            lon1, lat1 = geometry[i]
            lon2, lat2 = geometry[i + 1]
            distance = self._haversine_distance(lon1, lat1, lon2, lat2)
            distances.append(distance)
        return distances

    def _haversine_distance(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula.

        Args:
            lon1, lat1: First point coordinates.
            lon2, lat2: Second point coordinates.

        Returns:
            Distance in meters.
        """
        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _decode_polyline(
        self, encoded: str, is_3d: bool = False, precision: int = 5
    ) -> list[tuple[float, float]]:
        """Decode a Google-style encoded polyline.

        Args:
            encoded: Encoded polyline string.
            is_3d: If True, decode as 3D polyline with elevation (triplets).
                   If False, decode as 2D polyline (pairs).
            precision: Coordinate precision (5 for ORS).

        Returns:
            List of (longitude, latitude) coordinate tuples.
        """
        coordinates = []
        index = 0
        lat = 0
        lon = 0
        elevation = 0
        factor = 10**precision
        length = len(encoded)

        while index < length:
            # Decode latitude
            shift = 0
            result = 0
            while index < length:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            lat += (~(result >> 1)) if (result & 1) else (result >> 1)

            # Decode longitude
            shift = 0
            result = 0
            while index < length:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            lon += (~(result >> 1)) if (result & 1) else (result >> 1)

            # Decode elevation if 3D polyline
            if is_3d:
                shift = 0
                result = 0
                while index < length:
                    byte = ord(encoded[index]) - 63
                    index += 1
                    result |= (byte & 0x1F) << shift
                    shift += 5
                    if byte < 0x20:
                        break
                elevation += (~(result >> 1)) if (result & 1) else (result >> 1)

            coordinates.append((lon / factor, lat / factor))

        return coordinates

    async def get_multiple_routes(
        self,
        coordinate_pairs: list[tuple[float, float, float, float]],
    ) -> list[Optional[RouteResult]]:
        """Calculate multiple hiking routes.

        Args:
            coordinate_pairs: List of (start_lon, start_lat, end_lon, end_lat) tuples.

        Returns:
            List of RouteResult objects (None for failed routes).
        """
        results = []
        total = len(coordinate_pairs)

        for i, (start_lon, start_lat, end_lon, end_lat) in enumerate(coordinate_pairs):
            logger.info(f"Calculating route {i + 1}/{total}")
            result = await self.get_hiking_route(
                start_lon, start_lat, end_lon, end_lat
            )
            results.append(result)

        return results


def calculate_straight_line_distance_km(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Calculate straight-line distance between two points using Haversine formula.

    Args:
        lon1: First point longitude.
        lat1: First point latitude.
        lon2: Second point longitude.
        lat2: Second point latitude.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


async def main():
    """Test the ORS client."""
    async with OpenRouteServiceClient() as client:
        # Test route between two points in Cornwall
        result = await client.get_hiking_route(
            start_lon=-5.0527,
            start_lat=50.1500,  # Near Penzance
            end_lon=-5.0700,
            end_lat=50.1200,  # Nearby point
        )
        if result:
            print(f"Route: {result.distance_km:.2f} km, {result.duration_minutes} min")
            print(f"Elevation: +{result.elevation_gain_m}m / -{result.elevation_loss_m}m")
        else:
            print("Failed to get route (check API key)")


if __name__ == "__main__":
    asyncio.run(main())

