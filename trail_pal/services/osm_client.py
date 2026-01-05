"""OpenStreetMap Overpass API client for fetching waypoints."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from trail_pal.config import get_settings
from trail_pal.db.models import WaypointType

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """A geographic bounding box."""

    south: float  # min latitude
    west: float  # min longitude
    north: float  # max latitude
    east: float  # max longitude

    def to_overpass_bbox(self) -> str:
        """Convert to Overpass API bbox format (south,west,north,east)."""
        return f"{self.south},{self.west},{self.north},{self.east}"


@dataclass
class OSMElement:
    """A parsed OpenStreetMap element."""

    osm_id: str
    osm_type: str  # node, way, relation
    name: str
    waypoint_type: str
    latitude: float
    longitude: float
    tags: dict
    has_accommodation: bool = False
    has_water: bool = False
    has_food: bool = False


# Known regions with their bounding boxes
REGION_BOUNDS = {
    "cornwall": BoundingBox(
        south=49.95,
        west=-5.75,
        north=50.75,
        east=-4.15,
    ),
    "lake_district": BoundingBox(
        south=54.20,
        west=-3.50,
        north=54.70,
        east=-2.70,
    ),
}


class OverpassClient:
    """Client for querying OpenStreetMap via the Overpass API."""

    # Mapping of OSM tags to waypoint types
    # Order matters: more specific matches first
    TAG_MAPPINGS = [
        # Transport-accessible locations
        ({"railway": "station"}, WaypointType.TRAIN_STATION),
        ({"public_transport": "station", "railway": "station"}, WaypointType.TRAIN_STATION),
        ({"place": "city"}, WaypointType.CITY),
        ({"place": "town"}, WaypointType.TOWN),
        ({"place": "village"}, WaypointType.VILLAGE),
        # Accommodations
        ({"tourism": "camp_site"}, WaypointType.CAMPSITE),
        ({"tourism": "hostel"}, WaypointType.HOSTEL),
        ({"tourism": "guest_house"}, WaypointType.GUEST_HOUSE),
        ({"tourism": "hotel"}, WaypointType.HOTEL),
        # Scenic points
        ({"tourism": "viewpoint"}, WaypointType.VIEWPOINT),
        ({"natural": "peak"}, WaypointType.PEAK),
    ]

    def __init__(self, api_url: Optional[str] = None):
        """Initialize the Overpass client.

        Args:
            api_url: Overpass API endpoint URL. Defaults to public endpoint.
        """
        settings = get_settings()
        self.api_url = api_url or settings.overpass_api_url
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Enter async context manager."""
        self._client = httpx.AsyncClient(timeout=120.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()

    def _build_query(self, bbox: BoundingBox) -> str:
        """Build an Overpass QL query for waypoints in a bounding box.

        Args:
            bbox: The bounding box to search within.

        Returns:
            Overpass QL query string.
        """
        bbox_str = bbox.to_overpass_bbox()

        # Query for accommodations, scenic points, and transport-accessible locations
        query = f"""
[out:json][timeout:90];
(
  // Transport-accessible locations
  node["railway"="station"]({bbox_str});
  node["public_transport"="station"]({bbox_str});
  node["place"="village"]({bbox_str});
  way["place"="village"]({bbox_str});
  node["place"="town"]({bbox_str});
  way["place"="town"]({bbox_str});
  node["place"="city"]({bbox_str});
  way["place"="city"]({bbox_str});
  // Accommodations
  node["tourism"="camp_site"]({bbox_str});
  way["tourism"="camp_site"]({bbox_str});
  node["tourism"="hostel"]({bbox_str});
  way["tourism"="hostel"]({bbox_str});
  node["tourism"="guest_house"]({bbox_str});
  way["tourism"="guest_house"]({bbox_str});
  node["tourism"="hotel"]({bbox_str});
  way["tourism"="hotel"]({bbox_str});
  // Scenic points
  node["tourism"="viewpoint"]({bbox_str});
  node["natural"="peak"]({bbox_str});
);
out center;
"""
        return query

    async def fetch_waypoints(self, bbox: BoundingBox) -> list[OSMElement]:
        """Fetch waypoints from OpenStreetMap within a bounding box.

        Args:
            bbox: The bounding box to search within.

        Returns:
            List of parsed OSM elements.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        query = self._build_query(bbox)
        logger.info(f"Fetching waypoints from Overpass API for bbox: {bbox}")

        try:
            response = await self._client.post(
                self.api_url,
                data={"data": query},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Overpass API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch from Overpass API: {e}")
            raise

        elements = data.get("elements", [])
        logger.info(f"Received {len(elements)} elements from Overpass API")

        return self._parse_elements(elements)

    def _parse_elements(self, elements: list[dict]) -> list[OSMElement]:
        """Parse Overpass API response elements into OSMElement objects.

        Args:
            elements: Raw elements from Overpass API response.

        Returns:
            List of parsed OSMElement objects.
        """
        parsed = []

        for element in elements:
            osm_type = element.get("type", "node")
            osm_id = str(element.get("id", ""))
            tags = element.get("tags", {})

            # Get coordinates (center for ways/relations)
            if osm_type == "node":
                lat = element.get("lat")
                lon = element.get("lon")
            else:
                center = element.get("center", {})
                lat = center.get("lat")
                lon = center.get("lon")

            if lat is None or lon is None:
                continue

            # Determine waypoint type from tags
            waypoint_type = self._get_waypoint_type(tags)
            if not waypoint_type:
                continue

            # Get name (fallback to type if no name)
            name = tags.get("name")
            if not name:
                # Build a more descriptive name for unnamed accommodations
                name = f"Unnamed {waypoint_type}"
                # Try to add location context from address tags
                location_name = (
                    tags.get("addr:city")
                    or tags.get("addr:town")
                    or tags.get("addr:village")
                    or tags.get("addr:hamlet")
                    or tags.get("is_in")
                )
                if location_name:
                    # Clean up is_in format (often comma-separated list)
                    location_name = location_name.split(",")[0].strip()
                    name = f"{name}, {location_name}"

            # Determine amenities
            has_accommodation = waypoint_type in [
                WaypointType.CAMPSITE,
                WaypointType.HOSTEL,
                WaypointType.GUEST_HOUSE,
                WaypointType.HOTEL,
            ]
            has_water = tags.get("drinking_water") == "yes" or "water" in tags
            has_food = (
                tags.get("restaurant") == "yes"
                or tags.get("cafe") == "yes"
                or "food" in str(tags.get("amenity", "")).lower()
            )

            parsed.append(
                OSMElement(
                    osm_id=osm_id,
                    osm_type=osm_type,
                    name=name,
                    waypoint_type=waypoint_type,
                    latitude=lat,
                    longitude=lon,
                    tags=tags,
                    has_accommodation=has_accommodation,
                    has_water=has_water,
                    has_food=has_food,
                )
            )

        logger.info(f"Parsed {len(parsed)} valid waypoints")
        return parsed

    def _get_waypoint_type(self, tags: dict) -> Optional[str]:
        """Determine waypoint type from OSM tags.

        Args:
            tags: OSM tags dictionary.

        Returns:
            Waypoint type string or None if no match.
        """
        for tag_match, waypoint_type in self.TAG_MAPPINGS:
            if all(tags.get(k) == v for k, v in tag_match.items()):
                return waypoint_type
        return None

    async def fetch_region_waypoints(self, region_name: str) -> list[OSMElement]:
        """Fetch waypoints for a known region.

        Args:
            region_name: Name of the region (e.g., 'cornwall').

        Returns:
            List of parsed OSM elements.

        Raises:
            ValueError: If region is not known.
        """
        region_key = region_name.lower()
        if region_key not in REGION_BOUNDS:
            raise ValueError(
                f"Unknown region: {region_name}. "
                f"Known regions: {list(REGION_BOUNDS.keys())}"
            )

        bbox = REGION_BOUNDS[region_key]
        return await self.fetch_waypoints(bbox)


def get_region_bounds(region_name: str) -> BoundingBox:
    """Get bounding box for a known region.

    Args:
        region_name: Name of the region (e.g., 'cornwall').

    Returns:
        BoundingBox for the region.

    Raises:
        ValueError: If region is not known.
    """
    region_key = region_name.lower()
    if region_key not in REGION_BOUNDS:
        raise ValueError(
            f"Unknown region: {region_name}. "
            f"Known regions: {list(REGION_BOUNDS.keys())}"
        )
    return REGION_BOUNDS[region_key]


async def main():
    """Test the Overpass client."""
    async with OverpassClient() as client:
        waypoints = await client.fetch_region_waypoints("cornwall")
        print(f"Found {len(waypoints)} waypoints in Cornwall")
        for wp in waypoints[:5]:
            print(f"  - {wp.name} ({wp.waypoint_type}): {wp.latitude}, {wp.longitude}")


if __name__ == "__main__":
    asyncio.run(main())

