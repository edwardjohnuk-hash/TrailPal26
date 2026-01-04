"""Pub recommendation service for finding pubs along hiking routes."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from trail_pal.db.models import Connection
from trail_pal.services.google_places_client import GooglePlacesClient, PlaceResult

logger = logging.getLogger(__name__)


@dataclass
class PubRecommendation:
    """A pub recommendation for a location."""

    name: str
    rating: float
    latitude: float
    longitude: float
    place_id: str
    distance_m: float
    user_ratings_total: Optional[int] = None


@dataclass
class DayPubRecommendations:
    """Pub recommendations for a single day's hike."""

    day_number: int
    start_pub: Optional[PubRecommendation] = None
    end_pub: Optional[PubRecommendation] = None
    midpoint_pub: Optional[PubRecommendation] = None


class PubRecommenderService:
    """Service for recommending pubs at waypoints and route midpoints."""

    MIN_RATING = 4.2
    SEARCH_RADIUS_M = 500

    def __init__(self, db: Session):
        """Initialize the pub recommender.

        Args:
            db: SQLAlchemy session for database queries.
        """
        self._db = db

    def _haversine_distance_m(
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

    def _get_route_midpoint(self, connection_id: UUID) -> Optional[tuple[float, float]]:
        """Get the midpoint of a route from its geometry.

        Args:
            connection_id: The connection ID.

        Returns:
            Tuple of (latitude, longitude) for the midpoint, or None if not found.
        """
        stmt = select(Connection).where(Connection.id == connection_id)
        connection = self._db.execute(stmt).scalar_one_or_none()

        if not connection or not connection.route_geometry:
            return None

        try:
            line = to_shape(connection.route_geometry)
            coords = list(line.coords)

            if not coords:
                return None

            # Get the middle coordinate
            mid_index = len(coords) // 2
            mid_coord = coords[mid_index]

            # coords are (lon, lat) in GeoJSON/WKT format
            return (mid_coord[1], mid_coord[0])  # Return as (lat, lon)
        except Exception as e:
            logger.warning(f"Failed to extract midpoint from connection {connection_id}: {e}")
            return None

    def _select_best_pub(
        self, pubs: list[PlaceResult], ref_lat: float, ref_lon: float
    ) -> Optional[PubRecommendation]:
        """Select the best pub from a list based on rating and distance.

        Args:
            pubs: List of PlaceResult objects.
            ref_lat: Reference latitude for distance calculation.
            ref_lon: Reference longitude for distance calculation.

        Returns:
            The best PubRecommendation, or None if no pubs available.
        """
        if not pubs:
            return None

        # Sort by rating (descending), then by distance (ascending)
        pubs_with_distance = []
        for pub in pubs:
            distance = self._haversine_distance_m(
                ref_lon, ref_lat, pub.longitude, pub.latitude
            )
            pubs_with_distance.append((pub, distance))

        # Sort: higher rating first, then closer distance
        pubs_with_distance.sort(key=lambda x: (-x[0].rating, x[1]))

        best_pub, distance = pubs_with_distance[0]

        return PubRecommendation(
            name=best_pub.name,
            rating=best_pub.rating,
            latitude=best_pub.latitude,
            longitude=best_pub.longitude,
            place_id=best_pub.place_id,
            distance_m=round(distance, 1),
            user_ratings_total=best_pub.user_ratings_total,
        )

    async def find_pub_at_location(
        self,
        lat: float,
        lon: float,
        client: GooglePlacesClient,
    ) -> Optional[PubRecommendation]:
        """Find the best pub near a location.

        Args:
            lat: Latitude.
            lon: Longitude.
            client: Google Places API client.

        Returns:
            PubRecommendation or None if no pubs with 4.2+ rating found.
        """
        logger.info(f"Searching for pubs near ({lat}, {lon}) with radius {self.SEARCH_RADIUS_M}m")
        pubs = await client.search_nearby_pubs(
            lat=lat,
            lon=lon,
            radius_m=self.SEARCH_RADIUS_M,
            min_rating=self.MIN_RATING,
        )

        best_pub = self._select_best_pub(pubs, lat, lon)
        if best_pub:
            logger.info(f"Found pub: {best_pub.name} (rating: {best_pub.rating})")
        else:
            logger.info(f"No pubs found at ({lat}, {lon}) meeting criteria")
        return best_pub

    async def get_recommendations_for_day(
        self,
        day_number: int,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        connection_id: Optional[UUID],
        client: GooglePlacesClient,
    ) -> DayPubRecommendations:
        """Get pub recommendations for a single day's hike.

        Args:
            day_number: The day number.
            start_lat: Start waypoint latitude.
            start_lon: Start waypoint longitude.
            end_lat: End waypoint latitude.
            end_lon: End waypoint longitude.
            connection_id: The connection ID for midpoint calculation.
            client: Google Places API client.

        Returns:
            DayPubRecommendations with pubs for start, end, and midpoint.
        """
        # Get midpoint coordinates if connection exists
        midpoint = None
        if connection_id:
            midpoint = self._get_route_midpoint(connection_id)

        # Search for pubs at all locations in parallel
        tasks = [
            self.find_pub_at_location(start_lat, start_lon, client),
            self.find_pub_at_location(end_lat, end_lon, client),
        ]

        if midpoint:
            tasks.append(self.find_pub_at_location(midpoint[0], midpoint[1], client))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        start_pub = results[0] if not isinstance(results[0], Exception) else None
        end_pub = results[1] if not isinstance(results[1], Exception) else None
        midpoint_pub = None

        if len(results) > 2 and not isinstance(results[2], Exception):
            midpoint_pub = results[2]

        if isinstance(results[0], Exception):
            logger.warning(f"Error finding start pub for day {day_number}: {results[0]}")
        if isinstance(results[1], Exception):
            logger.warning(f"Error finding end pub for day {day_number}: {results[1]}")
        if len(results) > 2 and isinstance(results[2], Exception):
            logger.warning(f"Error finding midpoint pub for day {day_number}: {results[2]}")

        return DayPubRecommendations(
            day_number=day_number,
            start_pub=start_pub,
            end_pub=end_pub,
            midpoint_pub=midpoint_pub,
        )

    async def get_recommendations_for_itinerary(
        self,
        days: list[dict],
    ) -> list[DayPubRecommendations]:
        """Get pub recommendations for an entire itinerary.

        Args:
            days: List of day dictionaries with keys:
                - day_number: int
                - start_lat: float
                - start_lon: float
                - end_lat: float
                - end_lon: float
                - connection_id: Optional[UUID]

        Returns:
            List of DayPubRecommendations for each day.
        """
        async with GooglePlacesClient() as client:
            tasks = [
                self.get_recommendations_for_day(
                    day_number=day["day_number"],
                    start_lat=day["start_lat"],
                    start_lon=day["start_lon"],
                    end_lat=day["end_lat"],
                    end_lon=day["end_lon"],
                    connection_id=day.get("connection_id"),
                    client=client,
                )
                for day in days
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            recommendations = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting pub recommendations for day {i + 1}: {result}")
                    recommendations.append(
                        DayPubRecommendations(day_number=i + 1)
                    )
                else:
                    recommendations.append(result)

            return recommendations

