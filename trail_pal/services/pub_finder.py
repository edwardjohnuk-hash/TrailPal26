"""Pub finder service for discovering pubs near waypoints and routes."""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime
from typing import Optional

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, LineString
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Pub, Region, Waypoint
from trail_pal.services.google_places_client import GooglePlacesClient, PlaceResult

logger = logging.getLogger(__name__)


class PubFinder:
    """Service for finding pubs near waypoints and routes using Google Places API."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize the pub finder.

        Args:
            db: SQLAlchemy session. If not provided, creates a new one.
        """
        self._db = db
        self._owns_db = db is None
        self._settings = get_settings()

    def _get_db(self) -> Session:
        """Get or create database session."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def close(self):
        """Close database session if we own it."""
        if self._owns_db and self._db is not None:
            self._db.close()
            self._db = None

    def get_region(self, region_name: str) -> Optional[Region]:
        """Get a region by name.

        Args:
            region_name: Name of the region.

        Returns:
            Region or None if not found.
        """
        db = self._get_db()
        stmt = select(Region).where(Region.name == region_name.lower())
        return db.execute(stmt).scalar_one_or_none()

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

    def _store_pub(
        self,
        place: PlaceResult,
        waypoint_id: Optional[uuid.UUID] = None,
        connection_id: Optional[uuid.UUID] = None,
        distance_m: float = 0.0,
        location_type: str = "waypoint",
    ) -> Pub:
        """Store or update a pub in the database.

        Args:
            place: PlaceResult from Google Places API.
            waypoint_id: Associated waypoint ID (if any).
            connection_id: Associated connection ID (if any).
            distance_m: Distance from waypoint/route in meters.
            location_type: 'waypoint' or 'route'.

        Returns:
            Pub model instance.
        """
        db = self._get_db()

        # Check if pub already exists
        stmt = select(Pub).where(Pub.google_place_id == place.place_id)
        existing = db.execute(stmt).scalar_one_or_none()

        point = Point(place.longitude, place.latitude)

        if existing:
            # Update existing pub
            existing.name = place.name
            existing.latitude = place.latitude
            existing.longitude = place.longitude
            existing.location = from_shape(point, srid=4326)
            existing.rating = place.rating
            existing.user_ratings_total = place.user_ratings_total
            existing.distance_m = distance_m
            existing.location_type = location_type
            existing.pub_metadata = place.metadata
            existing.updated_at = datetime.utcnow()

            # Update associations if provided
            if waypoint_id:
                existing.waypoint_id = waypoint_id
            if connection_id:
                existing.connection_id = connection_id

            return existing
        else:
            # Create new pub
            pub = Pub(
                id=uuid.uuid4(),
                waypoint_id=waypoint_id,
                connection_id=connection_id,
                google_place_id=place.place_id,
                name=place.name,
                location=from_shape(point, srid=4326),
                latitude=place.latitude,
                longitude=place.longitude,
                rating=place.rating,
                user_ratings_total=place.user_ratings_total,
                distance_m=distance_m,
                location_type=location_type,
                pub_metadata=place.metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(pub)
            return pub

    async def find_pubs_near_waypoints(
        self, region_name: str, radius_m: int = 500
    ) -> dict:
        """Find pubs within radius_m of waypoints in a region.

        Args:
            region_name: Name of the region.
            radius_m: Search radius in meters (default 500m).

        Returns:
            Dictionary with discovery statistics.
        """
        logger.info(f"Finding pubs near waypoints in {region_name} (radius: {radius_m}m)")

        region = self.get_region(region_name)
        if not region:
            raise ValueError(f"Region not found: {region_name}")

        db = self._get_db()

        # Get all waypoints for the region
        stmt = select(Waypoint).where(Waypoint.region_id == region.id)
        waypoints = list(db.execute(stmt).scalars().all())

        if not waypoints:
            return {
                "region": region_name,
                "waypoints_processed": 0,
                "pubs_found": 0,
                "pubs_stored": 0,
            }

        total_pubs_found = 0
        total_pubs_stored = 0

        async with GooglePlacesClient() as places_client:
            for i, waypoint in enumerate(waypoints):
                logger.info(
                    f"Processing waypoint {i + 1}/{len(waypoints)}: {waypoint.name}"
                )

                # Check if we already have pubs for this waypoint
                existing_stmt = select(Pub).where(
                    Pub.waypoint_id == waypoint.id, Pub.location_type == "waypoint"
                )
                existing_pubs = list(db.execute(existing_stmt).scalars().all())

                if existing_pubs:
                    logger.debug(
                        f"Skipping {waypoint.name} - already has {len(existing_pubs)} pubs cached"
                    )
                    continue

                # Search for pubs near this waypoint
                places = await places_client.search_nearby_pubs(
                    lat=waypoint.latitude,
                    lon=waypoint.longitude,
                    radius_m=radius_m,
                    min_rating=4.2,
                )

                # Store pubs with distance calculations
                for place in places:
                    distance_m = self._haversine_distance_m(
                        waypoint.longitude,
                        waypoint.latitude,
                        place.longitude,
                        place.latitude,
                    )

                    # Only store if within radius
                    if distance_m <= radius_m:
                        self._store_pub(
                            place,
                            waypoint_id=waypoint.id,
                            distance_m=distance_m,
                            location_type="waypoint",
                        )
                        total_pubs_stored += 1

                total_pubs_found += len(places)

                # Commit periodically
                if (i + 1) % 10 == 0:
                    db.commit()

        # Final commit
        db.commit()

        stats = {
            "region": region_name,
            "waypoints_processed": len(waypoints),
            "pubs_found": total_pubs_found,
            "pubs_stored": total_pubs_stored,
        }

        logger.info(f"Pub discovery complete: {stats}")
        return stats

    async def find_pubs_near_routes(
        self, region_name: str, buffer_m: int = 50
    ) -> dict:
        """Find pubs within buffer_m of route segments in a region.

        Args:
            region_name: Name of the region.
            buffer_m: Buffer distance in meters (default 50m).

        Returns:
            Dictionary with discovery statistics.
        """
        logger.info(f"Finding pubs near routes in {region_name} (buffer: {buffer_m}m)")

        region = self.get_region(region_name)
        if not region:
            raise ValueError(f"Region not found: {region_name}")

        db = self._get_db()

        # Get all waypoint IDs for the region
        wp_ids_stmt = select(Waypoint.id).where(Waypoint.region_id == region.id)
        wp_ids = list(db.execute(wp_ids_stmt).scalars().all())

        if not wp_ids:
            return {
                "region": region_name,
                "connections_processed": 0,
                "pubs_found": 0,
                "pubs_stored": 0,
            }

        # Get all connections for this region
        stmt = select(Connection).where(Connection.from_waypoint_id.in_(wp_ids))
        connections = list(db.execute(stmt).scalars().all())

        if not connections:
            return {
                "region": region_name,
                "connections_processed": 0,
                "pubs_found": 0,
                "pubs_stored": 0,
            }

        total_pubs_found = 0
        total_pubs_stored = 0

        async with GooglePlacesClient() as places_client:
            for i, connection in enumerate(connections):
                if not connection.route_geometry:
                    logger.debug(f"Skipping connection {connection.id} - no geometry")
                    continue

                logger.info(
                    f"Processing connection {i + 1}/{len(connections)}: "
                    f"{connection.from_waypoint_id} -> {connection.to_waypoint_id}"
                )

                # Check if we already have pubs for this connection
                existing_stmt = select(Pub).where(
                    Pub.connection_id == connection.id, Pub.location_type == "route"
                )
                existing_pubs = list(db.execute(existing_stmt).scalars().all())

                if existing_pubs:
                    logger.debug(
                        f"Skipping connection {connection.id} - already has {len(existing_pubs)} pubs cached"
                    )
                    continue

                # Extract route geometry
                try:
                    line = to_shape(connection.route_geometry)
                    coords = list(line.coords)
                except Exception as e:
                    logger.warning(f"Failed to extract geometry for connection {connection.id}: {e}")
                    continue

                if not coords:
                    continue

                # Sample points along the route (every ~100m)
                # Calculate approximate sampling interval
                total_distance = 0
                for j in range(len(coords) - 1):
                    lon1, lat1 = coords[j][:2]  # Handle 3D coords
                    lon2, lat2 = coords[j + 1][:2]
                    total_distance += self._haversine_distance_m(lon1, lat1, lon2, lat2)

                # Sample every 100m or at each coordinate, whichever is more frequent
                sample_interval = max(1, int(len(coords) / (total_distance / 100)))
                sampled_coords = coords[::sample_interval]
                if coords[-1] not in sampled_coords:
                    sampled_coords.append(coords[-1])

                # Search for pubs near each sampled point
                found_place_ids = set()  # Deduplicate pubs found at multiple points

                for coord in sampled_coords:
                    lon, lat = coord[0], coord[1]

                    # Search for pubs
                    places = await places_client.search_nearby_pubs(
                        lat=lat,
                        lon=lon,
                        radius_m=buffer_m * 2,  # Search wider to account for route width
                        min_rating=4.2,
                    )

                    # Check distance to route for each place
                    for place in places:
                        if place.place_id in found_place_ids:
                            continue

                        # Calculate minimum distance from place to route
                        min_distance = float("inf")
                        for j in range(len(coords) - 1):
                            lon1, lat1 = coords[j][:2]
                            lon2, lat2 = coords[j + 1][:2]

                            # Calculate distance from point to line segment
                            distance = self._point_to_line_distance(
                                place.longitude,
                                place.latitude,
                                lon1,
                                lat1,
                                lon2,
                                lat2,
                            )
                            min_distance = min(min_distance, distance)

                        # Only store if within buffer
                        if min_distance <= buffer_m:
                            found_place_ids.add(place.place_id)
                            self._store_pub(
                                place,
                                connection_id=connection.id,
                                distance_m=min_distance,
                                location_type="route",
                            )
                            total_pubs_stored += 1

                total_pubs_found += len(found_place_ids)

                # Commit periodically
                if (i + 1) % 10 == 0:
                    db.commit()

        # Final commit
        db.commit()

        stats = {
            "region": region_name,
            "connections_processed": len(connections),
            "pubs_found": total_pubs_found,
            "pubs_stored": total_pubs_stored,
        }

        logger.info(f"Route pub discovery complete: {stats}")
        return stats

    def _point_to_line_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calculate distance from a point to a line segment.

        Args:
            px, py: Point coordinates.
            x1, y1: Line segment start.
            x2, y2: Line segment end.

        Returns:
            Distance in meters.
        """
        # Convert to radians
        lat1, lon1 = math.radians(y1), math.radians(x1)
        lat2, lon2 = math.radians(y2), math.radians(x2)
        plat, plon = math.radians(py), math.radians(px)

        # Vector from p1 to p2
        dx = lat2 - lat1
        dy = lon2 - lon1

        # Vector from p1 to p
        dpx = plat - lat1
        dpy = plon - lon1

        # Calculate dot product
        dot = dpx * dx + dpy * dy
        len_sq = dx * dx + dy * dy

        if len_sq == 0:
            # Line segment is a point
            return self._haversine_distance_m(x1, y1, px, py)

        # Calculate parameter t (projection of p onto line)
        t = max(0, min(1, dot / len_sq))

        # Find closest point on line segment
        closest_lat = lat1 + t * dx
        closest_lon = lon1 + t * dy

        # Convert back to degrees
        closest_lat_deg = math.degrees(closest_lat)
        closest_lon_deg = math.degrees(closest_lon)

        # Calculate distance
        return self._haversine_distance_m(closest_lon_deg, closest_lat_deg, px, py)

    def get_waypoints_near_pubs(
        self, region_name: str, radius_m: int = 1609
    ) -> list[Waypoint]:
        """Get waypoints within radius_m of any pub.

        Args:
            region_name: Name of the region.
            radius_m: Search radius in meters (default 1609m = 1 mile).

        Returns:
            List of waypoints within radius of pubs.
        """
        logger.info(
            f"Finding waypoints within {radius_m}m of pubs in {region_name}"
        )

        region = self.get_region(region_name)
        if not region:
            raise ValueError(f"Region not found: {region_name}")

        db = self._get_db()

        # Use PostGIS ST_DWithin with geography type for accurate distance calculations
        # Use a cross join with WHERE clause for efficient spatial query
        stmt = text("""
            SELECT DISTINCT w.*
            FROM waypoints w
            CROSS JOIN pubs p
            WHERE w.region_id = :region_id
            AND ST_DWithin(
                ST_GeographyFromText(ST_AsText(w.location)),
                ST_GeographyFromText(ST_AsText(p.location)),
                :radius_m
            )
        """)
        
        result = db.execute(stmt, {"region_id": region.id, "radius_m": radius_m})
        waypoints = []
        for row in result:
            # Fetch the full waypoint object
            wp_stmt = select(Waypoint).where(Waypoint.id == row.id)
            waypoint = db.execute(wp_stmt).scalar_one()
            waypoints.append(waypoint)

        logger.info(f"Found {len(waypoints)} waypoints within {radius_m}m of pubs")
        return waypoints

