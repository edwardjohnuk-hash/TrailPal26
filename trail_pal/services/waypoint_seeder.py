"""Waypoint seeder service for populating the database from OpenStreetMap."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from geoalchemy2.shape import from_shape
from shapely.geometry import Point, box
from sqlalchemy import select
from sqlalchemy.orm import Session

from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Region, Waypoint
from trail_pal.services.osm_client import (
    OSMElement,
    OverpassClient,
    get_region_bounds,
    REGION_BOUNDS,
)

logger = logging.getLogger(__name__)


class WaypointSeeder:
    """Service for seeding waypoints from OpenStreetMap into the database."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize the seeder.

        Args:
            db: SQLAlchemy session. If not provided, creates a new one.
        """
        self._db = db
        self._owns_db = db is None

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

    def get_or_create_region(self, region_name: str) -> Region:
        """Get or create a region record.

        Args:
            region_name: Name of the region (e.g., 'cornwall').

        Returns:
            Region database object.
        """
        db = self._get_db()
        region_key = region_name.lower()

        # Check if region exists
        stmt = select(Region).where(Region.name == region_key)
        region = db.execute(stmt).scalar_one_or_none()

        if region:
            logger.info(f"Found existing region: {region.name}")
            return region

        # Create new region
        bounds = get_region_bounds(region_name)

        # Create bounding box polygon
        bbox_polygon = box(bounds.west, bounds.south, bounds.east, bounds.north)

        region = Region(
            id=uuid.uuid4(),
            name=region_key,
            country="England" if region_key == "cornwall" else "Unknown",
            description=f"Hiking region: {region_name.title()}",
            bounds=from_shape(bbox_polygon, srid=4326),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        db.add(region)
        db.commit()
        db.refresh(region)

        logger.info(f"Created new region: {region.name}")
        return region

    def _osm_element_to_waypoint(
        self, element: OSMElement, region_id: uuid.UUID
    ) -> Waypoint:
        """Convert an OSM element to a Waypoint model.

        Args:
            element: Parsed OSM element.
            region_id: ID of the region.

        Returns:
            Waypoint model instance.
        """
        point = Point(element.longitude, element.latitude)

        return Waypoint(
            id=uuid.uuid4(),
            region_id=region_id,
            name=element.name,
            waypoint_type=element.waypoint_type,
            location=from_shape(point, srid=4326),
            latitude=element.latitude,
            longitude=element.longitude,
            osm_id=element.osm_id,
            osm_type=element.osm_type,
            amenities=element.tags,
            has_accommodation=element.has_accommodation,
            has_water=element.has_water,
            has_food=element.has_food,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def _upsert_waypoint(self, waypoint: Waypoint) -> bool:
        """Upsert a waypoint (update if exists, insert if not).

        Args:
            waypoint: Waypoint to upsert.

        Returns:
            True if inserted, False if updated.
        """
        db = self._get_db()

        # Check if waypoint exists by OSM ID
        stmt = select(Waypoint).where(
            Waypoint.osm_id == waypoint.osm_id,
            Waypoint.osm_type == waypoint.osm_type,
        )
        existing = db.execute(stmt).scalar_one_or_none()

        if existing:
            # Update existing waypoint
            existing.name = waypoint.name
            existing.waypoint_type = waypoint.waypoint_type
            existing.latitude = waypoint.latitude
            existing.longitude = waypoint.longitude
            existing.location = waypoint.location
            existing.amenities = waypoint.amenities
            existing.has_accommodation = waypoint.has_accommodation
            existing.has_water = waypoint.has_water
            existing.has_food = waypoint.has_food
            existing.updated_at = datetime.utcnow()
            return False
        else:
            db.add(waypoint)
            return True

    async def seed_region(self, region_name: str) -> dict:
        """Seed waypoints for a region from OpenStreetMap.

        Args:
            region_name: Name of the region (e.g., 'cornwall').

        Returns:
            Dictionary with seeding statistics.
        """
        logger.info(f"Starting waypoint seeding for region: {region_name}")

        # Get or create region
        region = self.get_or_create_region(region_name)

        # Fetch waypoints from OSM
        async with OverpassClient() as client:
            elements = await client.fetch_region_waypoints(region_name)

        logger.info(f"Fetched {len(elements)} elements from OpenStreetMap")

        # Convert and upsert waypoints
        db = self._get_db()
        inserted = 0
        updated = 0

        for element in elements:
            waypoint = self._osm_element_to_waypoint(element, region.id)
            if self._upsert_waypoint(waypoint):
                inserted += 1
            else:
                updated += 1

        # Update region sync timestamp
        region.last_synced = datetime.utcnow()
        region.updated_at = datetime.utcnow()

        db.commit()

        stats = {
            "region": region_name,
            "total_fetched": len(elements),
            "inserted": inserted,
            "updated": updated,
            "last_synced": region.last_synced.isoformat(),
        }

        logger.info(f"Seeding complete: {stats}")
        return stats

    def get_region_stats(self, region_name: str) -> Optional[dict]:
        """Get statistics for a region.

        Args:
            region_name: Name of the region.

        Returns:
            Dictionary with region statistics or None if not found.
        """
        db = self._get_db()
        region_key = region_name.lower()

        stmt = select(Region).where(Region.name == region_key)
        region = db.execute(stmt).scalar_one_or_none()

        if not region:
            return None

        # Count waypoints by type
        waypoint_stmt = select(Waypoint).where(Waypoint.region_id == region.id)
        waypoints = db.execute(waypoint_stmt).scalars().all()

        type_counts = {}
        for wp in waypoints:
            type_counts[wp.waypoint_type] = type_counts.get(wp.waypoint_type, 0) + 1

        return {
            "region": region.name,
            "country": region.country,
            "total_waypoints": len(waypoints),
            "waypoints_by_type": type_counts,
            "last_synced": region.last_synced.isoformat() if region.last_synced else None,
        }


def list_available_regions() -> list[str]:
    """List all available regions that can be seeded.

    Returns:
        List of region names.
    """
    return list(REGION_BOUNDS.keys())


async def seed_region(region_name: str) -> dict:
    """Convenience function to seed a region.

    Args:
        region_name: Name of the region.

    Returns:
        Seeding statistics.
    """
    seeder = WaypointSeeder()
    try:
        return await seeder.seed_region(region_name)
    finally:
        seeder.close()


async def main():
    """Test the waypoint seeder."""
    print("Available regions:", list_available_regions())
    print("\nSeeding Cornwall...")
    stats = await seed_region("cornwall")
    print(f"Seeding complete: {stats}")


if __name__ == "__main__":
    asyncio.run(main())

