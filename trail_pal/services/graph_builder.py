"""Feasibility graph builder for computing valid hiking connections."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from itertools import combinations
from typing import Optional

from geoalchemy2.shape import from_shape
from shapely.geometry import LineString
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Region, Waypoint
from trail_pal.services.ors_client import (
    OpenRouteServiceClient,
    RouteResult,
    calculate_straight_line_distance_km,
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Service for building the feasibility graph between waypoints."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize the graph builder.

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

    def get_region_waypoints(
        self, region_id: uuid.UUID, accommodation_only: bool = True
    ) -> list[Waypoint]:
        """Get waypoints for a region.

        Args:
            region_id: ID of the region.
            accommodation_only: If True, only return waypoints with accommodation.

        Returns:
            List of waypoints.
        """
        db = self._get_db()
        stmt = select(Waypoint).where(Waypoint.region_id == region_id)
        
        if accommodation_only:
            # Only get waypoints that can serve as overnight stops
            stmt = stmt.where(Waypoint.has_accommodation == True)  # noqa: E712
        
        return list(db.execute(stmt).scalars().all())

    def get_candidate_pairs(
        self, waypoints: list[Waypoint]
    ) -> list[tuple[Waypoint, Waypoint, float]]:
        """Get pairs of waypoints that are potential connections.

        Uses straight-line distance filtering to avoid unnecessary API calls.

        Args:
            waypoints: List of waypoints to consider.

        Returns:
            List of (waypoint1, waypoint2, straight_line_distance) tuples.
        """
        max_distance = self._settings.max_straight_line_distance_km
        candidates = []

        for wp1, wp2 in combinations(waypoints, 2):
            distance = calculate_straight_line_distance_km(
                wp1.longitude, wp1.latitude, wp2.longitude, wp2.latitude
            )
            # Only consider pairs within max straight-line distance
            # Actual hiking distance will be longer, so this is a pre-filter
            if distance <= max_distance:
                candidates.append((wp1, wp2, distance))

        logger.info(
            f"Found {len(candidates)} candidate pairs within {max_distance}km "
            f"straight-line distance"
        )
        return candidates

    def connection_exists(
        self, from_id: uuid.UUID, to_id: uuid.UUID
    ) -> bool:
        """Check if a connection already exists between two waypoints.

        Args:
            from_id: Source waypoint ID.
            to_id: Destination waypoint ID.

        Returns:
            True if connection exists.
        """
        db = self._get_db()
        stmt = select(Connection).where(
            ((Connection.from_waypoint_id == from_id) & (Connection.to_waypoint_id == to_id)) |
            ((Connection.from_waypoint_id == to_id) & (Connection.to_waypoint_id == from_id))
        )
        return db.execute(stmt).first() is not None

    def _create_connection(
        self,
        from_wp: Waypoint,
        to_wp: Waypoint,
        route: RouteResult,
        straight_line_km: float,
    ) -> Connection:
        """Create a Connection model from route data.

        Args:
            from_wp: Source waypoint.
            to_wp: Destination waypoint.
            route: Route result from ORS.
            straight_line_km: Straight-line distance.

        Returns:
            Connection model instance.
        """
        # Check if route is within feasible distance constraints
        min_dist = self._settings.min_daily_distance_km
        max_dist = self._settings.max_daily_distance_km
        is_feasible = min_dist <= route.distance_km <= max_dist

        # Convert geometry to LineString
        route_geometry = None
        if route.geometry:
            route_geometry = from_shape(
                LineString(route.geometry), srid=4326
            )

        return Connection(
            id=uuid.uuid4(),
            from_waypoint_id=from_wp.id,
            to_waypoint_id=to_wp.id,
            distance_km=route.distance_km,
            duration_minutes=route.duration_minutes,
            elevation_gain_m=route.elevation_gain_m,
            elevation_loss_m=route.elevation_loss_m,
            route_geometry=route_geometry,
            route_metadata=route.metadata,
            is_feasible=is_feasible,
            straight_line_distance_km=straight_line_km,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    async def build_graph(
        self,
        region_name: str,
        skip_existing: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Build the feasibility graph for a region.

        Args:
            region_name: Name of the region.
            skip_existing: Skip pairs that already have connections.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dictionary with build statistics.
        """
        logger.info(f"Building feasibility graph for region: {region_name}")

        # Get region and waypoints
        region = self.get_region(region_name)
        if not region:
            raise ValueError(f"Region not found: {region_name}")

        waypoints = self.get_region_waypoints(region.id)
        if not waypoints:
            raise ValueError(f"No waypoints found for region: {region_name}")

        logger.info(f"Found {len(waypoints)} waypoints in {region_name}")

        # Get candidate pairs
        candidates = self.get_candidate_pairs(waypoints)

        # Filter out existing connections if requested
        if skip_existing:
            new_candidates = []
            for wp1, wp2, dist in candidates:
                if not self.connection_exists(wp1.id, wp2.id):
                    new_candidates.append((wp1, wp2, dist))
            logger.info(
                f"Filtered to {len(new_candidates)} new pairs "
                f"(skipped {len(candidates) - len(new_candidates)} existing)"
            )
            candidates = new_candidates

        if not candidates:
            return {
                "region": region_name,
                "total_waypoints": len(waypoints),
                "total_candidates": 0,
                "connections_created": 0,
                "feasible_connections": 0,
                "failed_routes": 0,
            }

        # Process candidates with ORS
        db = self._get_db()
        connections_created = 0
        feasible_count = 0
        failed_count = 0

        async with OpenRouteServiceClient() as ors_client:
            total = len(candidates)
            for i, (wp1, wp2, straight_line_km) in enumerate(candidates):
                if progress_callback:
                    progress_callback(i + 1, total, wp1.name, wp2.name)
                else:
                    logger.info(
                        f"Processing {i + 1}/{total}: {wp1.name} -> {wp2.name}"
                    )

                # Get hiking route
                route = await ors_client.get_hiking_route(
                    start_lon=wp1.longitude,
                    start_lat=wp1.latitude,
                    end_lon=wp2.longitude,
                    end_lat=wp2.latitude,
                )

                if route is None:
                    failed_count += 1
                    continue

                # Create connection
                connection = self._create_connection(
                    wp1, wp2, route, straight_line_km
                )
                db.add(connection)

                connections_created += 1
                if connection.is_feasible:
                    feasible_count += 1

                # Commit in batches
                if connections_created % 10 == 0:
                    db.commit()

        # Final commit
        db.commit()

        stats = {
            "region": region_name,
            "total_waypoints": len(waypoints),
            "total_candidates": len(candidates),
            "connections_created": connections_created,
            "feasible_connections": feasible_count,
            "failed_routes": failed_count,
        }

        logger.info(f"Graph building complete: {stats}")
        return stats

    def get_graph_stats(self, region_name: str) -> Optional[dict]:
        """Get statistics about the feasibility graph for a region.

        Args:
            region_name: Name of the region.

        Returns:
            Dictionary with graph statistics or None if region not found.
        """
        region = self.get_region(region_name)
        if not region:
            return None

        db = self._get_db()

        # Count waypoints
        wp_count = db.execute(
            select(func.count(Waypoint.id)).where(Waypoint.region_id == region.id)
        ).scalar()

        # Get all waypoint IDs for this region
        wp_ids = db.execute(
            select(Waypoint.id).where(Waypoint.region_id == region.id)
        ).scalars().all()

        if not wp_ids:
            return {
                "region": region_name,
                "total_waypoints": 0,
                "total_connections": 0,
                "feasible_connections": 0,
                "avg_distance_km": 0,
                "avg_duration_min": 0,
            }

        # Count connections
        conn_stmt = select(Connection).where(
            Connection.from_waypoint_id.in_(wp_ids)
        )
        connections = list(db.execute(conn_stmt).scalars().all())

        feasible = [c for c in connections if c.is_feasible]

        avg_distance = (
            sum(c.distance_km for c in connections) / len(connections)
            if connections
            else 0
        )
        avg_duration = (
            sum(c.duration_minutes for c in connections) / len(connections)
            if connections
            else 0
        )

        return {
            "region": region_name,
            "total_waypoints": wp_count,
            "total_connections": len(connections),
            "feasible_connections": len(feasible),
            "avg_distance_km": round(avg_distance, 2),
            "avg_duration_min": round(avg_duration, 0),
        }


async def build_graph(region_name: str, skip_existing: bool = True) -> dict:
    """Convenience function to build a feasibility graph.

    Args:
        region_name: Name of the region.
        skip_existing: Skip pairs with existing connections.

    Returns:
        Build statistics.
    """
    builder = GraphBuilder()
    try:
        return await builder.build_graph(region_name, skip_existing)
    finally:
        builder.close()


async def main():
    """Test the graph builder."""
    print("Building feasibility graph for Cornwall...")
    stats = await build_graph("cornwall")
    print(f"Build complete: {stats}")


if __name__ == "__main__":
    asyncio.run(main())

