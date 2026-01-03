"""Overlap analyzer for computing geometric overlaps between route connections.

This service pre-computes overlap distances between connection pairs that share
a waypoint, enabling fast overlap filtering during itinerary generation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from itertools import combinations
from typing import Optional

from geoalchemy2.shape import to_shape
from shapely.geometry import LineString
from shapely.ops import nearest_points
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, ConnectionOverlap, Region, Waypoint

logger = logging.getLogger(__name__)

# Buffer distance in degrees (approximately 20 meters at mid-latitudes)
# Used to detect near-overlaps where paths are parallel but not exactly coincident
BUFFER_DEGREES = 0.0002  # ~20m


class OverlapAnalyzer:
    """Service for analyzing and storing route geometry overlaps."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize the analyzer.

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

    def _calculate_overlap_km(
        self,
        geom_a: LineString,
        geom_b: LineString,
    ) -> float:
        """Calculate the overlap distance between two LineString geometries.

        Uses a buffer-and-intersect approach to find shared trail segments.
        The overlap is measured as the length of the intersection area
        converted to an approximate distance.

        Args:
            geom_a: First route geometry.
            geom_b: Second route geometry.

        Returns:
            Overlap distance in kilometers.
        """
        try:
            # Buffer both geometries slightly to catch near-parallel paths
            buffer_a = geom_a.buffer(BUFFER_DEGREES)
            buffer_b = geom_b.buffer(BUFFER_DEGREES)

            # Find intersection of buffered geometries
            intersection = buffer_a.intersection(buffer_b)

            if intersection.is_empty:
                return 0.0

            # For the intersection area, we want to estimate the "length" of overlap
            # We'll use the skeleton/centerline approach: find what portion of each
            # original line falls within the intersection
            
            # Calculate how much of line A is within the buffered intersection
            a_in_intersection = geom_a.intersection(buffer_b)
            b_in_intersection = geom_b.intersection(buffer_a)

            # Use the average of both overlapping lengths
            overlap_length_deg = 0.0
            
            if not a_in_intersection.is_empty:
                overlap_length_deg = max(overlap_length_deg, a_in_intersection.length)
            if not b_in_intersection.is_empty:
                overlap_length_deg = max(overlap_length_deg, b_in_intersection.length)

            # Convert degrees to kilometers (approximate)
            # At 50° latitude: 1 degree ≈ 111 km latitude, ~71 km longitude
            # Use average of ~90 km per degree as rough approximation
            overlap_km = overlap_length_deg * 90.0

            return overlap_km

        except Exception as e:
            logger.warning(f"Error calculating overlap: {e}")
            return 0.0

    def _get_connections_at_waypoint(
        self,
        waypoint_id: uuid.UUID,
    ) -> list[Connection]:
        """Get all connections that have this waypoint as an endpoint.

        Args:
            waypoint_id: The waypoint ID.

        Returns:
            List of connections.
        """
        db = self._get_db()
        stmt = select(Connection).where(
            (Connection.from_waypoint_id == waypoint_id) |
            (Connection.to_waypoint_id == waypoint_id),
            Connection.is_feasible == True,  # noqa: E712
            Connection.route_geometry.isnot(None),
        )
        return list(db.execute(stmt).scalars().all())

    def build_overlaps(
        self,
        region_name: str,
        clear_existing: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """Build overlap data for all connection pairs in a region.

        For each waypoint, finds all connections that share it as an endpoint,
        then computes pairwise overlaps between those connections.

        Args:
            region_name: Name of the region.
            clear_existing: If True, delete existing overlap data first.
            progress_callback: Optional callback(current, total, waypoint_name).

        Returns:
            Dictionary with build statistics.
        """
        db = self._get_db()

        # Get region
        stmt = select(Region).where(Region.name == region_name.lower())
        region = db.execute(stmt).scalar_one_or_none()
        if not region:
            raise ValueError(f"Region not found: {region_name}")

        # Get all waypoints in region
        wp_stmt = select(Waypoint).where(Waypoint.region_id == region.id)
        waypoints = list(db.execute(wp_stmt).scalars().all())

        if not waypoints:
            raise ValueError(f"No waypoints found for region: {region_name}")

        logger.info(f"Building overlaps for {len(waypoints)} waypoints in {region_name}")

        # Clear existing overlaps if requested
        if clear_existing:
            # Get all connection IDs for this region's waypoints
            wp_ids = [wp.id for wp in waypoints]
            conn_stmt = select(Connection.id).where(
                Connection.from_waypoint_id.in_(wp_ids)
            )
            conn_ids = list(db.execute(conn_stmt).scalars().all())

            if conn_ids:
                delete_stmt = delete(ConnectionOverlap).where(
                    ConnectionOverlap.connection_a_id.in_(conn_ids) |
                    ConnectionOverlap.connection_b_id.in_(conn_ids)
                )
                db.execute(delete_stmt)
                db.commit()
                logger.info("Cleared existing overlap data")

        # Process each waypoint
        overlaps_created = 0
        overlaps_with_data = 0
        pairs_processed = 0

        total_waypoints = len(waypoints)
        for i, waypoint in enumerate(waypoints):
            if progress_callback:
                progress_callback(i + 1, total_waypoints, waypoint.name)
            else:
                logger.debug(f"Processing waypoint {i + 1}/{total_waypoints}: {waypoint.name}")

            # Get all connections at this waypoint
            connections = self._get_connections_at_waypoint(waypoint.id)

            if len(connections) < 2:
                continue

            # Process all pairs of connections at this waypoint
            for conn_a, conn_b in combinations(connections, 2):
                pairs_processed += 1

                # Check if overlap already exists (from processing another waypoint)
                existing = db.execute(
                    select(ConnectionOverlap).where(
                        ((ConnectionOverlap.connection_a_id == conn_a.id) &
                         (ConnectionOverlap.connection_b_id == conn_b.id)) |
                        ((ConnectionOverlap.connection_a_id == conn_b.id) &
                         (ConnectionOverlap.connection_b_id == conn_a.id))
                    )
                ).scalar_one_or_none()

                if existing:
                    continue

                # Extract geometries
                try:
                    geom_a = to_shape(conn_a.route_geometry)
                    geom_b = to_shape(conn_b.route_geometry)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract geometry for connection pair: {e}"
                    )
                    continue

                # Calculate overlap
                overlap_km = self._calculate_overlap_km(geom_a, geom_b)

                # Only store if there's meaningful overlap (> 100m)
                if overlap_km >= 0.1:
                    # Ensure consistent ordering (smaller UUID first)
                    if str(conn_a.id) < str(conn_b.id):
                        a_id, b_id = conn_a.id, conn_b.id
                    else:
                        a_id, b_id = conn_b.id, conn_a.id

                    overlap = ConnectionOverlap(
                        id=uuid.uuid4(),
                        connection_a_id=a_id,
                        connection_b_id=b_id,
                        shared_waypoint_id=waypoint.id,
                        overlap_km=overlap_km,
                        created_at=datetime.utcnow(),
                    )
                    db.add(overlap)
                    overlaps_created += 1

                    if overlap_km > 0:
                        overlaps_with_data += 1

            # Commit in batches
            if overlaps_created > 0 and overlaps_created % 100 == 0:
                db.commit()

        # Final commit
        db.commit()

        stats = {
            "region": region_name,
            "waypoints_processed": total_waypoints,
            "pairs_analyzed": pairs_processed,
            "overlaps_stored": overlaps_created,
            "overlaps_with_significant_data": overlaps_with_data,
        }

        logger.info(f"Overlap analysis complete: {stats}")
        return stats

    def get_overlap(
        self,
        connection_a_id: uuid.UUID,
        connection_b_id: uuid.UUID,
    ) -> float:
        """Get the pre-computed overlap between two connections.

        Args:
            connection_a_id: First connection ID.
            connection_b_id: Second connection ID.

        Returns:
            Overlap distance in km, or 0 if no overlap data exists.
        """
        db = self._get_db()

        stmt = select(ConnectionOverlap).where(
            ((ConnectionOverlap.connection_a_id == connection_a_id) &
             (ConnectionOverlap.connection_b_id == connection_b_id)) |
            ((ConnectionOverlap.connection_a_id == connection_b_id) &
             (ConnectionOverlap.connection_b_id == connection_a_id))
        )
        overlap = db.execute(stmt).scalar_one_or_none()

        return overlap.overlap_km if overlap else 0.0

    def get_overlap_stats(self, region_name: str) -> Optional[dict]:
        """Get statistics about overlap data for a region.

        Args:
            region_name: Name of the region.

        Returns:
            Dictionary with overlap statistics or None if region not found.
        """
        db = self._get_db()

        # Get region
        stmt = select(Region).where(Region.name == region_name.lower())
        region = db.execute(stmt).scalar_one_or_none()
        if not region:
            return None

        # Get waypoint IDs
        wp_stmt = select(Waypoint.id).where(Waypoint.region_id == region.id)
        wp_ids = list(db.execute(wp_stmt).scalars().all())

        if not wp_ids:
            return {
                "region": region_name,
                "total_overlaps": 0,
                "overlaps_above_1km": 0,
                "overlaps_above_3km": 0,
                "max_overlap_km": 0,
                "avg_overlap_km": 0,
            }

        # Get connection IDs for this region
        conn_stmt = select(Connection.id).where(
            Connection.from_waypoint_id.in_(wp_ids)
        )
        conn_ids = list(db.execute(conn_stmt).scalars().all())

        # Get overlaps
        overlap_stmt = select(ConnectionOverlap).where(
            ConnectionOverlap.connection_a_id.in_(conn_ids)
        )
        overlaps = list(db.execute(overlap_stmt).scalars().all())

        if not overlaps:
            return {
                "region": region_name,
                "total_overlaps": 0,
                "overlaps_above_1km": 0,
                "overlaps_above_3km": 0,
                "max_overlap_km": 0,
                "avg_overlap_km": 0,
            }

        overlap_values = [o.overlap_km for o in overlaps]

        return {
            "region": region_name,
            "total_overlaps": len(overlaps),
            "overlaps_above_1km": sum(1 for v in overlap_values if v > 1.0),
            "overlaps_above_3km": sum(1 for v in overlap_values if v > 3.0),
            "max_overlap_km": round(max(overlap_values), 2),
            "avg_overlap_km": round(sum(overlap_values) / len(overlap_values), 2),
        }


def build_overlaps(region_name: str, clear_existing: bool = True) -> dict:
    """Convenience function to build overlap data for a region.

    Args:
        region_name: Name of the region.
        clear_existing: If True, delete existing overlap data first.

    Returns:
        Build statistics.
    """
    analyzer = OverlapAnalyzer()
    try:
        return analyzer.build_overlaps(region_name, clear_existing)
    finally:
        analyzer.close()

