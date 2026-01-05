"""On-the-fly itinerary generator for regions without precomputed graphs.

Uses beam search with lazy route evaluation to generate itineraries
without requiring a pre-built connection graph.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from geoalchemy2.shape import from_shape
from shapely.geometry import LineString
from sqlalchemy import select
from sqlalchemy.orm import Session

from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Region, Waypoint, WaypointType
from trail_pal.services.ors_client import (
    OpenRouteServiceClient,
    RouteResult,
    calculate_straight_line_distance_km,
)
from trail_pal.algorithm.itinerary_generator import (
    DayRoute,
    Itinerary,
    ItineraryOptions,
    SurfaceStats,
)

logger = logging.getLogger(__name__)


@dataclass
class CandidateEdge:
    """A candidate connection between two waypoints."""

    from_waypoint: Waypoint
    to_waypoint: Waypoint
    straight_line_km: float
    # Filled in lazily when route is fetched
    route: Optional[RouteResult] = None
    is_fetched: bool = False


@dataclass
class BeamPath:
    """A partial path being explored in beam search."""

    waypoints: list[Waypoint]
    edges: list[CandidateEdge]
    score: float = 0.0

    @property
    def last_waypoint(self) -> Waypoint:
        return self.waypoints[-1]

    def visited_ids(self) -> set[uuid.UUID]:
        return {wp.id for wp in self.waypoints}


class OnTheFlyGenerator:
    """Generator for multi-day hiking itineraries using on-demand routing.

    Uses beam search with lazy route evaluation to generate itineraries
    without requiring a pre-built connection graph. Routes are fetched
    from OpenRouteService on demand.
    """

    # Beam search parameters
    BEAM_WIDTH = 8  # Number of candidates to keep per expansion
    MAX_ORS_CALLS = 120  # Maximum ORS API calls per generation request
    
    # Distance constraints (same as precomputed)
    MIN_DAILY_DISTANCE_KM = 10.0
    MAX_DAILY_DISTANCE_KM = 20.0
    
    # Straight-line distance filter
    # Hiking routes are typically 1.2-1.5x straight-line distance
    MIN_STRAIGHT_LINE_KM = 6.0  # ~10km hiking minimum
    MAX_STRAIGHT_LINE_KM = 14.0  # ~20km hiking maximum

    def __init__(self, db: Optional[Session] = None, persist_routes: bool = True):
        """Initialize the generator.

        Args:
            db: SQLAlchemy session. If not provided, creates a new one.
            persist_routes: If True, persist fetched routes to the database
                for future use (opportunistic caching).
        """
        self._db = db
        self._owns_db = db is None
        self._settings = get_settings()
        self._waypoints: dict[uuid.UUID, Waypoint] = {}
        self._route_cache: dict[tuple[uuid.UUID, uuid.UUID], RouteResult] = {}
        # Cache for connection IDs from persisted routes
        self._connection_ids: dict[tuple[uuid.UUID, uuid.UUID], uuid.UUID] = {}
        self._ors_calls = 0
        self._persist_routes = persist_routes
        self._routes_persisted = 0

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

    def _get_existing_connection(
        self, from_id: uuid.UUID, to_id: uuid.UUID
    ) -> Optional[Connection]:
        """Check if a connection already exists in the database.

        Args:
            from_id: Source waypoint ID.
            to_id: Destination waypoint ID.

        Returns:
            Connection if exists, None otherwise.
        """
        db = self._get_db()
        stmt = select(Connection).where(
            ((Connection.from_waypoint_id == from_id) & (Connection.to_waypoint_id == to_id)) |
            ((Connection.from_waypoint_id == to_id) & (Connection.to_waypoint_id == from_id))
        )
        return db.execute(stmt).scalar_one_or_none()

    def _persist_route(
        self,
        from_wp: Waypoint,
        to_wp: Waypoint,
        route: RouteResult,
        straight_line_km: float,
    ) -> Optional[uuid.UUID]:
        """Persist a route to the database as a Connection.

        Args:
            from_wp: Source waypoint.
            to_wp: Destination waypoint.
            route: Route result from ORS.
            straight_line_km: Straight-line distance.

        Returns:
            Connection ID if persisted, None if already exists or failed.
        """
        if not self._persist_routes:
            return None

        db = self._get_db()

        # Check if connection already exists
        existing = self._get_existing_connection(from_wp.id, to_wp.id)
        if existing:
            self._connection_ids[(from_wp.id, to_wp.id)] = existing.id
            self._connection_ids[(to_wp.id, from_wp.id)] = existing.id
            return existing.id

        # Check if route is within feasible distance constraints
        is_feasible = self.MIN_DAILY_DISTANCE_KM <= route.distance_km <= self.MAX_DAILY_DISTANCE_KM

        # Convert geometry to LineString
        route_geometry = None
        if route.geometry and len(route.geometry) >= 2:
            try:
                route_geometry = from_shape(
                    LineString(route.geometry), srid=4326
                )
            except Exception as e:
                logger.warning(f"Failed to create LineString from route geometry: {e}")
                route_geometry = None

        try:
            connection = Connection(
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
            db.add(connection)
            db.commit()

            self._connection_ids[(from_wp.id, to_wp.id)] = connection.id
            self._connection_ids[(to_wp.id, from_wp.id)] = connection.id
            self._routes_persisted += 1

            logger.debug(
                f"Persisted route: {from_wp.name} -> {to_wp.name} "
                f"(id={connection.id})"
            )
            return connection.id

        except Exception as e:
            logger.warning(f"Failed to persist route: {e}")
            db.rollback()
            return None

    def _load_waypoints(self, region_name: str) -> list[Waypoint]:
        """Load waypoints for a region from the database.

        Args:
            region_name: Name of the region.

        Returns:
            List of waypoints.
        """
        db = self._get_db()

        # Get region
        stmt = select(Region).where(Region.name == region_name.lower())
        region = db.execute(stmt).scalar_one_or_none()
        if not region:
            raise ValueError(f"Region not found: {region_name}")

        # Load waypoints
        wp_stmt = select(Waypoint).where(Waypoint.region_id == region.id)
        waypoints = list(db.execute(wp_stmt).scalars().all())
        self._waypoints = {wp.id: wp for wp in waypoints}

        if not waypoints:
            raise ValueError(f"No waypoints found for region: {region_name}")

        logger.info(f"Loaded {len(waypoints)} waypoints for {region_name}")
        return waypoints

    def _is_transport_accessible(self, waypoint_type: str) -> bool:
        """Check if a waypoint type is suitable for start/end positions."""
        return waypoint_type in [
            WaypointType.TRAIN_STATION,
            WaypointType.TOWN,
        ]

    def _get_candidate_neighbors(
        self, waypoint: Waypoint, visited: set[uuid.UUID]
    ) -> list[CandidateEdge]:
        """Get candidate next waypoints based on straight-line distance.

        Args:
            waypoint: Current waypoint.
            visited: Set of already-visited waypoint IDs.

        Returns:
            List of candidate edges sorted by distance.
        """
        candidates = []

        for wp in self._waypoints.values():
            if wp.id in visited:
                continue

            distance = calculate_straight_line_distance_km(
                waypoint.longitude, waypoint.latitude,
                wp.longitude, wp.latitude
            )

            # Filter by straight-line distance constraints
            if self.MIN_STRAIGHT_LINE_KM <= distance <= self.MAX_STRAIGHT_LINE_KM:
                candidates.append(CandidateEdge(
                    from_waypoint=waypoint,
                    to_waypoint=wp,
                    straight_line_km=distance,
                ))

        # Sort by distance (prefer moderate distances in the middle of range)
        target_distance = (self.MIN_STRAIGHT_LINE_KM + self.MAX_STRAIGHT_LINE_KM) / 2
        candidates.sort(key=lambda c: abs(c.straight_line_km - target_distance))

        return candidates

    def _score_candidate(
        self,
        edge: CandidateEdge,
        day_number: int,
        num_days: int,
        prefer_accommodation: bool,
    ) -> float:
        """Score a candidate edge for beam search selection.

        Args:
            edge: Candidate edge to score.
            day_number: Current day (1-indexed).
            num_days: Total number of days.
            prefer_accommodation: Whether to prefer accommodation waypoints.

        Returns:
            Score value (higher is better).
        """
        score = 0.0
        wp = edge.to_waypoint

        # Waypoint type scoring
        is_last_day = day_number == num_days
        is_intermediate = not is_last_day

        if is_last_day:
            # Last day should end at transport-accessible location
            if self._is_transport_accessible(wp.waypoint_type):
                score += 30
            elif wp.waypoint_type == WaypointType.VILLAGE:
                score += 20
        else:
            # Intermediate days should end at accommodation
            if wp.waypoint_type == WaypointType.CAMPSITE:
                score += 25
            elif wp.waypoint_type == WaypointType.HOSTEL:
                score += 30
            elif wp.waypoint_type == WaypointType.GUEST_HOUSE:
                score += 28
            elif wp.waypoint_type == WaypointType.HOTEL:
                score += 20
            elif wp.waypoint_type == WaypointType.VILLAGE:
                score += 15
            elif wp.waypoint_type == WaypointType.TOWN:
                score += 18

        # Amenity bonuses
        if wp.has_accommodation and prefer_accommodation and is_intermediate:
            score += 15
        if wp.has_food:
            score += 5
        if wp.has_water:
            score += 3

        # Scenic bonus for viewpoints/peaks (not as end points)
        if wp.waypoint_type == WaypointType.VIEWPOINT:
            score += 8
        elif wp.waypoint_type == WaypointType.PEAK:
            score += 10

        # Distance scoring - prefer moderate distances
        target_km = (self.MIN_STRAIGHT_LINE_KM + self.MAX_STRAIGHT_LINE_KM) / 2
        distance_penalty = abs(edge.straight_line_km - target_km) * 2
        score -= distance_penalty

        # Add small random factor for variety
        score += random.uniform(0, 5)

        return score

    async def _fetch_route(
        self,
        edge: CandidateEdge,
        ors_client: OpenRouteServiceClient,
    ) -> Optional[RouteResult]:
        """Fetch route from ORS, using cache or database if available.

        Args:
            edge: Edge to fetch route for.
            ors_client: ORS client instance.

        Returns:
            RouteResult or None if fetch failed.
        """
        cache_key = (edge.from_waypoint.id, edge.to_waypoint.id)
        reverse_key = (edge.to_waypoint.id, edge.from_waypoint.id)

        # Check in-memory cache first
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]
        if reverse_key in self._route_cache:
            return self._route_cache[reverse_key]

        # Check database for existing connection
        existing_conn = self._get_existing_connection(
            edge.from_waypoint.id, edge.to_waypoint.id
        )
        if existing_conn:
            # Convert Connection to RouteResult
            route = RouteResult(
                distance_km=existing_conn.distance_km,
                duration_minutes=existing_conn.duration_minutes,
                elevation_gain_m=existing_conn.elevation_gain_m,
                elevation_loss_m=existing_conn.elevation_loss_m,
                geometry=[],  # Geometry not needed for itinerary generation
                metadata=existing_conn.route_metadata or {},
                surface_breakdown=None,
            )
            self._route_cache[cache_key] = route
            self._connection_ids[cache_key] = existing_conn.id
            self._connection_ids[reverse_key] = existing_conn.id
            logger.debug(
                f"Found existing connection: {edge.from_waypoint.name} -> "
                f"{edge.to_waypoint.name} ({route.distance_km:.1f}km)"
            )
            return route

        # Check if we've exceeded ORS call budget
        if self._ors_calls >= self.MAX_ORS_CALLS:
            logger.warning(f"ORS call budget exhausted ({self.MAX_ORS_CALLS} calls)")
            return None

        # Fetch from ORS
        self._ors_calls += 1
        route = await ors_client.get_hiking_route(
            start_lon=edge.from_waypoint.longitude,
            start_lat=edge.from_waypoint.latitude,
            end_lon=edge.to_waypoint.longitude,
            end_lat=edge.to_waypoint.latitude,
        )

        if route:
            self._route_cache[cache_key] = route
            logger.debug(
                f"Fetched route: {edge.from_waypoint.name} -> {edge.to_waypoint.name} "
                f"({route.distance_km:.1f}km)"
            )

            # Persist route to database for future use
            self._persist_route(
                edge.from_waypoint,
                edge.to_waypoint,
                route,
                edge.straight_line_km,
            )

        return route

    def _is_feasible_route(self, route: RouteResult) -> bool:
        """Check if a route meets distance constraints.

        Args:
            route: Route to check.

        Returns:
            True if route is within feasible distance range.
        """
        return self.MIN_DAILY_DISTANCE_KM <= route.distance_km <= self.MAX_DAILY_DISTANCE_KM

    async def _expand_beam(
        self,
        beam: BeamPath,
        day_number: int,
        num_days: int,
        options: ItineraryOptions,
        ors_client: OpenRouteServiceClient,
    ) -> list[BeamPath]:
        """Expand a beam path by one day.

        Args:
            beam: Current beam path.
            day_number: Day being added (1-indexed).
            num_days: Total number of days.
            options: Generation options.
            ors_client: ORS client for route fetching.

        Returns:
            List of expanded beam paths.
        """
        candidates = self._get_candidate_neighbors(
            beam.last_waypoint,
            beam.visited_ids()
        )

        if not candidates:
            return []

        # Score candidates
        scored_candidates = []
        for edge in candidates:
            score = self._score_candidate(
                edge, day_number, num_days, options.prefer_accommodation
            )
            scored_candidates.append((score, edge))

        # Sort by score and take top candidates
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:self.BEAM_WIDTH + 5]  # Slight over-sample

        # Fetch routes for top candidates and filter by feasibility
        expanded = []
        for score, edge in top_candidates:
            route = await self._fetch_route(edge, ors_client)
            if route is None:
                continue

            if not self._is_feasible_route(route):
                continue

            # Create expanded path
            edge.route = route
            edge.is_fetched = True

            new_path = BeamPath(
                waypoints=beam.waypoints + [edge.to_waypoint],
                edges=beam.edges + [edge],
                score=beam.score + score,
            )
            expanded.append(new_path)

            # Stop if we have enough feasible paths
            if len(expanded) >= self.BEAM_WIDTH:
                break

        return expanded

    def _score_itinerary(
        self, path: BeamPath, options: ItineraryOptions
    ) -> float:
        """Calculate final score for a complete itinerary.

        Args:
            path: Complete beam path.
            options: Generation options.

        Returns:
            Score value (higher is better).
        """
        score = path.score

        # Transport accessibility bonus for start/end
        if self._is_transport_accessible(path.waypoints[0].waypoint_type):
            score += 15
        if self._is_transport_accessible(path.waypoints[-1].waypoint_type):
            score += 15

        # Distance balance bonus
        distances = [e.route.distance_km for e in path.edges if e.route]
        if distances:
            avg_distance = sum(distances) / len(distances)
            variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
            score += max(0, 10 - variance)

        # Elevation bonus (scenic value)
        total_gain = sum(
            e.route.elevation_gain_m or 0 for e in path.edges if e.route
        )
        score += min(total_gain / 100, 10)

        # Variety bonus - different waypoint types
        types_seen = {wp.waypoint_type for wp in path.waypoints}
        score += len(types_seen) * 2

        return score

    def _path_to_itinerary(self, path: BeamPath, region_name: str) -> Itinerary:
        """Convert a beam path to an Itinerary object.

        Args:
            path: Completed beam path.
            region_name: Name of the region.

        Returns:
            Itinerary object.
        """
        days = []
        for i, edge in enumerate(path.edges):
            if not edge.route:
                continue

            surface_stats = None
            if edge.route.surface_breakdown:
                surface_stats = SurfaceStats(
                    surfaces=edge.route.surface_breakdown.surfaces,
                    waytypes=edge.route.surface_breakdown.waytypes,
                    total_distance_km=edge.route.surface_breakdown.total_distance_km,
                )

            # Get connection ID if route was persisted
            connection_id = self._connection_ids.get(
                (edge.from_waypoint.id, edge.to_waypoint.id)
            )

            day = DayRoute(
                day_number=i + 1,
                start_waypoint=edge.from_waypoint,
                end_waypoint=edge.to_waypoint,
                distance_km=edge.route.distance_km,
                duration_minutes=edge.route.duration_minutes,
                elevation_gain_m=edge.route.elevation_gain_m,
                elevation_loss_m=edge.route.elevation_loss_m,
                connection_id=connection_id,
                surface_stats=surface_stats,
            )
            days.append(day)

        itinerary = Itinerary(
            id=uuid.uuid4(),
            region_name=region_name,
            days=days,
        )
        itinerary.score = path.score
        return itinerary

    async def generate(
        self, region_name: str, options: Optional[ItineraryOptions] = None
    ) -> list[Itinerary]:
        """Generate hiking itineraries for a region using on-the-fly routing.

        Args:
            region_name: Name of the region.
            options: Generation options.

        Returns:
            List of itineraries, sorted by score (best first).
        """
        if options is None:
            options = ItineraryOptions()

        logger.info(
            f"[On-the-fly] Generating {options.num_days}-day itineraries for {region_name}"
        )

        # Reset state
        self._route_cache.clear()
        self._ors_calls = 0

        # Load waypoints
        waypoints = self._load_waypoints(region_name)

        # Find starting points
        if options.start_waypoint_name:
            name_lower = options.start_waypoint_name.lower()
            start_waypoints = [
                wp for wp in waypoints
                if name_lower in wp.name.lower()
                and (options.allow_any_start or self._is_transport_accessible(wp.waypoint_type))
            ]
            if not start_waypoints:
                raise ValueError(
                    f"No suitable waypoint found matching name: {options.start_waypoint_name}"
                )
        elif options.allow_any_start:
            start_waypoints = waypoints
        else:
            start_waypoints = [
                wp for wp in waypoints
                if self._is_transport_accessible(wp.waypoint_type)
            ]

        if not start_waypoints:
            logger.warning("No suitable starting points found")
            return []

        # Limit starting points for performance
        max_starts = min(8, len(start_waypoints))
        if len(start_waypoints) > max_starts:
            start_waypoints = random.sample(start_waypoints, max_starts)

        logger.info(f"Starting beam search from {len(start_waypoints)} waypoints")

        # Initialize beams
        beams = [
            BeamPath(waypoints=[wp], edges=[], score=0.0)
            for wp in start_waypoints
        ]

        # Beam search expansion
        async with OpenRouteServiceClient() as ors_client:
            for day in range(1, options.num_days + 1):
                logger.info(f"Expanding day {day}, {len(beams)} active beams")

                new_beams = []
                for beam in beams:
                    expanded = await self._expand_beam(
                        beam, day, options.num_days, options, ors_client
                    )
                    new_beams.extend(expanded)

                if not new_beams:
                    logger.warning(f"No valid expansions at day {day}")
                    break

                # Keep top beams
                new_beams.sort(key=lambda b: b.score, reverse=True)
                beams = new_beams[:self.BEAM_WIDTH]

                logger.info(f"Day {day} complete, {len(beams)} beams, {self._ors_calls} ORS calls")

        logger.info(f"Beam search complete. Found {len(beams)} candidate paths")

        # Filter paths that end at transport-accessible waypoints
        if not options.allow_any_start:
            beams = [
                b for b in beams
                if self._is_transport_accessible(b.last_waypoint.waypoint_type)
            ]
            logger.info(f"Filtered to {len(beams)} paths with transport-accessible endpoints")

        # Score and convert to itineraries
        itineraries = []
        for beam in beams:
            if len(beam.edges) != options.num_days:
                continue  # Incomplete path

            beam.score = self._score_itinerary(beam, options)
            itinerary = self._path_to_itinerary(beam, region_name)
            itineraries.append(itinerary)

        # Sort by score and limit results
        itineraries.sort(key=lambda it: it.score, reverse=True)

        if options.randomize and len(itineraries) > options.max_results:
            # Randomly sample from top results
            top_pool = itineraries[:options.max_results * 3]
            itineraries = random.sample(
                top_pool, min(options.max_results, len(top_pool))
            )
        else:
            itineraries = itineraries[:options.max_results]

        logger.info(
            f"Returning {len(itineraries)} itineraries "
            f"({self._ors_calls} ORS calls, {self._routes_persisted} routes persisted)"
        )

        return itineraries


async def generate_itineraries_onthefly(
    region_name: str, options: Optional[ItineraryOptions] = None
) -> list[Itinerary]:
    """Convenience function to generate itineraries using on-the-fly routing.

    Args:
        region_name: Name of the region.
        options: Generation options.

    Returns:
        List of itineraries.
    """
    generator = OnTheFlyGenerator()
    try:
        return await generator.generate(region_name, options)
    finally:
        generator.close()


async def main():
    """Test the on-the-fly itinerary generator."""
    print("Generating itineraries for Lake District (on-the-fly)...")

    options = ItineraryOptions(
        num_days=3,
        prefer_accommodation=True,
        max_results=5,
    )

    generator = OnTheFlyGenerator()
    try:
        itineraries = await generator.generate("lake_district", options)

        if not itineraries:
            print("No itineraries found. Make sure to seed waypoints first.")
            return

        print(f"\nFound {len(itineraries)} itineraries:\n")
        for i, itinerary in enumerate(itineraries, 1):
            print(f"--- Option {i} (Score: {itinerary.score:.1f}) ---")
            print(f"Total: {itinerary.total_distance_km:.1f} km")
            for day in itinerary.days:
                print(
                    f"  Day {day.day_number}: {day.start_waypoint.name} -> "
                    f"{day.end_waypoint.name} ({day.distance_km:.1f} km)"
                )
            print()
    finally:
        generator.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

