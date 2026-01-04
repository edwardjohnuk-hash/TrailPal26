"""Itinerary generator algorithm for multi-day hiking routes."""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, ConnectionOverlap, Region, Waypoint, WaypointType

logger = logging.getLogger(__name__)

# Module-level cache for graph data to avoid reloading on every request
# This is the main performance optimization - loading 1.4M overlaps takes ~20s
_graph_cache: dict[str, tuple[nx.DiGraph, dict, dict, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minute cache TTL


@dataclass
class SurfaceStats:
    """Surface breakdown statistics for a route segment."""

    # surface type -> distance in km
    surfaces: dict[str, float] = field(default_factory=dict)
    # way type -> distance in km
    waytypes: dict[str, float] = field(default_factory=dict)
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

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["SurfaceStats"]:
        """Create from dictionary (e.g., from route_metadata)."""
        if not data:
            return None
        return cls(
            surfaces=data.get("surfaces", {}),
            waytypes=data.get("waytypes", {}),
            total_distance_km=data.get("total_distance_km", 0.0),
        )


@dataclass
class DayRoute:
    """A single day's hiking route."""

    day_number: int
    start_waypoint: Waypoint
    end_waypoint: Waypoint
    distance_km: float
    duration_minutes: int
    elevation_gain_m: Optional[float] = None
    elevation_loss_m: Optional[float] = None
    connection_id: Optional[uuid.UUID] = None
    surface_stats: Optional[SurfaceStats] = None


@dataclass
class Itinerary:
    """A complete multi-day hiking itinerary."""

    id: uuid.UUID
    region_name: str
    days: list[DayRoute]
    total_distance_km: float = 0.0
    total_duration_minutes: int = 0
    total_elevation_gain_m: float = 0.0
    score: float = 0.0

    def __post_init__(self):
        """Calculate totals after initialization."""
        self.total_distance_km = sum(d.distance_km for d in self.days)
        self.total_duration_minutes = sum(d.duration_minutes for d in self.days)
        self.total_elevation_gain_m = sum(
            d.elevation_gain_m or 0 for d in self.days
        )


@dataclass
class ItineraryOptions:
    """Options for itinerary generation."""

    num_days: int = 3
    start_waypoint_id: Optional[uuid.UUID] = None
    start_waypoint_name: Optional[str] = None
    prefer_accommodation: bool = True
    max_results: int = 10
    min_distance_km: float = 10.0
    max_distance_km: float = 20.0
    randomize: bool = False  # If True, randomly select routes instead of top-scored
    max_overlap_km: float = 3.0  # Max allowed overlap between consecutive days
    allow_any_start: bool = False  # If True, allow any waypoint type as start


class ItineraryGenerator:
    """Generator for multi-day hiking itineraries using graph traversal."""

    # Maximum paths to explore before early termination
    # This prevents excessive search times while still finding good routes
    MAX_PATHS_TO_EXPLORE = 50000

    def __init__(self, db: Optional[Session] = None):
        """Initialize the generator.

        Args:
            db: SQLAlchemy session. If not provided, creates a new one.
        """
        self._db = db
        self._owns_db = db is None
        self._settings = get_settings()
        self._graph: Optional[nx.DiGraph] = None
        self._waypoints: dict[uuid.UUID, Waypoint] = {}
        # Overlap lookup: (connection_a_id, connection_b_id) -> overlap_km
        # Stores both orderings for O(1) lookup
        self._overlaps: dict[tuple[uuid.UUID, uuid.UUID], float] = {}
        # Counter for early termination
        self._paths_found = 0

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

    def _load_graph(self, region_name: str) -> nx.DiGraph:
        """Load the feasibility graph from the database.

        Uses module-level caching to avoid reloading 1.4M+ overlap records
        on every request. Cache is invalidated after TTL expires.

        Args:
            region_name: Name of the region.

        Returns:
            NetworkX directed graph.
        """
        global _graph_cache

        cache_key = region_name.lower()
        now = time.time()

        # Check cache first
        if cache_key in _graph_cache:
            cached_graph, cached_waypoints, cached_overlaps, cached_time = _graph_cache[cache_key]
            if now - cached_time < _CACHE_TTL_SECONDS:
                logger.info(
                    f"Using cached graph for {region_name} "
                    f"(age: {now - cached_time:.0f}s)"
                )
                self._waypoints = cached_waypoints
                self._overlaps = cached_overlaps
                return cached_graph
            else:
                logger.info(f"Cache expired for {region_name}, reloading...")
                del _graph_cache[cache_key]

        # Cache miss - load from database
        load_start = time.time()
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

        # Create graph
        graph = nx.DiGraph()

        # Add nodes (waypoints)
        for wp in waypoints:
            graph.add_node(
                wp.id,
                name=wp.name,
                waypoint_type=wp.waypoint_type,
                has_accommodation=wp.has_accommodation,
                latitude=wp.latitude,
                longitude=wp.longitude,
            )

        # Load feasible connections
        wp_ids = list(self._waypoints.keys())
        conn_stmt = select(Connection).where(
            Connection.from_waypoint_id.in_(wp_ids),
            Connection.is_feasible == True,  # noqa: E712
        )
        connections = list(db.execute(conn_stmt).scalars().all())

        # Add edges (bidirectional for hiking)
        for conn in connections:
            # Extract surface breakdown from route_metadata if available
            surface_data = None
            if conn.route_metadata and "surface_breakdown" in conn.route_metadata:
                surface_data = conn.route_metadata["surface_breakdown"]

            edge_data = {
                "connection_id": conn.id,
                "distance_km": conn.distance_km,
                "duration_minutes": conn.duration_minutes,
                "elevation_gain_m": conn.elevation_gain_m,
                "elevation_loss_m": conn.elevation_loss_m,
                "surface_breakdown": surface_data,
            }
            # Add both directions
            graph.add_edge(conn.from_waypoint_id, conn.to_waypoint_id, **edge_data)
            # Reverse direction has swapped elevation (surface data is the same)
            reverse_data = edge_data.copy()
            reverse_data["elevation_gain_m"] = conn.elevation_loss_m
            reverse_data["elevation_loss_m"] = conn.elevation_gain_m
            graph.add_edge(conn.to_waypoint_id, conn.from_waypoint_id, **reverse_data)

        logger.info(
            f"Loaded graph with {graph.number_of_nodes()} nodes "
            f"and {graph.number_of_edges()} edges"
        )

        # Load pre-computed overlap data for fast filtering (optimized query)
        self._load_overlaps_fast(wp_ids)

        # Cache the loaded data
        _graph_cache[cache_key] = (graph, self._waypoints.copy(), self._overlaps.copy(), now)
        
        load_time = time.time() - load_start
        logger.info(f"Graph loaded and cached in {load_time:.2f}s")

        return graph

    def _load_overlaps(self, waypoint_ids: list[uuid.UUID]) -> None:
        """Load pre-computed overlap data into memory for fast lookup.

        Args:
            waypoint_ids: List of waypoint IDs in the region.
        """
        # Delegate to the faster implementation
        self._load_overlaps_fast(waypoint_ids)

    def _load_overlaps_fast(self, waypoint_ids: list[uuid.UUID]) -> None:
        """Load pre-computed overlap data using optimized query.

        This version fetches only the columns we need (not full ORM objects)
        to reduce memory and serialization overhead for 1.4M+ rows.

        Args:
            waypoint_ids: List of waypoint IDs in the region.
        """
        db = self._get_db()

        # Get all connection IDs for this region
        conn_stmt = select(Connection.id).where(
            Connection.from_waypoint_id.in_(waypoint_ids)
        )
        conn_ids = list(db.execute(conn_stmt).scalars().all())

        if not conn_ids:
            logger.info("No connections found, skipping overlap loading")
            return

        # Fetch only the columns we need (much faster than loading full ORM objects)
        overlap_stmt = select(
            ConnectionOverlap.connection_a_id,
            ConnectionOverlap.connection_b_id,
            ConnectionOverlap.overlap_km,
        ).where(
            ConnectionOverlap.connection_a_id.in_(conn_ids) |
            ConnectionOverlap.connection_b_id.in_(conn_ids)
        )
        
        result = db.execute(overlap_stmt).fetchall()

        # Build lookup dict with both orderings for O(1) access
        self._overlaps.clear()
        for conn_a, conn_b, overlap_km in result:
            self._overlaps[(conn_a, conn_b)] = overlap_km
            self._overlaps[(conn_b, conn_a)] = overlap_km

        logger.info(f"Loaded {len(result)} overlap records into memory")

    def _get_overlap(
        self, connection_a_id: Optional[uuid.UUID], connection_b_id: Optional[uuid.UUID]
    ) -> float:
        """Get the overlap between two connections.

        Args:
            connection_a_id: First connection ID.
            connection_b_id: Second connection ID.

        Returns:
            Overlap distance in km, or 0 if no overlap data exists.
        """
        if connection_a_id is None or connection_b_id is None:
            return 0.0
        return self._overlaps.get((connection_a_id, connection_b_id), 0.0)

    def _is_transport_accessible(self, waypoint_type: str) -> bool:
        """Check if a waypoint type is suitable for default start/end positions.

        By default, only train stations and towns are considered valid
        starting/ending points for itineraries.

        Args:
            waypoint_type: Waypoint type string.

        Returns:
            True if waypoint type is transport-accessible (train station or town).
        """
        return waypoint_type in [
            WaypointType.TRAIN_STATION,
            WaypointType.TOWN,
        ]

    def _score_waypoint(
        self, waypoint: Waypoint, prefer_accommodation: bool, is_start_or_end: bool = False
    ) -> float:
        """Calculate a quality score for a waypoint.

        Args:
            waypoint: Waypoint to score.
            prefer_accommodation: Whether to prioritize accommodation.
            is_start_or_end: Whether this waypoint is used as start/end position.

        Returns:
            Score value (higher is better).
        """
        score = 0.0

        # Transport-accessible types score higher for start/end positions
        if is_start_or_end:
            if waypoint.waypoint_type == WaypointType.TRAIN_STATION:
                score += 20  # Highest priority for train stations
            elif waypoint.waypoint_type == WaypointType.CITY:
                score += 18
            elif waypoint.waypoint_type == WaypointType.TOWN:
                score += 16
            elif waypoint.waypoint_type == WaypointType.VILLAGE:
                score += 14
        else:
            # Accommodation types score higher as intermediate end points
            if waypoint.waypoint_type == WaypointType.CAMPSITE:
                score += 10
            elif waypoint.waypoint_type == WaypointType.HOSTEL:
                score += 15
            elif waypoint.waypoint_type == WaypointType.GUEST_HOUSE:
                score += 12
            elif waypoint.waypoint_type == WaypointType.HOTEL:
                score += 8  # Hotels less hiker-friendly typically
            elif waypoint.waypoint_type == WaypointType.VIEWPOINT:
                score += 5  # Good for scenic value but not for staying
            elif waypoint.waypoint_type == WaypointType.PEAK:
                score += 7  # Achievement value

        # Amenity bonuses
        if waypoint.has_accommodation and prefer_accommodation:
            score += 10
        if waypoint.has_water:
            score += 3
        if waypoint.has_food:
            score += 5

        return score

    def _score_itinerary(
        self, days: list[DayRoute], options: ItineraryOptions
    ) -> float:
        """Calculate a quality score for an itinerary.

        Args:
            days: List of day routes.
            options: Generation options.

        Returns:
            Score value (higher is better).
        """
        score = 0.0

        if not days:
            return score

        # Score waypoints
        for i, day in enumerate(days):
            is_first_day = i == 0
            is_last_day = i == len(days) - 1

            # First day start and last day end should be transport-accessible
            if is_first_day:
                score += self._score_waypoint(
                    day.start_waypoint, options.prefer_accommodation, is_start_or_end=True
                )
            else:
                score += self._score_waypoint(
                    day.start_waypoint, options.prefer_accommodation, is_start_or_end=False
                ) * 0.3

            # Last day end should be transport-accessible
            if is_last_day:
                score += self._score_waypoint(
                    day.end_waypoint, options.prefer_accommodation, is_start_or_end=True
                )
            else:
                # Intermediate end waypoints are more important (where you stay)
                score += self._score_waypoint(
                    day.end_waypoint, options.prefer_accommodation, is_start_or_end=False
                )

        # Bonus points for transport-accessible start/end
        first_waypoint = days[0].start_waypoint
        last_waypoint = days[-1].end_waypoint
        if self._is_transport_accessible(first_waypoint.waypoint_type):
            score += 10  # Bonus for transport-accessible start
        if self._is_transport_accessible(last_waypoint.waypoint_type):
            score += 10  # Bonus for transport-accessible end

        # Variety bonus: different waypoint types
        types_seen = set()
        for day in days:
            types_seen.add(day.start_waypoint.waypoint_type)
            types_seen.add(day.end_waypoint.waypoint_type)
        score += len(types_seen) * 2

        # Distance balance: prefer itineraries with balanced daily distances
        distances = [day.distance_km for day in days]
        if distances:
            avg_distance = sum(distances) / len(distances)
            variance = sum((d - avg_distance) ** 2 for d in distances) / len(distances)
            # Lower variance is better
            score += max(0, 10 - variance)

        # Scenic bonus for elevation
        total_gain = sum(day.elevation_gain_m or 0 for day in days)
        score += min(total_gain / 100, 10)  # Cap at 10 bonus points

        return score

    def _find_paths_dfs(
        self,
        graph: nx.DiGraph,
        start_node: uuid.UUID,
        num_days: int,
        visited: set[uuid.UUID],
        current_path: list[tuple[uuid.UUID, uuid.UUID, dict]],
        max_overlap_km: float = 3.0,
    ) -> list[list[tuple[uuid.UUID, uuid.UUID, dict]]]:
        """Find valid paths using depth-first search with early termination.

        Args:
            graph: NetworkX graph.
            start_node: Current node.
            num_days: Remaining days to fill.
            visited: Set of visited nodes.
            current_path: Current path being built.
            max_overlap_km: Maximum allowed overlap with previous day's route.

        Returns:
            List of valid complete paths.
        """
        # Early termination check - stop if we've found enough paths
        if self._paths_found >= self.MAX_PATHS_TO_EXPLORE:
            return []

        if num_days == 0:
            self._paths_found += 1
            return [current_path.copy()]

        paths = []
        visited.add(start_node)

        for neighbor in graph.neighbors(start_node):
            # Early termination check inside loop
            if self._paths_found >= self.MAX_PATHS_TO_EXPLORE:
                break

            if neighbor not in visited:
                edge_data = graph.edges[start_node, neighbor]
                next_conn_id = edge_data.get("connection_id")

                # Check overlap with previous day's route
                if current_path:
                    prev_conn_id = current_path[-1][2].get("connection_id")
                    overlap = self._get_overlap(prev_conn_id, next_conn_id)
                    if overlap > max_overlap_km:
                        # Skip this edge - too much backtracking
                        continue

                current_path.append((start_node, neighbor, edge_data))

                sub_paths = self._find_paths_dfs(
                    graph, neighbor, num_days - 1, visited, current_path, max_overlap_km
                )
                paths.extend(sub_paths)

                current_path.pop()

        visited.remove(start_node)
        return paths

    def _find_start_waypoint(
        self, graph: nx.DiGraph, options: ItineraryOptions
    ) -> Optional[uuid.UUID]:
        """Find the starting waypoint based on options.

        Args:
            graph: NetworkX graph.
            options: Generation options.

        Returns:
            Waypoint ID or None if not found.
        """
        if options.start_waypoint_id:
            if options.start_waypoint_id not in graph.nodes:
                raise ValueError(
                    f"Start waypoint {options.start_waypoint_id} not in graph"
                )
            # Validate it's transport-accessible (unless allow_any_start is True)
            if not options.allow_any_start:
                node_data = graph.nodes[options.start_waypoint_id]
                waypoint_type = node_data.get("waypoint_type")
                if waypoint_type and not self._is_transport_accessible(waypoint_type):
                    raise ValueError(
                        f"Start waypoint must be transport-accessible "
                        f"(train station, village, town, or city), "
                        f"got: {waypoint_type}"
                    )
            return options.start_waypoint_id

        if options.start_waypoint_name:
            name_lower = options.start_waypoint_name.lower()
            for node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                if name_lower in node_data.get("name", "").lower():
                    # If allow_any_start, return first match; otherwise validate transport-accessible
                    if options.allow_any_start:
                        return node_id
                    waypoint_type = node_data.get("waypoint_type")
                    if waypoint_type and self._is_transport_accessible(waypoint_type):
                        return node_id
            if options.allow_any_start:
                raise ValueError(
                    f"No waypoint found matching name: {options.start_waypoint_name}"
                )
            raise ValueError(
                f"No transport-accessible waypoint found matching name: {options.start_waypoint_name}"
            )

        return None

    def generate(
        self, region_name: str, options: Optional[ItineraryOptions] = None
    ) -> list[Itinerary]:
        """Generate hiking itineraries for a region.

        Args:
            region_name: Name of the region.
            options: Generation options.

        Returns:
            List of itineraries, sorted by score (best first).
        """
        if options is None:
            options = ItineraryOptions()

        logger.info(
            f"Generating {options.num_days}-day itineraries for {region_name}"
        )

        # Load graph
        graph = self._load_graph(region_name)

        # Find starting points
        start_nodes = []
        specified_start = self._find_start_waypoint(graph, options)

        if specified_start:
            start_nodes = [specified_start]
        else:
            if options.allow_any_start:
                # Use all waypoints as potential starts
                start_nodes = list(graph.nodes)
            else:
                # Use transport-accessible waypoints as potential starts
                for node_id in graph.nodes:
                    node_data = graph.nodes[node_id]
                    waypoint_type = node_data.get("waypoint_type")
                    if waypoint_type and self._is_transport_accessible(waypoint_type):
                        start_nodes.append(node_id)

            # If no suitable waypoints, log warning
            if not start_nodes:
                if options.allow_any_start:
                    logger.warning("No waypoints found in the graph.")
                else:
                    logger.warning(
                        "No transport-accessible waypoints found. "
                        "Ensure waypoints are seeded with train stations, villages, towns, or cities."
                    )
            else:
                # Limit to a reasonable number of starting points for performance
                # When no specific start is provided, randomly sample up to 20 starting points
                # This prevents exponential search time when there are many waypoints
                max_start_nodes = 20
                if len(start_nodes) > max_start_nodes:
                    total_start_nodes = len(start_nodes)
                    start_nodes = random.sample(start_nodes, max_start_nodes)
                    logger.info(
                        f"Limited to {max_start_nodes} random starting points "
                        f"(out of {total_start_nodes} total) for performance"
                    )

        if not start_nodes:
            logger.warning("No suitable starting points found")
            return []

        logger.info(f"Searching from {len(start_nodes)} potential starting points")

        # Reset early termination counter
        self._paths_found = 0
        dfs_start = time.time()

        # Find valid paths with early termination
        all_paths = []
        for start_node in start_nodes:
            # Stop searching if we've found enough paths
            if self._paths_found >= self.MAX_PATHS_TO_EXPLORE:
                logger.info(
                    f"Early termination: found {self._paths_found} paths, "
                    f"stopping search"
                )
                break

            paths = self._find_paths_dfs(
                graph, start_node, options.num_days, set(), [],
                max_overlap_km=options.max_overlap_km
            )
            all_paths.extend(paths)

        dfs_time = time.time() - dfs_start
        logger.info(f"Found {len(all_paths)} valid paths in {dfs_time:.2f}s")

        if not all_paths:
            return []

        # Filter paths to ensure end waypoint is transport-accessible (unless allow_any_start)
        if options.allow_any_start:
            # No filtering needed - any end waypoint is acceptable
            filtered_paths = [path for path in all_paths if path]
            logger.info(f"Using all {len(filtered_paths)} valid paths (any end waypoint allowed)")
        else:
            filtered_paths = []
            for path in all_paths:
                if not path:
                    continue
                # Get the final waypoint (last day's end)
                last_edge = path[-1]
                final_waypoint_id = last_edge[1]  # to_id of last edge
                final_waypoint = self._waypoints[final_waypoint_id]
                if self._is_transport_accessible(final_waypoint.waypoint_type):
                    filtered_paths.append(path)

            logger.info(
                f"Filtered to {len(filtered_paths)} paths with transport-accessible end waypoints"
            )

        if not filtered_paths:
            if options.allow_any_start:
                logger.warning("No valid paths found.")
            else:
                logger.warning(
                    "No paths found ending at transport-accessible waypoints. "
                    "Ensure connections exist from accommodations to transport waypoints."
                )
            return []

        # Convert paths to itineraries
        itineraries = []
        for path in filtered_paths:
            days = []
            for i, (from_id, to_id, edge_data) in enumerate(path):
                # Parse surface stats from edge data if available
                surface_stats = SurfaceStats.from_dict(
                    edge_data.get("surface_breakdown")
                )

                day = DayRoute(
                    day_number=i + 1,
                    start_waypoint=self._waypoints[from_id],
                    end_waypoint=self._waypoints[to_id],
                    distance_km=edge_data["distance_km"],
                    duration_minutes=edge_data["duration_minutes"],
                    elevation_gain_m=edge_data.get("elevation_gain_m"),
                    elevation_loss_m=edge_data.get("elevation_loss_m"),
                    connection_id=edge_data.get("connection_id"),
                    surface_stats=surface_stats,
                )
                days.append(day)

            itinerary = Itinerary(
                id=uuid.uuid4(),
                region_name=region_name,
                days=days,
            )
            itinerary.score = self._score_itinerary(days, options)
            itineraries.append(itinerary)

        # Select itineraries based on options
        if options.randomize:
            # Randomly sample from all itineraries
            sample_size = min(options.max_results, len(itineraries))
            itineraries = random.sample(itineraries, sample_size)
            logger.info(f"Returning {len(itineraries)} random itineraries")
        else:
            # Sort by score (best first) and limit results
            itineraries.sort(key=lambda it: it.score, reverse=True)
            itineraries = itineraries[: options.max_results]
            logger.info(f"Returning top {len(itineraries)} itineraries")

        return itineraries

    def format_itinerary(
        self, itinerary: Itinerary, show_surfaces: bool = False
    ) -> str:
        """Format an itinerary as a human-readable string.

        Args:
            itinerary: Itinerary to format.
            show_surfaces: Whether to include surface breakdown per day.

        Returns:
            Formatted string.
        """
        lines = [
            f"=== {itinerary.region_name.title()} Hiking Itinerary ===",
            f"Total: {itinerary.total_distance_km:.1f} km, "
            f"{itinerary.total_duration_minutes // 60}h {itinerary.total_duration_minutes % 60}min",
            f"Elevation gain: {itinerary.total_elevation_gain_m:.0f}m",
            "",
        ]

        for day in itinerary.days:
            duration_h = day.duration_minutes // 60
            duration_m = day.duration_minutes % 60
            lines.append(f"Day {day.day_number}:")
            lines.append(f"  Start: {day.start_waypoint.name} ({day.start_waypoint.waypoint_type})")
            lines.append(f"  End:   {day.end_waypoint.name} ({day.end_waypoint.waypoint_type})")
            lines.append(f"  Distance: {day.distance_km:.1f} km")
            lines.append(f"  Duration: {duration_h}h {duration_m}min")
            if day.elevation_gain_m:
                lines.append(f"  Elevation: +{day.elevation_gain_m:.0f}m / -{day.elevation_loss_m or 0:.0f}m")

            # Add surface breakdown if requested and available
            if show_surfaces and day.surface_stats:
                lines.append("  Surface breakdown:")
                surface_pcts = day.surface_stats.surface_percentages()
                for surface, pct in surface_pcts.items():
                    distance = day.surface_stats.surfaces.get(surface, 0)
                    # Create a simple bar visualization (10 chars max)
                    bar_filled = int(pct / 10)
                    bar = "▓" * bar_filled + "░" * (10 - bar_filled)
                    # Format surface name for display
                    surface_display = surface.replace("_", " ").title()
                    lines.append(f"    {bar}  {surface_display:18} {pct:5.1f}%  ({distance:.1f} km)")

                # Also show way types
                lines.append("  Way types:")
                waytype_pcts = day.surface_stats.waytype_percentages()
                for waytype, pct in waytype_pcts.items():
                    distance = day.surface_stats.waytypes.get(waytype, 0)
                    bar_filled = int(pct / 10)
                    bar = "▓" * bar_filled + "░" * (10 - bar_filled)
                    waytype_display = waytype.replace("_", " ").title()
                    lines.append(f"    {bar}  {waytype_display:18} {pct:5.1f}%  ({distance:.1f} km)")
            elif show_surfaces and not day.surface_stats:
                lines.append("  Surface breakdown: Not available (rebuild graph to fetch)")

            lines.append("")

        return "\n".join(lines)


def generate_itineraries(
    region_name: str, options: Optional[ItineraryOptions] = None
) -> list[Itinerary]:
    """Convenience function to generate itineraries.

    Args:
        region_name: Name of the region.
        options: Generation options.

    Returns:
        List of itineraries.
    """
    generator = ItineraryGenerator()
    try:
        return generator.generate(region_name, options)
    finally:
        generator.close()


def main():
    """Test the itinerary generator."""
    print("Generating itineraries for Cornwall...")

    options = ItineraryOptions(
        num_days=3,
        prefer_accommodation=True,
        max_results=5,
    )

    generator = ItineraryGenerator()
    try:
        itineraries = generator.generate("cornwall", options)

        if not itineraries:
            print("No itineraries found. Make sure to seed waypoints and build the graph first.")
            return

        print(f"\nFound {len(itineraries)} itineraries:\n")
        for i, itinerary in enumerate(itineraries, 1):
            print(f"--- Option {i} (Score: {itinerary.score:.1f}) ---")
            print(generator.format_itinerary(itinerary))
            print()
    finally:
        generator.close()


if __name__ == "__main__":
    main()
