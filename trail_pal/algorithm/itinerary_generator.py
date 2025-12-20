"""Itinerary generator algorithm for multi-day hiking routes."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Region, Waypoint, WaypointType

logger = logging.getLogger(__name__)


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


class ItineraryGenerator:
    """Generator for multi-day hiking itineraries using graph traversal."""

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

        Args:
            region_name: Name of the region.

        Returns:
            NetworkX directed graph.
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
            edge_data = {
                "connection_id": conn.id,
                "distance_km": conn.distance_km,
                "duration_minutes": conn.duration_minutes,
                "elevation_gain_m": conn.elevation_gain_m,
                "elevation_loss_m": conn.elevation_loss_m,
            }
            # Add both directions
            graph.add_edge(conn.from_waypoint_id, conn.to_waypoint_id, **edge_data)
            # Reverse direction has swapped elevation
            reverse_data = edge_data.copy()
            reverse_data["elevation_gain_m"] = conn.elevation_loss_m
            reverse_data["elevation_loss_m"] = conn.elevation_gain_m
            graph.add_edge(conn.to_waypoint_id, conn.from_waypoint_id, **reverse_data)

        logger.info(
            f"Loaded graph with {graph.number_of_nodes()} nodes "
            f"and {graph.number_of_edges()} edges"
        )

        return graph

    def _score_waypoint(self, waypoint: Waypoint, prefer_accommodation: bool) -> float:
        """Calculate a quality score for a waypoint.

        Args:
            waypoint: Waypoint to score.
            prefer_accommodation: Whether to prioritize accommodation.

        Returns:
            Score value (higher is better).
        """
        score = 0.0

        # Accommodation types score higher as end points
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

        # Score waypoints
        for day in days:
            # End waypoints are more important (where you stay)
            score += self._score_waypoint(day.end_waypoint, options.prefer_accommodation)
            # Start waypoints matter too
            score += self._score_waypoint(day.start_waypoint, options.prefer_accommodation) * 0.3

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
    ) -> list[list[tuple[uuid.UUID, uuid.UUID, dict]]]:
        """Find all valid paths using depth-first search.

        Args:
            graph: NetworkX graph.
            start_node: Current node.
            num_days: Remaining days to fill.
            visited: Set of visited nodes.
            current_path: Current path being built.

        Returns:
            List of valid complete paths.
        """
        if num_days == 0:
            return [current_path.copy()]

        paths = []
        visited.add(start_node)

        for neighbor in graph.neighbors(start_node):
            if neighbor not in visited:
                edge_data = graph.edges[start_node, neighbor]
                current_path.append((start_node, neighbor, edge_data))

                sub_paths = self._find_paths_dfs(
                    graph, neighbor, num_days - 1, visited, current_path
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
            if options.start_waypoint_id in graph.nodes:
                return options.start_waypoint_id
            raise ValueError(
                f"Start waypoint {options.start_waypoint_id} not in graph"
            )

        if options.start_waypoint_name:
            name_lower = options.start_waypoint_name.lower()
            for node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                if name_lower in node_data.get("name", "").lower():
                    return node_id
            raise ValueError(
                f"No waypoint found matching name: {options.start_waypoint_name}"
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
            # Use waypoints with accommodation as potential starts
            for node_id in graph.nodes:
                if graph.nodes[node_id].get("has_accommodation"):
                    start_nodes.append(node_id)

            # If no accommodation waypoints, use all with sufficient connections
            if not start_nodes:
                for node_id in graph.nodes:
                    if graph.out_degree(node_id) >= options.num_days:
                        start_nodes.append(node_id)

        if not start_nodes:
            logger.warning("No suitable starting points found")
            return []

        logger.info(f"Searching from {len(start_nodes)} potential starting points")

        # Find all valid paths
        all_paths = []
        for start_node in start_nodes:
            paths = self._find_paths_dfs(
                graph, start_node, options.num_days, set(), []
            )
            all_paths.extend(paths)

        logger.info(f"Found {len(all_paths)} valid paths")

        if not all_paths:
            return []

        # Convert paths to itineraries
        itineraries = []
        for path in all_paths:
            days = []
            for i, (from_id, to_id, edge_data) in enumerate(path):
                day = DayRoute(
                    day_number=i + 1,
                    start_waypoint=self._waypoints[from_id],
                    end_waypoint=self._waypoints[to_id],
                    distance_km=edge_data["distance_km"],
                    duration_minutes=edge_data["duration_minutes"],
                    elevation_gain_m=edge_data.get("elevation_gain_m"),
                    elevation_loss_m=edge_data.get("elevation_loss_m"),
                    connection_id=edge_data.get("connection_id"),
                )
                days.append(day)

            itinerary = Itinerary(
                id=uuid.uuid4(),
                region_name=region_name,
                days=days,
            )
            itinerary.score = self._score_itinerary(days, options)
            itineraries.append(itinerary)

        # Sort by score (best first) and limit results
        itineraries.sort(key=lambda it: it.score, reverse=True)
        itineraries = itineraries[: options.max_results]

        logger.info(f"Returning top {len(itineraries)} itineraries")
        return itineraries

    def format_itinerary(self, itinerary: Itinerary) -> str:
        """Format an itinerary as a human-readable string.

        Args:
            itinerary: Itinerary to format.

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

