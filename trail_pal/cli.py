"""Command-line interface for Trail Pal."""

import asyncio
import logging
import sys
import uuid
from typing import Optional

import click
import gpxpy
import gpxpy.gpx

from trail_pal.algorithm.itinerary_generator import (
    Itinerary,
    ItineraryGenerator,
    ItineraryOptions,
)
from trail_pal.db.database import init_db
from trail_pal.services.graph_builder import GraphBuilder
from trail_pal.services.waypoint_seeder import WaypointSeeder, list_available_regions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """Trail Pal - Generate multi-day hiking itineraries."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
def init():
    """Initialize the database tables."""
    click.echo("Initializing database...")
    try:
        init_db()
        click.echo("Database initialized successfully!")
    except Exception as e:
        click.echo(f"Error initializing database: {e}", err=True)
        sys.exit(1)


@cli.command()
def regions():
    """List available regions."""
    click.echo("Available regions:")
    for region in list_available_regions():
        click.echo(f"  - {region}")


@cli.command()
@click.option(
    "--region",
    "-r",
    required=True,
    help="Region to seed (e.g., 'cornwall')",
)
def seed(region: str):
    """Seed waypoints from OpenStreetMap for a region."""
    click.echo(f"Seeding waypoints for region: {region}")

    try:
        seeder = WaypointSeeder()
        stats = asyncio.run(seeder.seed_region(region))
        seeder.close()

        click.echo("\nSeeding complete!")
        click.echo(f"  Region: {stats['region']}")
        click.echo(f"  Total fetched: {stats['total_fetched']}")
        click.echo(f"  Inserted: {stats['inserted']}")
        click.echo(f"  Updated: {stats['updated']}")
        click.echo(f"  Last synced: {stats['last_synced']}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error seeding waypoints: {e}", err=True)
        logger.exception("Seeding failed")
        sys.exit(1)


@cli.command("build-graph")
@click.option(
    "--region",
    "-r",
    required=True,
    help="Region to build graph for",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Rebuild existing connections",
)
def build_graph(region: str, force: bool):
    """Build the feasibility graph for a region."""
    click.echo(f"Building feasibility graph for region: {region}")
    if force:
        click.echo("(Force mode: rebuilding existing connections)")

    def progress_callback(current: int, total: int, from_name: str, to_name: str):
        click.echo(f"  [{current}/{total}] {from_name} -> {to_name}")

    try:
        builder = GraphBuilder()

        async def run_build():
            return await builder.build_graph(
                region,
                skip_existing=not force,
                progress_callback=progress_callback,
            )

        stats = asyncio.run(run_build())
        builder.close()

        click.echo("\nGraph building complete!")
        click.echo(f"  Region: {stats['region']}")
        click.echo(f"  Total waypoints: {stats['total_waypoints']}")
        click.echo(f"  Candidate pairs: {stats['total_candidates']}")
        click.echo(f"  Connections created: {stats['connections_created']}")
        click.echo(f"  Feasible connections: {stats['feasible_connections']}")
        click.echo(f"  Failed routes: {stats['failed_routes']}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error building graph: {e}", err=True)
        logger.exception("Graph building failed")
        sys.exit(1)


@cli.command("graph-stats")
@click.option(
    "--region",
    "-r",
    required=True,
    help="Region to get stats for",
)
def graph_stats(region: str):
    """Show statistics about the feasibility graph."""
    try:
        builder = GraphBuilder()
        stats = builder.get_graph_stats(region)
        builder.close()

        if stats is None:
            click.echo(f"Region not found: {region}", err=True)
            sys.exit(1)

        click.echo(f"Graph statistics for {region}:")
        click.echo(f"  Total waypoints: {stats['total_waypoints']}")
        click.echo(f"  Total connections: {stats['total_connections']}")
        click.echo(f"  Feasible connections: {stats['feasible_connections']}")
        click.echo(f"  Average distance: {stats['avg_distance_km']} km")
        click.echo(f"  Average duration: {stats['avg_duration_min']} min")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--region",
    "-r",
    required=True,
    help="Region to generate itinerary for",
)
@click.option(
    "--days",
    "-d",
    default=3,
    help="Number of hiking days (default: 3)",
)
@click.option(
    "--start",
    "-s",
    default=None,
    help="Starting waypoint name (partial match)",
)
@click.option(
    "--results",
    "-n",
    default=5,
    help="Number of results to show (default: 5)",
)
@click.option(
    "--no-accommodation",
    is_flag=True,
    help="Don't prefer waypoints with accommodation",
)
@click.option(
    "--show-surfaces",
    is_flag=True,
    help="Show surface type breakdown for each day",
)
@click.option(
    "--random",
    "randomize",
    is_flag=True,
    help="Randomize results instead of returning top-scored routes",
)
def generate(
    region: str,
    days: int,
    start: Optional[str],
    results: int,
    no_accommodation: bool,
    show_surfaces: bool,
    randomize: bool,
):
    """Generate hiking itineraries."""
    click.echo(f"Generating {days}-day itineraries for {region}...")

    options = ItineraryOptions(
        num_days=days,
        start_waypoint_name=start,
        prefer_accommodation=not no_accommodation,
        max_results=results,
        randomize=randomize,
    )

    try:
        generator = ItineraryGenerator()
        itineraries = generator.generate(region, options)

        if not itineraries:
            click.echo("\nNo itineraries found.")
            click.echo("Make sure to run 'seed' and 'build-graph' first.")
            return

        click.echo(f"\nFound {len(itineraries)} itineraries:\n")

        for i, itinerary in enumerate(itineraries, 1):
            click.echo(f"{'=' * 50}")
            click.echo(f"Option {i} (Score: {itinerary.score:.1f})")
            click.echo(f"{'=' * 50}")
            click.echo(generator.format_itinerary(itinerary, show_surfaces=show_surfaces))

        generator.close()

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error generating itineraries: {e}", err=True)
        logger.exception("Generation failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--region",
    "-r",
    required=True,
    help="Region to generate from",
)
@click.option(
    "--days",
    "-d",
    default=3,
    help="Number of hiking days",
)
@click.option(
    "--start",
    "-s",
    default=None,
    help="Starting waypoint name",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output GPX file path",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["gpx", "json"]),
    default="gpx",
    help="Output format",
)
@click.option(
    "--random",
    "randomize",
    is_flag=True,
    help="Export a random route instead of top-scored",
)
@click.option(
    "--fresh",
    is_flag=True,
    help="Fetch fresh routes from API instead of using cached database routes",
)
def export(
    region: str,
    days: int,
    start: Optional[str],
    output: str,
    output_format: str,
    randomize: bool,
    fresh: bool,
):
    """Export the best itinerary to GPX or JSON."""
    click.echo(f"Generating and exporting itinerary...")

    options = ItineraryOptions(
        num_days=days,
        start_waypoint_name=start,
        max_results=1,
        randomize=randomize,
    )

    try:
        generator = ItineraryGenerator()
        itineraries = generator.generate(region, options)

        if not itineraries:
            click.echo("No itineraries found.", err=True)
            sys.exit(1)

        itinerary = itineraries[0]
        
        # Show which route was selected
        click.echo(f"Route: {itinerary.days[0].start_waypoint.name} → ... → {itinerary.days[-1].end_waypoint.name}")
        click.echo(f"Total: {itinerary.total_distance_km:.1f} km, {len(itinerary.days)} days")

        if output_format == "gpx":
            _export_gpx(itinerary, output, fetch_fresh=fresh)
        else:
            _export_json(itinerary, output)

        click.echo(f"Exported to: {output}")
        generator.close()

    except Exception as e:
        click.echo(f"Error exporting: {e}", err=True)
        logger.exception("Export failed")
        sys.exit(1)


def _export_gpx(itinerary: Itinerary, output_path: str, fetch_fresh: bool = False):
    """Export itinerary to GPX format with full trail geometry.
    
    Args:
        itinerary: The itinerary to export.
        output_path: Path to write the GPX file.
        fetch_fresh: If True, always fetch fresh routes from ORS API.
    """
    from geoalchemy2.shape import to_shape
    from sqlalchemy import select
    from trail_pal.db.database import SessionLocal
    from trail_pal.db.models import Connection
    from trail_pal.services.ors_client import OpenRouteServiceClient

    gpx = gpxpy.gpx.GPX()
    gpx.name = f"{itinerary.region_name.title()} Hiking Itinerary"
    gpx.description = (
        f"{len(itinerary.days)}-day hiking route, "
        f"{itinerary.total_distance_km:.1f} km total"
    )

    # Add waypoints for accommodations
    for day in itinerary.days:
        # Start waypoint
        wp_start = gpxpy.gpx.GPXWaypoint(
            latitude=day.start_waypoint.latitude,
            longitude=day.start_waypoint.longitude,
            name=f"Day {day.day_number} Start: {day.start_waypoint.name}",
            description=f"Type: {day.start_waypoint.waypoint_type}",
        )
        gpx.waypoints.append(wp_start)

        # End waypoint (only add if it's the last day)
        if day.day_number == len(itinerary.days):
            wp_end = gpxpy.gpx.GPXWaypoint(
                latitude=day.end_waypoint.latitude,
                longitude=day.end_waypoint.longitude,
                name=f"Day {day.day_number} End: {day.end_waypoint.name}",
                description=f"Type: {day.end_waypoint.waypoint_type}",
            )
            gpx.waypoints.append(wp_end)

    # Fetch route geometries from database and add as tracks
    db = SessionLocal()
    try:
        # Check if we need to fetch any routes on-the-fly
        needs_fetch = []
        for day in itinerary.days:
            # If fetch_fresh is True, always fetch new routes
            if fetch_fresh:
                needs_fetch.append(day)
                continue
                
            has_geometry = False
            
            # Check forward connection
            if day.connection_id:
                stmt = select(Connection).where(Connection.id == day.connection_id)
                connection = db.execute(stmt).scalar_one_or_none()
                if connection and connection.route_geometry:
                    has_geometry = True
            
            # Check reverse connection
            if not has_geometry:
                stmt = select(Connection).where(
                    (Connection.from_waypoint_id == day.end_waypoint.id) &
                    (Connection.to_waypoint_id == day.start_waypoint.id)
                )
                reverse_conn = db.execute(stmt).scalar_one_or_none()
                if reverse_conn and reverse_conn.route_geometry:
                    has_geometry = True
            
            if not has_geometry:
                needs_fetch.append(day)
        
        # Fetch missing routes if needed
        fetched_routes = {}
        if needs_fetch:
            logger.info(f"Fetching {len(needs_fetch)} missing route(s) from OpenRouteService...")
            async def fetch_missing_routes():
                async with OpenRouteServiceClient() as ors_client:
                    for day in needs_fetch:
                        route = await ors_client.get_hiking_route(
                            start_lon=day.start_waypoint.longitude,
                            start_lat=day.start_waypoint.latitude,
                            end_lon=day.end_waypoint.longitude,
                            end_lat=day.end_waypoint.latitude,
                        )
                        if route:
                            fetched_routes[day.day_number] = route
                        else:
                            logger.warning(
                                f"Failed to fetch route for Day {day.day_number}"
                            )
            asyncio.run(fetch_missing_routes())

        for day in itinerary.days:
            # Create a track for each day
            track = gpxpy.gpx.GPXTrack(
                name=f"Day {day.day_number}: {day.start_waypoint.name} to {day.end_waypoint.name}"
            )
            track.description = (
                f"{day.distance_km:.1f} km, {day.duration_minutes} min"
            )

            segment = gpxpy.gpx.GPXTrackSegment()

            # Try to get the route geometry from the connection or fetched route
            geometry_found = False
            
            # First, try fetched route if available
            if day.day_number in fetched_routes:
                route = fetched_routes[day.day_number]
                if route.geometry:
                    for lon, lat in route.geometry:
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            segment.points.append(
                                gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
                            )
                    if segment.points:
                        geometry_found = True
                        logger.info(
                            f"Added {len(segment.points)} points from fetched route "
                            f"for Day {day.day_number}"
                        )
            
            # Otherwise, try database - check both forward and reverse connections
            if not geometry_found:
                connection = None
                
                # First try the connection_id from the itinerary
                if day.connection_id:
                    stmt = select(Connection).where(Connection.id == day.connection_id)
                    connection = db.execute(stmt).scalar_one_or_none()
                
                # If no geometry in forward direction, try reverse connection
                # (since graph is bidirectional, reverse might have geometry)
                if not connection or not connection.route_geometry:
                    stmt = select(Connection).where(
                        (Connection.from_waypoint_id == day.end_waypoint.id) &
                        (Connection.to_waypoint_id == day.start_waypoint.id)
                    )
                    reverse_conn = db.execute(stmt).scalar_one_or_none()
                    if reverse_conn and reverse_conn.route_geometry:
                        connection = reverse_conn
                        logger.debug(
                            f"Using reverse connection geometry for Day {day.day_number}"
                        )

                if connection and connection.route_geometry:
                    try:
                        # Convert PostGIS geometry to Shapely and extract coordinates
                        line = to_shape(connection.route_geometry)
                        
                        # Extract all coordinates from the LineString
                        # LineString.coords returns (lon, lat) tuples
                        # If using reverse connection, reverse the coordinates
                        coords = list(line.coords)
                        if connection.from_waypoint_id == day.end_waypoint.id:
                            # Reverse connection - reverse the coordinate order
                            coords = list(reversed(coords))
                        
                        for coord in coords:
                            lon, lat = coord[0], coord[1]
                            # Basic validation: check for reasonable coordinate values
                            # (WGS84 bounds: lon -180 to 180, lat -90 to 90)
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                segment.points.append(
                                    gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
                                )
                        
                        if segment.points:
                            geometry_found = True
                            logger.debug(
                                f"Added {len(segment.points)} points from route geometry "
                                f"for Day {day.day_number}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract geometry for Day {day.day_number}: {e}"
                        )

            # Fallback: if no geometry, just add start and end points
            if not geometry_found:
                logger.warning(
                    f"No route geometry found for Day {day.day_number} "
                    f"({day.start_waypoint.name} -> {day.end_waypoint.name}). "
                    f"Using straight-line fallback."
                )
                segment.points.append(
                    gpxpy.gpx.GPXTrackPoint(
                        latitude=day.start_waypoint.latitude,
                        longitude=day.start_waypoint.longitude,
                    )
                )
                segment.points.append(
                    gpxpy.gpx.GPXTrackPoint(
                        latitude=day.end_waypoint.latitude,
                        longitude=day.end_waypoint.longitude,
                    )
                )

            track.segments.append(segment)
            gpx.tracks.append(track)
    finally:
        db.close()

    with open(output_path, "w") as f:
        f.write(gpx.to_xml())


def _export_json(itinerary: Itinerary, output_path: str):
    """Export itinerary to JSON format."""
    import json

    data = {
        "id": str(itinerary.id),
        "region": itinerary.region_name,
        "total_distance_km": itinerary.total_distance_km,
        "total_duration_minutes": itinerary.total_duration_minutes,
        "total_elevation_gain_m": itinerary.total_elevation_gain_m,
        "score": itinerary.score,
        "days": [],
    }

    for day in itinerary.days:
        day_data = {
            "day_number": day.day_number,
            "start": {
                "name": day.start_waypoint.name,
                "type": day.start_waypoint.waypoint_type,
                "latitude": day.start_waypoint.latitude,
                "longitude": day.start_waypoint.longitude,
            },
            "end": {
                "name": day.end_waypoint.name,
                "type": day.end_waypoint.waypoint_type,
                "latitude": day.end_waypoint.latitude,
                "longitude": day.end_waypoint.longitude,
            },
            "distance_km": day.distance_km,
            "duration_minutes": day.duration_minutes,
            "elevation_gain_m": day.elevation_gain_m,
            "elevation_loss_m": day.elevation_loss_m,
        }

        # Include surface breakdown if available
        if day.surface_stats:
            day_data["surface_breakdown"] = {
                "surfaces": day.surface_stats.surfaces,
                "waytypes": day.surface_stats.waytypes,
                "surface_percentages": day.surface_stats.surface_percentages(),
                "waytype_percentages": day.surface_stats.waytype_percentages(),
            }

        data["days"].append(day_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


@cli.command("waypoint-stats")
@click.option(
    "--region",
    "-r",
    required=True,
    help="Region to get stats for",
)
def waypoint_stats(region: str):
    """Show waypoint statistics for a region."""
    try:
        seeder = WaypointSeeder()
        stats = seeder.get_region_stats(region)
        seeder.close()

        if stats is None:
            click.echo(f"Region not found: {region}", err=True)
            sys.exit(1)

        click.echo(f"Waypoint statistics for {region}:")
        click.echo(f"  Country: {stats['country']}")
        click.echo(f"  Total waypoints: {stats['total_waypoints']}")
        click.echo(f"  Last synced: {stats['last_synced']}")
        click.echo("\n  Waypoints by type:")
        for wp_type, count in stats["waypoints_by_type"].items():
            click.echo(f"    - {wp_type}: {count}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()

