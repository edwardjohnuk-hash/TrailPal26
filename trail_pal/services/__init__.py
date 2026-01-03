"""Services for Trail Pal."""

from trail_pal.services.osm_client import (
    BoundingBox,
    OSMElement,
    OverpassClient,
    REGION_BOUNDS,
    get_region_bounds,
)
from trail_pal.services.ors_client import (
    OpenRouteServiceClient,
    RateLimiter,
    RouteResult,
    calculate_straight_line_distance_km,
)
from trail_pal.services.waypoint_seeder import (
    WaypointSeeder,
    list_available_regions,
    seed_region,
)
from trail_pal.services.graph_builder import (
    GraphBuilder,
    build_graph,
)
from trail_pal.services.google_places_client import (
    GooglePlacesClient,
    PlaceResult,
)
from trail_pal.services.pub_finder import (
    PubFinder,
)

__all__ = [
    # OSM Client
    "BoundingBox",
    "OSMElement",
    "OverpassClient",
    "REGION_BOUNDS",
    "get_region_bounds",
    # ORS Client
    "OpenRouteServiceClient",
    "RateLimiter",
    "RouteResult",
    "calculate_straight_line_distance_km",
    # Waypoint Seeder
    "WaypointSeeder",
    "list_available_regions",
    "seed_region",
    # Graph Builder
    "GraphBuilder",
    "build_graph",
    # Google Places Client
    "GooglePlacesClient",
    "PlaceResult",
    # Pub Finder
    "PubFinder",
]

