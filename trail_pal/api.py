"""FastAPI application for Trail Pal REST API."""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
import gpxpy
import gpxpy.gpx
from geoalchemy2.shape import to_shape

from trail_pal.algorithm.itinerary_generator import (
    ItineraryGenerator,
    ItineraryOptions,
)
from trail_pal.algorithm.onthefly_generator import OnTheFlyGenerator
from trail_pal.db.models import RoutingMode
from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Region, RouteFeedback, Waypoint
from trail_pal.services.pub_recommender import PubRecommenderService, PubRecommendation

logger = logging.getLogger(__name__)
settings = get_settings()

# --- Pydantic Models ---


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str = "trail-pal"


class RegionResponse(BaseModel):
    """Region information."""

    id: UUID
    name: str
    country: str
    description: Optional[str] = None
    waypoint_count: int = 0

    class Config:
        from_attributes = True


class WaypointResponse(BaseModel):
    """Waypoint information."""

    id: UUID
    name: str
    waypoint_type: str
    latitude: float
    longitude: float
    has_accommodation: bool
    has_water: bool
    has_food: bool
    elevation_m: Optional[float] = None

    class Config:
        from_attributes = True


class RegionStatsResponse(BaseModel):
    """Statistics for a region."""

    region: str
    country: str
    routing_mode: str = Field(
        default="precomputed",
        description="Routing mode: 'precomputed' uses cached graph, 'on_the_fly' fetches routes on demand"
    )
    total_waypoints: int
    total_connections: int
    feasible_connections: int
    avg_distance_km: Optional[float] = None
    avg_duration_min: Optional[float] = None
    waypoints_by_type: dict[str, int]


class SurfaceStatsResponse(BaseModel):
    """Surface breakdown statistics for a route segment."""

    surfaces: dict[str, float] = Field(
        default_factory=dict, description="Surface type -> distance in km"
    )
    waytypes: dict[str, float] = Field(
        default_factory=dict, description="Way type -> distance in km"
    )
    total_distance_km: float = 0.0


class PubRecommendationResponse(BaseModel):
    """A pub recommendation for a location along the hike."""

    name: str = Field(..., description="Name of the pub")
    rating: float = Field(..., description="Google rating (4.2+)")
    latitude: float = Field(..., description="Pub latitude")
    longitude: float = Field(..., description="Pub longitude")
    place_id: str = Field(..., description="Google Place ID")
    distance_m: float = Field(..., description="Distance from the waypoint/midpoint in meters")
    user_ratings_total: Optional[int] = Field(None, description="Total number of ratings")


class DayRouteResponse(BaseModel):
    """A single day's hiking route."""

    day_number: int
    start: WaypointResponse
    end: WaypointResponse
    distance_km: float
    duration_minutes: int
    elevation_gain_m: Optional[float] = None
    elevation_loss_m: Optional[float] = None
    surface_stats: Optional[SurfaceStatsResponse] = None
    start_pub: Optional[PubRecommendationResponse] = Field(
        None, description="Recommended pub near start waypoint (4.2+ rating)"
    )
    end_pub: Optional[PubRecommendationResponse] = Field(
        None, description="Recommended pub near end waypoint (4.2+ rating)"
    )
    midpoint_pub: Optional[PubRecommendationResponse] = Field(
        None, description="Recommended pub near route midpoint (4.2+ rating)"
    )


class ItineraryResponse(BaseModel):
    """Complete hiking itinerary."""

    id: UUID
    region: str
    days: list[DayRouteResponse]
    total_distance_km: float
    total_duration_minutes: int
    total_elevation_gain_m: float
    score: float


class GenerateRequest(BaseModel):
    """Request to generate itineraries."""

    region: str = Field(..., description="Region name (e.g., 'cornwall')")
    days: int = Field(default=3, ge=1, le=14, description="Number of hiking days")
    start_waypoint_name: Optional[str] = Field(
        default=None, description="Optional starting waypoint name (partial match)"
    )
    prefer_accommodation: bool = Field(
        default=True, description="Prefer waypoints with accommodation"
    )
    max_results: int = Field(
        default=5, ge=1, le=20, description="Maximum number of itineraries to return"
    )
    randomize: bool = Field(
        default=True,
        description="Return random itinerary combinations (set False for consistent top-scored results)",
    )
    allow_any_start: bool = Field(
        default=False,
        description="Allow any waypoint type as start/end (not just train stations/towns)",
    )


class DayDataRequest(BaseModel):
    """Day data from a previously generated itinerary."""

    day_number: int
    start_id: UUID
    end_id: UUID


class ItineraryDataRequest(BaseModel):
    """Request containing previously generated itinerary data."""

    region: str = Field(..., description="Region name")
    itinerary_id: UUID = Field(..., description="The ID of the previously generated itinerary")
    days: list[DayDataRequest] = Field(..., description="Day data with waypoint IDs")


class GenerateResponse(BaseModel):
    """Response containing generated itineraries."""

    count: int
    itineraries: list[ItineraryResponse]


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str


class DayPubsRequest(BaseModel):
    """Day data for pub recommendations request."""

    day_number: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    connection_id: Optional[UUID] = None


class PubsRequest(BaseModel):
    """Request for on-demand pub recommendations."""

    days: list[DayPubsRequest] = Field(..., description="Day data with coordinates")


class DayPubsResponse(BaseModel):
    """Pub recommendations for a single day."""

    day_number: int
    start_pub: Optional[PubRecommendationResponse] = None
    end_pub: Optional[PubRecommendationResponse] = None
    midpoint_pub: Optional[PubRecommendationResponse] = None


class PubsResponse(BaseModel):
    """Response containing pub recommendations for all days."""

    days: list[DayPubsResponse]


class DayGeometryResponse(BaseModel):
    """Route geometry for a single day."""

    day_number: int
    start: WaypointResponse
    end: WaypointResponse
    geometry: list[list[float]] = Field(
        default_factory=list,
        description="Array of [lon, lat] coordinate pairs for the route path",
    )


class ItineraryGeometryResponse(BaseModel):
    """Route geometry for an entire itinerary."""

    region: str
    days: list[DayGeometryResponse]


class PubResponse(BaseModel):
    """Pub information."""

    id: UUID
    name: str
    latitude: float
    longitude: float
    rating: float
    user_ratings_total: Optional[int] = None
    distance_m: float
    location_type: str
    google_place_id: str

    class Config:
        from_attributes = True


class WaypointPubResponse(BaseModel):
    """Waypoint with associated pubs."""

    waypoint: WaypointResponse
    pubs: list[PubResponse] = Field(default_factory=list)


class PubDiscoveryResponse(BaseModel):
    """Pub discovery statistics."""

    region: str
    waypoints_processed: Optional[int] = None
    connections_processed: Optional[int] = None
    pubs_found: int
    pubs_stored: int


class FeedbackRequest(BaseModel):
    """Request to submit route feedback."""

    itinerary_id: UUID = Field(..., description="ID of the itinerary being rated")
    region: str = Field(..., description="Region name")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback_reasons: list[str] = Field(
        default_factory=list, description="List of selected feedback reasons"
    )
    route_summary: dict = Field(
        default_factory=dict,
        description="Summary of the route (days, distance, waypoints)",
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    id: UUID
    message: str = "Thank you for your feedback!"


# --- Dependencies ---


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Verify the API key from request header."""
    valid_keys = settings.api_keys_list

    # If no keys configured, allow all requests (development mode)
    if not valid_keys:
        logger.warning("No API keys configured - allowing unauthenticated access")
        return None

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key


async def optional_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Optionally verify API key - allows unauthenticated access for web UI.
    
    If a key is provided, it must be valid. If no key is provided, access is allowed.
    """
    valid_keys = settings.api_keys_list

    # If no keys configured or no key provided, allow access
    if not valid_keys or not x_api_key:
        return None

    # If key provided, validate it
    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key


# --- FastAPI App ---

app = FastAPI(
    title="Trail Pal API",
    description="Generate multi-day hiking itineraries for scenic trails",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for third-party websites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# --- Startup Events ---


@app.on_event("startup")
async def warmup_graph_cache():
    """Pre-load graph cache for known regions on startup.
    
    This avoids a 15-20 second delay on the first itinerary request
    by loading the 1.4M+ overlap records into memory at startup.
    Only warms cache for regions with precomputed routing mode.
    """
    import asyncio
    
    # Run cache warmup in background to not block startup
    async def warmup():
        try:
            logger.info("Warming up graph cache...")
            generator = ItineraryGenerator()
            
            # Get all regions from database
            db = SessionLocal()
            try:
                stmt = select(Region)
                regions = list(db.execute(stmt).scalars().all())
            finally:
                db.close()
            
            # Pre-load graph for each precomputed region
            for region in regions:
                routing_mode = getattr(region, 'routing_mode', RoutingMode.PRECOMPUTED)
                
                if routing_mode == RoutingMode.ON_THE_FLY:
                    logger.info(f"Skipping cache warmup for on-the-fly region: {region.name}")
                    continue
                
                try:
                    # Directly load the graph to populate cache without generating itineraries
                    # This avoids "no paths found" warnings during startup
                    generator._load_graph(region.name)
                    logger.info(f"Graph cache warmed for region: {region.name}")
                except Exception as e:
                    logger.warning(f"Failed to warm cache for {region.name}: {e}")
            
            generator.close()
            logger.info("Graph cache warmup complete")
        except Exception as e:
            logger.error(f"Graph cache warmup failed: {e}")
    
    # Start warmup as background task (don't block startup)
    asyncio.create_task(warmup())


# --- Routes ---


@app.get("/")
async def root():
    """Serve the main UI page."""
    static_dir = Path(__file__).parent.parent / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "UI not found. Please ensure static/index.html exists."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse()


@app.get(
    "/regions",
    response_model=list[RegionResponse],
    tags=["Regions"],
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def list_regions(
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """List all available hiking regions."""
    try:
        stmt = select(Region)
        regions = list(db.execute(stmt).scalars().all())

        result = []
        for region in regions:
            # Count waypoints for each region
            count_stmt = select(func.count(Waypoint.id)).where(
                Waypoint.region_id == region.id
            )
            waypoint_count = db.execute(count_stmt).scalar() or 0

            result.append(
                RegionResponse(
                    id=region.id,
                    name=region.name,
                    country=region.country,
                    description=region.description,
                    waypoint_count=waypoint_count,
                )
            )

        return result
    except Exception as e:
        logger.error(f"Error listing regions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@app.get(
    "/regions/{name}/waypoints",
    response_model=list[WaypointResponse],
    tags=["Regions"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def list_region_waypoints(
    name: str,
    waypoint_type: Optional[str] = Query(
        None, description="Filter by waypoint type (e.g., 'campsite', 'hostel')"
    ),
    has_accommodation: Optional[bool] = Query(
        None, description="Filter by accommodation availability"
    ),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """List waypoints in a region with optional filtering."""
    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    # Build query with filters
    stmt = select(Waypoint).where(Waypoint.region_id == region.id)

    if waypoint_type:
        stmt = stmt.where(Waypoint.waypoint_type == waypoint_type.lower())

    if has_accommodation is not None:
        stmt = stmt.where(Waypoint.has_accommodation == has_accommodation)

    stmt = stmt.offset(offset).limit(limit)
    waypoints = list(db.execute(stmt).scalars().all())

    return [
        WaypointResponse(
            id=wp.id,
            name=wp.name,
            waypoint_type=wp.waypoint_type,
            latitude=wp.latitude,
            longitude=wp.longitude,
            has_accommodation=wp.has_accommodation,
            has_water=wp.has_water,
            has_food=wp.has_food,
            elevation_m=wp.elevation_m,
        )
        for wp in waypoints
    ]


@app.get(
    "/regions/{name}/waypoints/search",
    response_model=list[WaypointResponse],
    tags=["Regions"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def search_waypoints(
    name: str,
    q: str = Query(..., description="Search query (waypoint name)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    include_all: bool = Query(
        False, description="Include all waypoint types (not just train stations/towns)"
    ),
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """Search waypoints in a region by name (for autocomplete).

    Returns waypoints whose names contain the search query (case-insensitive).
    By default, only returns train stations and towns (suitable for starting points).
    Set include_all=true to return all waypoint types.
    Results are limited and sorted by name.
    """
    from trail_pal.db.models import WaypointType
    
    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    # Build query with name search (case-insensitive)
    search_term = f"%{q.lower()}%"
    
    if include_all:
        # Return all waypoint types
        stmt = (
            select(Waypoint)
            .where(
                Waypoint.region_id == region.id,
                func.lower(Waypoint.name).like(search_term),
            )
            .order_by(Waypoint.name)
            .limit(limit)
        )
    else:
        # Default: only train stations and towns (suitable for starting points)
        stmt = (
            select(Waypoint)
            .where(
                Waypoint.region_id == region.id,
                func.lower(Waypoint.name).like(search_term),
                Waypoint.waypoint_type.in_([
                    WaypointType.TRAIN_STATION,
                    WaypointType.TOWN,
                ])
            )
            .order_by(Waypoint.name)
            .limit(limit)
        )
    
    waypoints = list(db.execute(stmt).scalars().all())

    return [
        WaypointResponse(
            id=wp.id,
            name=wp.name,
            waypoint_type=wp.waypoint_type,
            latitude=wp.latitude,
            longitude=wp.longitude,
            has_accommodation=wp.has_accommodation,
            has_water=wp.has_water,
            has_food=wp.has_food,
            elevation_m=wp.elevation_m,
        )
        for wp in waypoints
    ]


@app.get(
    "/regions/{name}/stats",
    response_model=RegionStatsResponse,
    tags=["Regions"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_region_stats(
    name: str,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """Get statistics for a region including waypoints and connections."""
    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    # Count waypoints
    wp_count_stmt = select(func.count(Waypoint.id)).where(
        Waypoint.region_id == region.id
    )
    total_waypoints = db.execute(wp_count_stmt).scalar() or 0

    # Get waypoint IDs for connection queries
    wp_ids_stmt = select(Waypoint.id).where(Waypoint.region_id == region.id)
    wp_ids = list(db.execute(wp_ids_stmt).scalars().all())

    # Count connections
    conn_count_stmt = select(func.count(Connection.id)).where(
        Connection.from_waypoint_id.in_(wp_ids)
    )
    total_connections = db.execute(conn_count_stmt).scalar() or 0

    # Count feasible connections
    feasible_stmt = select(func.count(Connection.id)).where(
        Connection.from_waypoint_id.in_(wp_ids),
        Connection.is_feasible == True,  # noqa: E712
    )
    feasible_connections = db.execute(feasible_stmt).scalar() or 0

    # Average distance and duration for feasible connections
    avg_stmt = select(
        func.avg(Connection.distance_km),
        func.avg(Connection.duration_minutes),
    ).where(
        Connection.from_waypoint_id.in_(wp_ids),
        Connection.is_feasible == True,  # noqa: E712
    )
    avg_result = db.execute(avg_stmt).one()
    avg_distance = round(avg_result[0], 2) if avg_result[0] else None
    avg_duration = round(avg_result[1], 1) if avg_result[1] else None

    # Waypoints by type
    type_stmt = (
        select(Waypoint.waypoint_type, func.count(Waypoint.id))
        .where(Waypoint.region_id == region.id)
        .group_by(Waypoint.waypoint_type)
    )
    type_results = db.execute(type_stmt).all()
    waypoints_by_type = {row[0]: row[1] for row in type_results}

    return RegionStatsResponse(
        region=region.name,
        country=region.country,
        routing_mode=getattr(region, 'routing_mode', RoutingMode.PRECOMPUTED),
        total_waypoints=total_waypoints,
        total_connections=total_connections,
        feasible_connections=feasible_connections,
        avg_distance_km=avg_distance,
        avg_duration_min=avg_duration,
        waypoints_by_type=waypoints_by_type,
    )


@app.post(
    "/itineraries/generate",
    response_model=GenerateResponse,
    tags=["Itineraries"],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def generate_itineraries(
    request: GenerateRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """Generate hiking itineraries for a region.

    Returns a list of multi-day hiking itineraries ranked by quality score.
    Each itinerary includes daily routes with distances, durations, and elevation data.
    """
    # Validate region exists
    region_stmt = select(Region).where(Region.name == request.region.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {request.region}",
        )

    # Create options
    options = ItineraryOptions(
        num_days=request.days,
        start_waypoint_name=request.start_waypoint_name,
        prefer_accommodation=request.prefer_accommodation,
        max_results=request.max_results,
        randomize=request.randomize,
        allow_any_start=request.allow_any_start,
    )

    # Generate itineraries using appropriate generator based on routing mode
    try:
        routing_mode = getattr(region, 'routing_mode', RoutingMode.PRECOMPUTED)
        
        if routing_mode == RoutingMode.ON_THE_FLY:
            # Use on-the-fly generator for regions without precomputed graphs
            logger.info(f"Using on-the-fly generator for region: {request.region}")
            generator = OnTheFlyGenerator(db=db, persist_routes=True)
            itineraries = await generator.generate(request.region, options)
        else:
            # Use precomputed graph generator (default)
            generator = ItineraryGenerator(db=db)
            itineraries = generator.generate(request.region, options)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error generating itineraries")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating itineraries: {str(e)}",
        )

    if not itineraries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No itineraries found. Ensure waypoints and connections exist for this region.",
        )

    # Convert to response models
    # Note: Pub recommendations are fetched on-demand via /itineraries/pubs endpoint
    response_itineraries = []
    for it in itineraries:
        days = []
        for day in it.days:
            # Convert surface_stats if available
            surface_stats_response = None
            if day.surface_stats:
                surface_stats_response = SurfaceStatsResponse(
                    surfaces=day.surface_stats.surfaces,
                    waytypes=day.surface_stats.waytypes,
                    total_distance_km=day.surface_stats.total_distance_km,
                )

            days.append(
                DayRouteResponse(
                    day_number=day.day_number,
                    start=WaypointResponse(
                        id=day.start_waypoint.id,
                        name=day.start_waypoint.name,
                        waypoint_type=day.start_waypoint.waypoint_type,
                        latitude=day.start_waypoint.latitude,
                        longitude=day.start_waypoint.longitude,
                        has_accommodation=day.start_waypoint.has_accommodation,
                        has_water=day.start_waypoint.has_water,
                        has_food=day.start_waypoint.has_food,
                        elevation_m=day.start_waypoint.elevation_m,
                    ),
                    end=WaypointResponse(
                        id=day.end_waypoint.id,
                        name=day.end_waypoint.name,
                        waypoint_type=day.end_waypoint.waypoint_type,
                        latitude=day.end_waypoint.latitude,
                        longitude=day.end_waypoint.longitude,
                        has_accommodation=day.end_waypoint.has_accommodation,
                        has_water=day.end_waypoint.has_water,
                        has_food=day.end_waypoint.has_food,
                        elevation_m=day.end_waypoint.elevation_m,
                    ),
                    distance_km=day.distance_km,
                    duration_minutes=day.duration_minutes,
                    elevation_gain_m=day.elevation_gain_m,
                    elevation_loss_m=day.elevation_loss_m,
                    surface_stats=surface_stats_response,
                    # Pubs are fetched on-demand via /itineraries/pubs
                    start_pub=None,
                    end_pub=None,
                    midpoint_pub=None,
                )
            )

        response_itineraries.append(
            ItineraryResponse(
                id=it.id,
                region=it.region_name,
                days=days,
                total_distance_km=it.total_distance_km,
                total_duration_minutes=it.total_duration_minutes,
                total_elevation_gain_m=it.total_elevation_gain_m,
                score=it.score,
            )
        )

    return GenerateResponse(
        count=len(response_itineraries),
        itineraries=response_itineraries,
    )


@app.post(
    "/itineraries/pubs",
    response_model=PubsResponse,
    tags=["Itineraries"],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_itinerary_pubs(
    request: PubsRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """Get pub recommendations for an itinerary on-demand.

    This endpoint fetches pub recommendations from Google Places API for each day's
    start waypoint, end waypoint, and route midpoint. Only pubs with 4.2+ star rating
    are returned. If no pub meets the criteria, that field will be null.

    This is called separately from itinerary generation to keep generation fast.
    """
    logger.info(f"Fetching pub recommendations for {len(request.days)} days")

    # Convert request to format expected by PubRecommenderService
    days_data = [
        {
            "day_number": day.day_number,
            "start_lat": day.start_lat,
            "start_lon": day.start_lon,
            "end_lat": day.end_lat,
            "end_lon": day.end_lon,
            "connection_id": day.connection_id,
        }
        for day in request.days
    ]

    try:
        pub_service = PubRecommenderService(db=db)
        pub_recs = await pub_service.get_recommendations_for_itinerary(days_data)

        # Log what we found
        total_pubs = sum(
            (1 if r.start_pub else 0) + (1 if r.end_pub else 0) + (1 if r.midpoint_pub else 0)
            for r in pub_recs
        )
        logger.info(f"Found {total_pubs} pub recommendations across {len(pub_recs)} days")

        # Convert to response
        def _pub_to_response(pub: PubRecommendation) -> PubRecommendationResponse:
            return PubRecommendationResponse(
                name=pub.name,
                rating=pub.rating,
                latitude=pub.latitude,
                longitude=pub.longitude,
                place_id=pub.place_id,
                distance_m=pub.distance_m,
                user_ratings_total=pub.user_ratings_total,
            )

        days_response = []
        for rec in pub_recs:
            days_response.append(
                DayPubsResponse(
                    day_number=rec.day_number,
                    start_pub=_pub_to_response(rec.start_pub) if rec.start_pub else None,
                    end_pub=_pub_to_response(rec.end_pub) if rec.end_pub else None,
                    midpoint_pub=_pub_to_response(rec.midpoint_pub) if rec.midpoint_pub else None,
                )
            )

        return PubsResponse(days=days_response)

    except Exception as e:
        logger.exception(f"Failed to get pub recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch pub recommendations: {str(e)}",
        )


@app.post(
    "/itineraries/export",
    tags=["Itineraries"],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def export_itinerary_gpx(
    request: ItineraryDataRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """Export a previously generated itinerary as a GPX file.
    
    Returns a GPX file that can be loaded into GPS devices or mapping applications.
    Pass the itinerary data from a previous /generate call to get consistent results.
    """
    # Validate region exists
    region_stmt = select(Region).where(Region.name == request.region.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {request.region}",
        )

    # Load waypoints from the provided day data
    from trail_pal.algorithm.itinerary_generator import DayRoute, Itinerary
    
    days = []
    for day_data in request.days:
        # Load start waypoint
        start_wp = db.execute(
            select(Waypoint).where(Waypoint.id == day_data.start_id)
        ).scalar_one_or_none()
        if not start_wp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Start waypoint not found: {day_data.start_id}",
            )
        
        # Load end waypoint
        end_wp = db.execute(
            select(Waypoint).where(Waypoint.id == day_data.end_id)
        ).scalar_one_or_none()
        if not end_wp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"End waypoint not found: {day_data.end_id}",
            )
        
        # Find connection between waypoints
        conn_stmt = select(Connection).where(
            ((Connection.from_waypoint_id == day_data.start_id) & (Connection.to_waypoint_id == day_data.end_id)) |
            ((Connection.from_waypoint_id == day_data.end_id) & (Connection.to_waypoint_id == day_data.start_id))
        )
        connection = db.execute(conn_stmt).scalar_one_or_none()
        
        days.append(DayRoute(
            day_number=day_data.day_number,
            start_waypoint=start_wp,
            end_waypoint=end_wp,
            distance_km=connection.distance_km if connection else 0,
            duration_minutes=connection.duration_minutes if connection else 0,
            connection_id=connection.id if connection else None,
        ))
    
    # Create itinerary from provided data
    itinerary = Itinerary(
        id=request.itinerary_id,
        region_name=request.region,
        days=days,
    )

    # Generate GPX file
    gpx = gpxpy.gpx.GPX()
    gpx.name = f"{itinerary.region_name.title()} Hiking Itinerary"
    gpx.description = (
        f"{len(itinerary.days)}-day hiking route, "
        f"{itinerary.total_distance_km:.1f} km total"
    )

    # Add waypoints
    for day in itinerary.days:
        wp_start = gpxpy.gpx.GPXWaypoint(
            latitude=day.start_waypoint.latitude,
            longitude=day.start_waypoint.longitude,
            name=f"Day {day.day_number} Start: {day.start_waypoint.name}",
            description=f"Type: {day.start_waypoint.waypoint_type}",
        )
        gpx.waypoints.append(wp_start)

        if day.day_number == len(itinerary.days):
            wp_end = gpxpy.gpx.GPXWaypoint(
                latitude=day.end_waypoint.latitude,
                longitude=day.end_waypoint.longitude,
                name=f"Day {day.day_number} End: {day.end_waypoint.name}",
                description=f"Type: {day.end_waypoint.waypoint_type}",
            )
            gpx.waypoints.append(wp_end)

    # Add tracks with route geometry
    from trail_pal.services.ors_client import OpenRouteServiceClient

    # Check if we need to fetch any routes
    needs_fetch = []
    for day in itinerary.days:
        has_geometry = False
        
        if day.connection_id:
            stmt = select(Connection).where(Connection.id == day.connection_id)
            connection = db.execute(stmt).scalar_one_or_none()
            if connection and connection.route_geometry:
                has_geometry = True
        
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
        await fetch_missing_routes()

    # Create reusable ORS client for on-demand fetches when geometry is invalid
    ors_client = OpenRouteServiceClient()
    await ors_client.__aenter__()
    
    try:
        # Add tracks for each day
        for day in itinerary.days:
            track = gpxpy.gpx.GPXTrack(
                name=f"Day {day.day_number}: {day.start_waypoint.name} to {day.end_waypoint.name}"
            )
            track.description = (
                f"{day.distance_km:.1f} km, {day.duration_minutes} min"
            )

            segment = gpxpy.gpx.GPXTrackSegment()
            geometry_found = False

            # Try fetched route first
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

            # Try database connection
            if not geometry_found:
                connection = None
                
                if day.connection_id:
                    stmt = select(Connection).where(Connection.id == day.connection_id)
                    connection = db.execute(stmt).scalar_one_or_none()
                
                if not connection or not connection.route_geometry:
                    stmt = select(Connection).where(
                        (Connection.from_waypoint_id == day.end_waypoint.id) &
                        (Connection.to_waypoint_id == day.start_waypoint.id)
                    )
                    reverse_conn = db.execute(stmt).scalar_one_or_none()
                    if reverse_conn and reverse_conn.route_geometry:
                        connection = reverse_conn

                if connection and connection.route_geometry:
                    try:
                        line = to_shape(connection.route_geometry)
                        coords = list(line.coords)
                        is_reverse = connection.from_waypoint_id == day.end_waypoint.id
                        
                        if is_reverse:
                            coords = list(reversed(coords))
                        
                        # Validate that geometry endpoints match waypoints (within 0.01 degrees ~1km)
                        geometry_valid = True
                        if coords:
                            first_coord = coords[0]
                            last_coord = coords[-1]
                            start_match = (abs(first_coord[0] - day.start_waypoint.longitude) < 0.01 and 
                                         abs(first_coord[1] - day.start_waypoint.latitude) < 0.01)
                            end_match = (abs(last_coord[0] - day.end_waypoint.longitude) < 0.01 and 
                                        abs(last_coord[1] - day.end_waypoint.latitude) < 0.01)
                            geometry_valid = start_match and end_match
                        
                        # If geometry is invalid, try fetching fresh route from ORS
                        if not geometry_valid:
                            try:
                                route = await ors_client.get_hiking_route(
                                    start_lon=day.start_waypoint.longitude,
                                    start_lat=day.start_waypoint.latitude,
                                    end_lon=day.end_waypoint.longitude,
                                    end_lat=day.end_waypoint.latitude,
                                )
                                
                                if route and route.geometry:
                                    for lon, lat in route.geometry:
                                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                                            segment.points.append(
                                                gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
                                            )
                                    if segment.points:
                                        geometry_found = True
                            except Exception as e:
                                logger.warning(f"Failed to fetch on-demand route for Day {day.day_number}: {e}")
                        else:
                            # Check if coordinates might be swapped
                            first_coord = coords[0] if coords else None
                            if first_coord:
                                coord0_is_likely_lat = -90 <= first_coord[0] <= 90 and abs(first_coord[0]) > 10
                                coord1_is_likely_lon = -180 <= first_coord[1] <= 180 and abs(first_coord[1]) < 10
                                coords_swapped = coord0_is_likely_lat and coord1_is_likely_lon
                            else:
                                coords_swapped = False
                            
                            for coord in coords:
                                if coords_swapped:
                                    lat, lon = coord[0], coord[1]
                                else:
                                    lon, lat = coord[0], coord[1]
                                if -180 <= lon <= 180 and -90 <= lat <= 90:
                                    segment.points.append(
                                        gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
                                    )
                            
                            if segment.points:
                                geometry_found = True
                    except Exception as e:
                        logger.warning(f"Failed to extract geometry for Day {day.day_number}: {e}")

            # Fallback to start/end points only
            if not geometry_found:
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
        # Clean up ORS client
        try:
            await ors_client.__aexit__(None, None, None)
        except Exception:
            pass  # Ignore cleanup errors

    # Return GPX file as response
    gpx_xml = gpx.to_xml()
    return Response(
        content=gpx_xml,
        media_type="application/gpx+xml",
        headers={
            "Content-Disposition": f'attachment; filename="{request.region}_itinerary_{len(request.days)}days.gpx"'
        }
    )


@app.post(
    "/itineraries/geometry",
    response_model=ItineraryGeometryResponse,
    tags=["Itineraries"],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_itinerary_geometry(
    request: ItineraryDataRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """Get route geometry for a previously generated itinerary as JSON.
    
    Returns route geometry that can be used to display the route on a map.
    Each day includes the route path as an array of [lon, lat] coordinates.
    Pass the itinerary data from a previous /generate call to get consistent results.
    """
    # Validate region exists
    region_stmt = select(Region).where(Region.name == request.region.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {request.region}",
        )

    # Load waypoints from the provided day data
    from trail_pal.algorithm.itinerary_generator import DayRoute, Itinerary
    
    days = []
    for day_data in request.days:
        # Load start waypoint
        start_wp = db.execute(
            select(Waypoint).where(Waypoint.id == day_data.start_id)
        ).scalar_one_or_none()
        if not start_wp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Start waypoint not found: {day_data.start_id}",
            )
        
        # Load end waypoint
        end_wp = db.execute(
            select(Waypoint).where(Waypoint.id == day_data.end_id)
        ).scalar_one_or_none()
        if not end_wp:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"End waypoint not found: {day_data.end_id}",
            )
        
        # Find connection between waypoints
        conn_stmt = select(Connection).where(
            ((Connection.from_waypoint_id == day_data.start_id) & (Connection.to_waypoint_id == day_data.end_id)) |
            ((Connection.from_waypoint_id == day_data.end_id) & (Connection.to_waypoint_id == day_data.start_id))
        )
        connection = db.execute(conn_stmt).scalar_one_or_none()
        
        days.append(DayRoute(
            day_number=day_data.day_number,
            start_waypoint=start_wp,
            end_waypoint=end_wp,
            distance_km=connection.distance_km if connection else 0,
            duration_minutes=connection.duration_minutes if connection else 0,
            connection_id=connection.id if connection else None,
        ))
    
    # Create itinerary from provided data
    itinerary = Itinerary(
        id=request.itinerary_id,
        region_name=request.region,
        days=days,
    )

    # Extract geometry for each day
    from trail_pal.services.ors_client import OpenRouteServiceClient

    # Check if we need to fetch any routes
    needs_fetch = []
    for day in itinerary.days:
        has_geometry = False
        
        if day.connection_id:
            stmt = select(Connection).where(Connection.id == day.connection_id)
            connection = db.execute(stmt).scalar_one_or_none()
            if connection and connection.route_geometry:
                has_geometry = True
        
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
        await fetch_missing_routes()

    # Create reusable ORS client for on-demand fetches when geometry is invalid
    ors_client = OpenRouteServiceClient()
    await ors_client.__aenter__()
    
    try:
        # Extract geometry for each day
        day_geometries = []
        for day in itinerary.days:
            geometry = []
            geometry_found = False

            # Try fetched route first
            if day.day_number in fetched_routes:
                route = fetched_routes[day.day_number]
                if route.geometry:
                    for lon, lat in route.geometry:
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            geometry.append([lon, lat])
                    if geometry:
                        geometry_found = True

            # Try database connection
            if not geometry_found:
                connection = None
                
                if day.connection_id:
                    stmt = select(Connection).where(Connection.id == day.connection_id)
                    connection = db.execute(stmt).scalar_one_or_none()
                
                if not connection or not connection.route_geometry:
                    stmt = select(Connection).where(
                        (Connection.from_waypoint_id == day.end_waypoint.id) &
                        (Connection.to_waypoint_id == day.start_waypoint.id)
                    )
                    reverse_conn = db.execute(stmt).scalar_one_or_none()
                    if reverse_conn and reverse_conn.route_geometry:
                        connection = reverse_conn

                if connection and connection.route_geometry:
                    try:
                        line = to_shape(connection.route_geometry)
                        coords = list(line.coords)
                        is_reverse = connection.from_waypoint_id == day.end_waypoint.id
                        
                        if is_reverse:
                            coords = list(reversed(coords))
                        
                        # Validate that geometry endpoints match waypoints (within 0.01 degrees ~1km)
                        geometry_valid = True
                        if coords:
                            first_coord = coords[0]
                            last_coord = coords[-1]
                            start_match = (abs(first_coord[0] - day.start_waypoint.longitude) < 0.01 and 
                                         abs(first_coord[1] - day.start_waypoint.latitude) < 0.01)
                            end_match = (abs(last_coord[0] - day.end_waypoint.longitude) < 0.01 and 
                                        abs(last_coord[1] - day.end_waypoint.latitude) < 0.01)
                            geometry_valid = start_match and end_match
                        
                        # If geometry is invalid, try fetching fresh route from ORS
                        if not geometry_valid:
                            try:
                                route = await ors_client.get_hiking_route(
                                    start_lon=day.start_waypoint.longitude,
                                    start_lat=day.start_waypoint.latitude,
                                    end_lon=day.end_waypoint.longitude,
                                    end_lat=day.end_waypoint.latitude,
                                )
                                
                                if route and route.geometry:
                                    for lon, lat in route.geometry:
                                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                                            geometry.append([lon, lat])
                                    if geometry:
                                        geometry_found = True
                            except Exception as e:
                                logger.warning(f"Failed to fetch on-demand route for Day {day.day_number}: {e}")
                        else:
                            # Check if coordinates might be swapped
                            first_coord = coords[0] if coords else None
                            if first_coord:
                                coord0_is_likely_lat = -90 <= first_coord[0] <= 90 and abs(first_coord[0]) > 10
                                coord1_is_likely_lon = -180 <= first_coord[1] <= 180 and abs(first_coord[1]) < 10
                                coords_swapped = coord0_is_likely_lat and coord1_is_likely_lon
                            else:
                                coords_swapped = False
                            
                            for coord in coords:
                                if coords_swapped:
                                    lat, lon = coord[0], coord[1]
                                else:
                                    lon, lat = coord[0], coord[1]
                                if -180 <= lon <= 180 and -90 <= lat <= 90:
                                    geometry.append([lon, lat])
                            
                            if geometry:
                                geometry_found = True
                    except Exception as e:
                        logger.warning(f"Failed to extract geometry for Day {day.day_number}: {e}")

            # Fallback to start/end points only
            if not geometry_found:
                geometry = [
                    [day.start_waypoint.longitude, day.start_waypoint.latitude],
                    [day.end_waypoint.longitude, day.end_waypoint.latitude],
                ]

            day_geometries.append(
                DayGeometryResponse(
                    day_number=day.day_number,
                    start=WaypointResponse(
                        id=day.start_waypoint.id,
                        name=day.start_waypoint.name,
                        waypoint_type=day.start_waypoint.waypoint_type,
                        latitude=day.start_waypoint.latitude,
                        longitude=day.start_waypoint.longitude,
                        has_accommodation=day.start_waypoint.has_accommodation,
                        has_water=day.start_waypoint.has_water,
                        has_food=day.start_waypoint.has_food,
                        elevation_m=day.start_waypoint.elevation_m,
                    ),
                    end=WaypointResponse(
                        id=day.end_waypoint.id,
                        name=day.end_waypoint.name,
                        waypoint_type=day.end_waypoint.waypoint_type,
                        latitude=day.end_waypoint.latitude,
                        longitude=day.end_waypoint.longitude,
                        has_accommodation=day.end_waypoint.has_accommodation,
                        has_water=day.end_waypoint.has_water,
                        has_food=day.end_waypoint.has_food,
                        elevation_m=day.end_waypoint.elevation_m,
                    ),
                    geometry=geometry,
                )
            )
    finally:
        # Clean up ORS client
        try:
            await ors_client.__aexit__(None, None, None)
        except Exception:
            pass  # Ignore cleanup errors

    return ItineraryGeometryResponse(
        region=itinerary.region_name,
        days=day_geometries,
    )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    responses={
        400: {"model": ErrorResponse},
    },
)
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(optional_api_key),
):
    """Submit feedback for a generated route.

    Allows users to rate routes 1-5 and provide reasons for their rating.
    Feedback is stored for analysis and route improvement.
    """
    try:
        feedback = RouteFeedback(
            itinerary_id=request.itinerary_id,
            region=request.region.lower(),
            rating=request.rating,
            feedback_reasons=request.feedback_reasons,
            route_summary=request.route_summary,
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        logger.info(
            f"Feedback submitted: region={request.region}, "
            f"rating={request.rating}, reasons={request.feedback_reasons}"
        )

        return FeedbackResponse(id=feedback.id)
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to submit feedback: {str(e)}",
        )


@app.post(
    "/regions/{name}/pubs/discover",
    response_model=PubDiscoveryResponse,
    tags=["Pubs"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def discover_pubs(
    name: str,
    waypoints: bool = Query(True, description="Discover pubs near waypoints"),
    routes: bool = Query(True, description="Discover pubs near routes"),
    waypoint_radius_m: int = Query(500, ge=1, le=5000, description="Radius for waypoint search in meters"),
    route_buffer_m: int = Query(50, ge=1, le=500, description="Buffer for route search in meters"),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """Discover pubs near waypoints and/or routes in a region.

    This endpoint triggers pub discovery using Google Places API and stores results in the database.
    """
    from trail_pal.services.pub_finder import PubFinder

    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    finder = PubFinder(db=db)
    try:
        total_pubs_found = 0
        total_pubs_stored = 0
        waypoints_processed = None
        connections_processed = None

        if waypoints:
            result = await finder.find_pubs_near_waypoints(name, radius_m=waypoint_radius_m)
            total_pubs_found += result.get("pubs_found", 0)
            total_pubs_stored += result.get("pubs_stored", 0)
            waypoints_processed = result.get("waypoints_processed", 0)

        if routes:
            result = await finder.find_pubs_near_routes(name, buffer_m=route_buffer_m)
            total_pubs_found += result.get("pubs_found", 0)
            total_pubs_stored += result.get("pubs_stored", 0)
            connections_processed = result.get("connections_processed", 0)

        return PubDiscoveryResponse(
            region=name,
            waypoints_processed=waypoints_processed,
            connections_processed=connections_processed,
            pubs_found=total_pubs_found,
            pubs_stored=total_pubs_stored,
        )
    finally:
        finder.close()


@app.get(
    "/regions/{name}/waypoints/near-pubs",
    response_model=list[WaypointResponse],
    tags=["Pubs"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_waypoints_near_pubs(
    name: str,
    radius_m: int = Query(1609, ge=1, le=5000, description="Radius in meters (default 1609m = 1 mile)"),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """Get waypoints within radius_m of any pub in the region."""
    from trail_pal.services.pub_finder import PubFinder

    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    finder = PubFinder(db=db)
    try:
        waypoints = finder.get_waypoints_near_pubs(name, radius_m=radius_m)
        return [
            WaypointResponse(
                id=wp.id,
                name=wp.name,
                waypoint_type=wp.waypoint_type,
                latitude=wp.latitude,
                longitude=wp.longitude,
                has_accommodation=wp.has_accommodation,
                has_water=wp.has_water,
                has_food=wp.has_food,
                elevation_m=wp.elevation_m,
            )
            for wp in waypoints
        ]
    finally:
        finder.close()


@app.get(
    "/regions/{name}/waypoints/{waypoint_id}/pubs",
    response_model=list[PubResponse],
    tags=["Pubs"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_waypoint_pubs(
    name: str,
    waypoint_id: UUID,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """Get pubs within 500m of a specific waypoint."""
    from trail_pal.db.models import Pub

    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    # Verify waypoint exists and belongs to region
    wp_stmt = select(Waypoint).where(
        Waypoint.id == waypoint_id, Waypoint.region_id == region.id
    )
    waypoint = db.execute(wp_stmt).scalar_one_or_none()

    if not waypoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Waypoint not found: {waypoint_id}",
        )

    # Get pubs for this waypoint
    stmt = select(Pub).where(
        Pub.waypoint_id == waypoint_id, Pub.location_type == "waypoint"
    ).order_by(Pub.distance_m)
    pubs = list(db.execute(stmt).scalars().all())

    return [
        PubResponse(
            id=pub.id,
            name=pub.name,
            latitude=pub.latitude,
            longitude=pub.longitude,
            rating=pub.rating,
            user_ratings_total=pub.user_ratings_total,
            distance_m=pub.distance_m,
            location_type=pub.location_type,
            google_place_id=pub.google_place_id,
        )
        for pub in pubs
    ]


@app.get(
    "/regions/{name}/connections/{connection_id}/pubs",
    response_model=list[PubResponse],
    tags=["Pubs"],
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_connection_pubs(
    name: str,
    connection_id: UUID,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """Get pubs within 50m of a specific route segment."""
    from trail_pal.db.models import Pub

    # Find region
    region_stmt = select(Region).where(Region.name == name.lower())
    region = db.execute(region_stmt).scalar_one_or_none()

    if not region:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region not found: {name}",
        )

    # Verify connection exists and belongs to region
    wp_ids_stmt = select(Waypoint.id).where(Waypoint.region_id == region.id)
    wp_ids = list(db.execute(wp_ids_stmt).scalars().all())

    conn_stmt = select(Connection).where(
        Connection.id == connection_id,
        Connection.from_waypoint_id.in_(wp_ids),
    )
    connection = db.execute(conn_stmt).scalar_one_or_none()

    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connection not found: {connection_id}",
        )

    # Get pubs for this connection
    stmt = select(Pub).where(
        Pub.connection_id == connection_id, Pub.location_type == "route"
    ).order_by(Pub.distance_m)
    pubs = list(db.execute(stmt).scalars().all())

    return [
        PubResponse(
            id=pub.id,
            name=pub.name,
            latitude=pub.latitude,
            longitude=pub.longitude,
            rating=pub.rating,
            user_ratings_total=pub.user_ratings_total,
            distance_m=pub.distance_m,
            location_type=pub.location_type,
            google_place_id=pub.google_place_id,
        )
        for pub in pubs
    ]


# --- Entry Point ---


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "trail_pal.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
