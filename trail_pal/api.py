"""FastAPI application for Trail Pal REST API."""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trail_pal.algorithm.itinerary_generator import (
    ItineraryGenerator,
    ItineraryOptions,
)
from trail_pal.config import get_settings
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Region, Waypoint

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
    total_waypoints: int
    total_connections: int
    feasible_connections: int
    avg_distance_km: Optional[float] = None
    avg_duration_min: Optional[float] = None
    waypoints_by_type: dict[str, int]


class DayRouteResponse(BaseModel):
    """A single day's hiking route."""

    day_number: int
    start: WaypointResponse
    end: WaypointResponse
    distance_km: float
    duration_minutes: int
    elevation_gain_m: Optional[float] = None
    elevation_loss_m: Optional[float] = None


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


class GenerateResponse(BaseModel):
    """Response containing generated itineraries."""

    count: int
    itineraries: list[ItineraryResponse]


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str


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


# --- Routes ---


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse()


@app.get(
    "/regions",
    response_model=list[RegionResponse],
    tags=["Regions"],
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
async def list_regions(
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """List all available hiking regions."""
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
    _api_key: str = Depends(verify_api_key),
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
    )

    # Generate itineraries
    try:
        generator = ItineraryGenerator(db=db)
        itineraries = generator.generate(request.region, options)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not itineraries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No itineraries found. Ensure waypoints and connections exist for this region.",
        )

    # Convert to response models
    response_itineraries = []
    for it in itineraries:
        days = []
        for day in it.days:
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

