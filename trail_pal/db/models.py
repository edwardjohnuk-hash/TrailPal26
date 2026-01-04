"""SQLAlchemy models for Trail Pal."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trail_pal.db.database import Base


class Region(Base):
    """A geographic region for hiking itineraries (e.g., Cornwall, England)."""

    __tablename__ = "regions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    country: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Bounding box as a polygon geometry
    bounds = mapped_column(Geometry("POLYGON", srid=4326), nullable=False)

    # Tracking when data was last synced
    last_synced: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    waypoints: Mapped[list["Waypoint"]] = relationship(
        "Waypoint", back_populates="region", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Region(name='{self.name}', country='{self.country}')>"


class WaypointType:
    """Constants for waypoint types."""

    CAMPSITE = "campsite"
    HOSTEL = "hostel"
    GUEST_HOUSE = "guest_house"
    HOTEL = "hotel"
    VIEWPOINT = "viewpoint"
    PEAK = "peak"
    VILLAGE = "village"
    TRAIN_STATION = "train_station"
    TOWN = "town"
    CITY = "city"


class Waypoint(Base):
    """A point of interest that can serve as a start/end point for a hiking day."""

    __tablename__ = "waypoints"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    region_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regions.id"), nullable=False
    )

    # Basic information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    waypoint_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Location as a point geometry (EPSG:4326 = WGS84)
    location = mapped_column(Geometry("POINT", srid=4326), nullable=False)

    # Latitude and longitude stored separately for easy access
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    # OSM reference
    osm_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    osm_type: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # node, way, relation

    # Additional metadata
    amenities: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    elevation_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Quality indicators
    has_accommodation: Mapped[bool] = mapped_column(Boolean, default=False)
    has_water: Mapped[bool] = mapped_column(Boolean, default=False)
    has_food: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    region: Mapped["Region"] = relationship("Region", back_populates="waypoints")
    connections_from: Mapped[list["Connection"]] = relationship(
        "Connection",
        foreign_keys="Connection.from_waypoint_id",
        back_populates="from_waypoint",
        cascade="all, delete-orphan",
    )
    connections_to: Mapped[list["Connection"]] = relationship(
        "Connection",
        foreign_keys="Connection.to_waypoint_id",
        back_populates="to_waypoint",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("osm_id", "osm_type", name="uq_waypoint_osm"),
    )

    def __repr__(self) -> str:
        return f"<Waypoint(name='{self.name}', type='{self.waypoint_type}')>"


class Connection(Base):
    """A hiking route connection between two waypoints."""

    __tablename__ = "connections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    from_waypoint_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("waypoints.id"), nullable=False
    )
    to_waypoint_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("waypoints.id"), nullable=False
    )

    # Route metrics
    distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    elevation_gain_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_loss_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Route geometry (LineString for the path)
    route_geometry = mapped_column(Geometry("LINESTRING", srid=4326), nullable=True)

    # Additional route metadata from ORS
    route_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Feasibility flag (True if within 10-20km constraint)
    is_feasible: Mapped[bool] = mapped_column(Boolean, default=False)

    # Straight-line distance for quick filtering
    straight_line_distance_km: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    from_waypoint: Mapped["Waypoint"] = relationship(
        "Waypoint", foreign_keys=[from_waypoint_id], back_populates="connections_from"
    )
    to_waypoint: Mapped["Waypoint"] = relationship(
        "Waypoint", foreign_keys=[to_waypoint_id], back_populates="connections_to"
    )

    __table_args__ = (
        UniqueConstraint(
            "from_waypoint_id", "to_waypoint_id", name="uq_connection_waypoints"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Connection(from='{self.from_waypoint_id}', "
            f"to='{self.to_waypoint_id}', distance={self.distance_km}km)>"
        )


class ConnectionOverlap(Base):
    """Pre-computed overlap between two connections that share a waypoint.
    
    Used to quickly filter out itineraries where consecutive days
    share significant trail geometry (backtracking).
    """

    __tablename__ = "connection_overlaps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    connection_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("connections.id", ondelete="CASCADE"), nullable=False
    )
    connection_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("connections.id", ondelete="CASCADE"), nullable=False
    )
    shared_waypoint_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("waypoints.id", ondelete="CASCADE"), nullable=False
    )
    
    # The length of overlapping geometry in kilometers
    overlap_km: Mapped[float] = mapped_column(Float, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    connection_a: Mapped["Connection"] = relationship(
        "Connection", foreign_keys=[connection_a_id]
    )
    connection_b: Mapped["Connection"] = relationship(
        "Connection", foreign_keys=[connection_b_id]
    )
    shared_waypoint: Mapped["Waypoint"] = relationship(
        "Waypoint", foreign_keys=[shared_waypoint_id]
    )

    __table_args__ = (
        UniqueConstraint(
            "connection_a_id", "connection_b_id", name="uq_connection_overlap_pair"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ConnectionOverlap(a={self.connection_a_id}, "
            f"b={self.connection_b_id}, overlap={self.overlap_km:.2f}km)>"
        )


class RouteFeedback(Base):
    """User feedback and rating for a generated route."""

    __tablename__ = "route_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    itinerary_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    region: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Rating from 1-5
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Selected feedback reasons as a JSON array
    feedback_reasons: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    
    # Route summary with details (days, distance, waypoints, etc.)
    route_summary: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<RouteFeedback(itinerary_id={self.itinerary_id}, rating={self.rating})>"


class Pub(Base):
    """A pub found near a waypoint or route segment."""

    __tablename__ = "pubs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    waypoint_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("waypoints.id", ondelete="CASCADE"), nullable=True
    )
    connection_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("connections.id", ondelete="CASCADE"), nullable=True
    )

    # Google Places data
    google_place_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Location as a point geometry (EPSG:4326 = WGS84)
    location = mapped_column(Geometry("POINT", srid=4326), nullable=False)

    # Latitude and longitude stored separately for easy access
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    # Rating and metadata
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    user_ratings_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    location_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'waypoint' or 'route'
    pub_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    waypoint: Mapped[Optional["Waypoint"]] = relationship("Waypoint", foreign_keys=[waypoint_id])
    connection: Mapped[Optional["Connection"]] = relationship("Connection", foreign_keys=[connection_id])

    def __repr__(self) -> str:
        return f"<Pub(name='{self.name}', rating={self.rating}, distance={self.distance_m}m)>"

