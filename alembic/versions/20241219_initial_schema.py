"""Initial schema with regions, waypoints, and connections.

Revision ID: 001
Revises:
Create Date: 2024-12-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure PostGIS extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # Create regions table
    op.create_table(
        "regions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("country", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "bounds",
            geoalchemy2.Geometry("POLYGON", srid=4326, spatial_index=True),
            nullable=False,
        ),
        sa.Column("last_synced", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )

    # Create waypoints table
    op.create_table(
        "waypoints",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "region_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("regions.id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("waypoint_type", sa.String(50), nullable=False),
        sa.Column(
            "location",
            geoalchemy2.Geometry("POINT", srid=4326, spatial_index=True),
            nullable=False,
        ),
        sa.Column("latitude", sa.Float, nullable=False),
        sa.Column("longitude", sa.Float, nullable=False),
        sa.Column("osm_id", sa.String(50), nullable=True),
        sa.Column("osm_type", sa.String(10), nullable=True),
        sa.Column("amenities", postgresql.JSONB, nullable=True),
        sa.Column("elevation_m", sa.Float, nullable=True),
        sa.Column("has_accommodation", sa.Boolean, default=False),
        sa.Column("has_water", sa.Boolean, default=False),
        sa.Column("has_food", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("osm_id", "osm_type", name="uq_waypoint_osm"),
    )

    # Create index on waypoint type
    op.create_index("ix_waypoints_type", "waypoints", ["waypoint_type"])

    # Create connections table
    op.create_table(
        "connections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "from_waypoint_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("waypoints.id"),
            nullable=False,
        ),
        sa.Column(
            "to_waypoint_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("waypoints.id"),
            nullable=False,
        ),
        sa.Column("distance_km", sa.Float, nullable=False),
        sa.Column("duration_minutes", sa.Integer, nullable=False),
        sa.Column("elevation_gain_m", sa.Float, nullable=True),
        sa.Column("elevation_loss_m", sa.Float, nullable=True),
        sa.Column(
            "route_geometry",
            geoalchemy2.Geometry("LINESTRING", srid=4326),
            nullable=True,
        ),
        sa.Column("route_metadata", postgresql.JSONB, nullable=True),
        sa.Column("is_feasible", sa.Boolean, default=False),
        sa.Column("straight_line_distance_km", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint(
            "from_waypoint_id", "to_waypoint_id", name="uq_connection_waypoints"
        ),
    )

    # Create indexes for efficient graph traversal
    op.create_index("ix_connections_from", "connections", ["from_waypoint_id"])
    op.create_index("ix_connections_to", "connections", ["to_waypoint_id"])
    op.create_index("ix_connections_feasible", "connections", ["is_feasible"])


def downgrade() -> None:
    op.drop_table("connections")
    op.drop_table("waypoints")
    op.drop_table("regions")

