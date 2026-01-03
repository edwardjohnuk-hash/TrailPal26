"""Add pubs table for Google Places integration.

Revision ID: 002
Revises: 001
Create Date: 2024-12-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create pubs table
    op.create_table(
        "pubs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "waypoint_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("waypoints.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "connection_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connections.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("google_place_id", sa.String(255), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "location",
            geoalchemy2.Geometry("POINT", srid=4326, spatial_index=True),
            nullable=False,
        ),
        sa.Column("latitude", sa.Float, nullable=False),
        sa.Column("longitude", sa.Float, nullable=False),
        sa.Column("rating", sa.Float, nullable=False),
        sa.Column("user_ratings_total", sa.Integer, nullable=True),
        sa.Column("distance_m", sa.Float, nullable=False),
        sa.Column("location_type", sa.String(50), nullable=False),  # 'waypoint' or 'route'
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )

    # Create indexes for efficient queries
    op.create_index("ix_pubs_waypoint_id", "pubs", ["waypoint_id"])
    op.create_index("ix_pubs_connection_id", "pubs", ["connection_id"])
    op.create_index("ix_pubs_location_type", "pubs", ["location_type"])
    op.create_index("ix_pubs_rating", "pubs", ["rating"])
    # Ensure at least one of waypoint_id or connection_id is set
    op.create_check_constraint(
        "ck_pubs_has_reference",
        "pubs",
        "waypoint_id IS NOT NULL OR connection_id IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_table("pubs")





