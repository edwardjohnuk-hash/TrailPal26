"""Add connection_overlaps table for pre-computed route overlap detection.

Revision ID: 003
Revises: 002
Create Date: 2026-01-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create connection_overlaps table to store pre-computed overlap distances
    # between connection pairs that share a waypoint
    op.create_table(
        "connection_overlaps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "connection_a_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "connection_b_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("connections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "shared_waypoint_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("waypoints.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("overlap_km", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    # Create indexes for efficient lookups during itinerary generation
    op.create_index(
        "ix_connection_overlaps_connection_a",
        "connection_overlaps",
        ["connection_a_id"],
    )
    op.create_index(
        "ix_connection_overlaps_connection_b",
        "connection_overlaps",
        ["connection_b_id"],
    )
    op.create_index(
        "ix_connection_overlaps_shared_waypoint",
        "connection_overlaps",
        ["shared_waypoint_id"],
    )
    # Composite index for fast pair lookups (both directions)
    op.create_index(
        "ix_connection_overlaps_pair",
        "connection_overlaps",
        ["connection_a_id", "connection_b_id"],
    )
    # Unique constraint to prevent duplicate pairs
    op.create_unique_constraint(
        "uq_connection_overlap_pair",
        "connection_overlaps",
        ["connection_a_id", "connection_b_id"],
    )


def downgrade() -> None:
    op.drop_table("connection_overlaps")

