"""Add route_feedback table for storing user ratings and feedback.

Revision ID: 004
Revises: 003
Create Date: 2026-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'route_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('itinerary_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('region', sa.String(255), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('feedback_reasons', postgresql.JSONB(), nullable=True),
        sa.Column('route_summary', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    # Index on region for querying feedback by region
    op.create_index('ix_route_feedback_region', 'route_feedback', ['region'])
    # Index on rating for analytics
    op.create_index('ix_route_feedback_rating', 'route_feedback', ['rating'])
    # Index on created_at for time-based queries
    op.create_index('ix_route_feedback_created_at', 'route_feedback', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_route_feedback_created_at', table_name='route_feedback')
    op.drop_index('ix_route_feedback_rating', table_name='route_feedback')
    op.drop_index('ix_route_feedback_region', table_name='route_feedback')
    op.drop_table('route_feedback')

