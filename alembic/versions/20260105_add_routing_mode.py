"""Add routing_mode column to regions table.

Revision ID: 20260105_add_routing_mode
Revises: 20260104_fix_unnamed_waypoints
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '008'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add routing_mode column with default 'precomputed' for existing regions
    op.add_column(
        'regions',
        sa.Column(
            'routing_mode',
            sa.String(50),
            nullable=False,
            server_default='precomputed'
        )
    )


def downgrade() -> None:
    op.drop_column('regions', 'routing_mode')

