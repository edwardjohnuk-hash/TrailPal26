"""Add near_pub and nearby_pub_count columns to waypoints table.

Revision ID: 005
Revises: 004
Create Date: 2026-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add near_pub column with default False
    op.add_column(
        'waypoints',
        sa.Column('near_pub', sa.Boolean(), nullable=False, server_default='false')
    )
    # Add nearby_pub_count column with default 0
    op.add_column(
        'waypoints',
        sa.Column('nearby_pub_count', sa.Integer(), nullable=False, server_default='0')
    )


def downgrade() -> None:
    op.drop_column('waypoints', 'nearby_pub_count')
    op.drop_column('waypoints', 'near_pub')

