"""Update unnamed accommodation waypoints with nearest settlement name.

Revision ID: 006
Revises: 005
Create Date: 2026-01-04

"""
from typing import Sequence, Union
import math

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def upgrade() -> None:
    conn = op.get_bind()
    
    # Get unnamed accommodations
    unnamed = conn.execute(
        text("""
            SELECT id, name, waypoint_type, latitude, longitude 
            FROM waypoints 
            WHERE name LIKE 'Unnamed %' 
            AND waypoint_type IN ('campsite', 'hostel', 'guest_house', 'hotel')
            AND name NOT LIKE '%,%'
        """)
    ).fetchall()
    
    # Get settlements
    settlements = conn.execute(
        text("""
            SELECT name, latitude, longitude 
            FROM waypoints 
            WHERE waypoint_type IN ('village', 'town', 'city')
        """)
    ).fetchall()
    
    print(f"Found {len(unnamed)} unnamed accommodations and {len(settlements)} settlements")
    
    # Update each unnamed accommodation with nearest settlement
    for wp_id, wp_name, wp_type, wp_lat, wp_lon in unnamed:
        nearest_name = None
        min_dist = float('inf')
        
        for s_name, s_lat, s_lon in settlements:
            dist = haversine_km(wp_lat, wp_lon, s_lat, s_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_name = s_name
        
        if nearest_name and min_dist < 10:  # Only if within 10km
            type_display = wp_type.replace("_", " ")
            new_name = f"Unnamed {type_display}, {nearest_name}"
            conn.execute(
                text("UPDATE waypoints SET name = :name WHERE id = :id"),
                {"name": new_name, "id": wp_id}
            )
            print(f"  Updated: {wp_name} -> {new_name}")


def downgrade() -> None:
    # Remove location suffix from unnamed waypoints
    conn = op.get_bind()
    conn.execute(
        text("""
            UPDATE waypoints 
            SET name = REGEXP_REPLACE(name, ', [^,]+$', '')
            WHERE name LIKE 'Unnamed %,%'
        """)
    )

