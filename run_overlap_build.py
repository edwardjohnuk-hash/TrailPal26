#!/usr/bin/env python3
"""Optimized script to run overlap analysis on production database.

This version loads ALL data upfront to minimize network round-trips.
"""

import logging
import sys
import os
import uuid
from collections import defaultdict
from datetime import datetime
from itertools import combinations

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def flush_print(msg):
    print(msg, flush=True)

# Buffer distance in degrees (approximately 20 meters at mid-latitudes)
BUFFER_DEGREES = 0.0002

def calculate_overlap_km(geom_a, geom_b) -> float:
    """Calculate the overlap distance between two LineString geometries."""
    try:
        buffer_a = geom_a.buffer(BUFFER_DEGREES)
        buffer_b = geom_b.buffer(BUFFER_DEGREES)
        intersection = buffer_a.intersection(buffer_b)
        
        if intersection.is_empty:
            return 0.0
        
        a_in_intersection = geom_a.intersection(buffer_b)
        b_in_intersection = geom_b.intersection(buffer_a)
        
        overlap_length_deg = 0.0
        if not a_in_intersection.is_empty:
            overlap_length_deg = max(overlap_length_deg, a_in_intersection.length)
        if not b_in_intersection.is_empty:
            overlap_length_deg = max(overlap_length_deg, b_in_intersection.length)
        
        return overlap_length_deg * 90.0  # Convert degrees to km (approx)
    except Exception as e:
        logger.warning(f"Error calculating overlap: {e}")
        return 0.0

def main():
    flush_print("=" * 60)
    flush_print("OPTIMIZED Overlap Build for Production")
    flush_print("=" * 60)
    
    flush_print("\nStep 1: Importing modules...")
    from geoalchemy2.shape import to_shape
    from sqlalchemy import select, delete, text
    from trail_pal.db.database import SessionLocal
    from trail_pal.db.models import Connection, ConnectionOverlap, Region, Waypoint
    flush_print("  Modules imported OK")
    
    flush_print("\nStep 2: Connecting to database...")
    db = SessionLocal()
    flush_print("  Connected OK")
    
    # Check region
    flush_print("\nStep 3: Finding region 'cornwall'...")
    region = db.execute(select(Region).where(Region.name == 'cornwall')).scalar_one_or_none()
    if not region:
        flush_print("  ERROR: Region 'cornwall' not found!")
        return
    flush_print(f"  Found region: {region.name} (id={region.id})")
    
    # Load all waypoints
    flush_print("\nStep 4: Loading all waypoints...")
    wp_stmt = select(Waypoint).where(Waypoint.region_id == region.id)
    waypoints = list(db.execute(wp_stmt).scalars().all())
    wp_by_id = {wp.id: wp for wp in waypoints}
    flush_print(f"  Loaded {len(waypoints)} waypoints")
    
    # Load ALL connections with geometry in a single query
    flush_print("\nStep 5: Loading all connections with geometry (this may take a minute)...")
    wp_ids = list(wp_by_id.keys())
    
    conn_stmt = select(Connection).where(
        Connection.from_waypoint_id.in_(wp_ids),
        Connection.is_feasible == True,
        Connection.route_geometry.isnot(None),
    )
    connections = list(db.execute(conn_stmt).scalars().all())
    flush_print(f"  Loaded {len(connections)} connections")
    
    # Convert all geometries to Shapely (one-time operation)
    flush_print("\nStep 6: Converting geometries to Shapely objects...")
    conn_geoms = {}
    failed = 0
    for i, conn in enumerate(connections):
        if i % 5000 == 0:
            flush_print(f"  [{i}/{len(connections)}] converting...")
        try:
            conn_geoms[conn.id] = to_shape(conn.route_geometry)
        except Exception as e:
            failed += 1
    flush_print(f"  Converted {len(conn_geoms)} geometries ({failed} failed)")
    
    # Build waypoint -> connections mapping
    flush_print("\nStep 7: Building waypoint-to-connection index...")
    wp_to_conns = defaultdict(list)
    for conn in connections:
        if conn.id in conn_geoms:
            wp_to_conns[conn.from_waypoint_id].append(conn)
            wp_to_conns[conn.to_waypoint_id].append(conn)
    flush_print(f"  Indexed {len(wp_to_conns)} waypoints with connections")
    
    # Clear existing overlaps
    flush_print("\nStep 8: Clearing existing overlap data...")
    conn_ids = list(conn_geoms.keys())
    if conn_ids:
        delete_stmt = delete(ConnectionOverlap).where(
            ConnectionOverlap.connection_a_id.in_(conn_ids) |
            ConnectionOverlap.connection_b_id.in_(conn_ids)
        )
        db.execute(delete_stmt)
        db.commit()
    flush_print("  Cleared")
    
    # Process all waypoints
    flush_print("\nStep 9: Computing overlaps...")
    flush_print("-" * 60)
    
    overlaps_created = 0
    pairs_processed = 0
    seen_pairs = set()  # Track already-processed pairs
    
    overlap_batch = []
    
    total = len(waypoints)
    for i, waypoint in enumerate(waypoints):
        if i % 100 == 0 or i == total - 1:
            flush_print(f"  [{i+1}/{total}] ({100*(i+1)/total:.1f}%) {waypoint.name} - {overlaps_created} overlaps found")
        
        conns = wp_to_conns.get(waypoint.id, [])
        if len(conns) < 2:
            continue
        
        for conn_a, conn_b in combinations(conns, 2):
            # Use consistent ordering
            if str(conn_a.id) < str(conn_b.id):
                a_id, b_id = conn_a.id, conn_b.id
            else:
                a_id, b_id = conn_b.id, conn_a.id
            
            pair_key = (a_id, b_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pairs_processed += 1
            
            geom_a = conn_geoms.get(conn_a.id)
            geom_b = conn_geoms.get(conn_b.id)
            
            if not geom_a or not geom_b:
                continue
            
            overlap_km = calculate_overlap_km(geom_a, geom_b)
            
            if overlap_km >= 0.1:
                overlap = ConnectionOverlap(
                    id=uuid.uuid4(),
                    connection_a_id=a_id,
                    connection_b_id=b_id,
                    shared_waypoint_id=waypoint.id,
                    overlap_km=overlap_km,
                    created_at=datetime.utcnow(),
                )
                overlap_batch.append(overlap)
                overlaps_created += 1
                
                # Batch insert every 500 overlaps
                if len(overlap_batch) >= 500:
                    db.add_all(overlap_batch)
                    db.commit()
                    overlap_batch = []
    
    # Final batch
    if overlap_batch:
        db.add_all(overlap_batch)
        db.commit()
    
    flush_print("-" * 60)
    flush_print("\nOverlap analysis complete!")
    flush_print(f"  Waypoints processed: {total}")
    flush_print(f"  Pairs analyzed: {pairs_processed}")
    flush_print(f"  Overlaps stored: {overlaps_created}")
    
    db.close()

if __name__ == "__main__":
    main()

