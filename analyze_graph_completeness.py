#!/usr/bin/env python3
"""Analyze graph completeness and estimate time to 100% completion."""

import sys
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection, Region, Waypoint, WaypointType
from trail_pal.services.graph_builder import GraphBuilder
from trail_pal.services.ors_client import calculate_straight_line_distance_km
from sqlalchemy import func, select


def calculate_candidate_pairs(accommodation_waypoints, transport_waypoints, max_distance_km):
    """Calculate total candidate pairs that need to be processed."""
    candidates = []
    
    # Transport to Accommodation
    for transport_wp in transport_waypoints:
        for accommodation_wp in accommodation_waypoints:
            distance = calculate_straight_line_distance_km(
                transport_wp.longitude, transport_wp.latitude,
                accommodation_wp.longitude, accommodation_wp.latitude
            )
            if distance <= max_distance_km:
                candidates.append((transport_wp.id, accommodation_wp.id))
    
    # Accommodation to Accommodation
    from itertools import combinations
    for wp1, wp2 in combinations(accommodation_waypoints, 2):
        distance = calculate_straight_line_distance_km(
            wp1.longitude, wp1.latitude,
            wp2.longitude, wp2.latitude
        )
        if distance <= max_distance_km:
            candidates.append((wp1.id, wp2.id))
    
    # Accommodation to Transport
    for accommodation_wp in accommodation_waypoints:
        for transport_wp in transport_waypoints:
            distance = calculate_straight_line_distance_km(
                accommodation_wp.longitude, accommodation_wp.latitude,
                transport_wp.longitude, transport_wp.latitude
            )
            if distance <= max_distance_km:
                candidates.append((accommodation_wp.id, transport_wp.id))
    
    return candidates


def count_existing_connections(db, waypoint_ids):
    """Count existing connections between waypoints."""
    stmt = select(Connection).where(
        Connection.from_waypoint_id.in_(waypoint_ids)
    )
    connections = list(db.execute(stmt).scalars().all())
    
    # Create a set of (from_id, to_id) pairs (normalized)
    existing_pairs = set()
    for conn in connections:
        # Normalize: always store as (min_id, max_id) to handle bidirectional
        pair = tuple(sorted([conn.from_waypoint_id, conn.to_waypoint_id]))
        existing_pairs.add(pair)
    
    return len(existing_pairs), len(connections)


def analyze_region(region_name: str, daily_api_quota: int = 4000):
    """Analyze graph completeness for a region."""
    db = SessionLocal()
    builder = GraphBuilder(db=db)
    
    try:
        # Get region
        region = builder.get_region(region_name)
        if not region:
            print(f"Error: Region '{region_name}' not found")
            return
        
        # Get waypoints
        accommodation_waypoints = builder.get_region_waypoints(region.id, accommodation_only=True)
        transport_waypoints = builder.get_transport_waypoints(region.id)
        
        print(f"\n{'='*60}")
        print(f"Graph Completeness Analysis: {region_name.upper()}")
        print(f"{'='*60}\n")
        
        print(f"Waypoints:")
        print(f"  - Accommodation waypoints: {len(accommodation_waypoints)}")
        print(f"  - Transport waypoints: {len(transport_waypoints)}")
        print(f"  - Total waypoints: {len(accommodation_waypoints) + len(transport_waypoints)}")
        
        # Get all waypoint IDs
        all_waypoint_ids = [wp.id for wp in accommodation_waypoints + transport_waypoints]
        
        # Count existing connections
        unique_connections, total_connection_records = count_existing_connections(db, all_waypoint_ids)
        
        # Calculate candidate pairs
        max_distance = builder._settings.max_straight_line_distance_km
        candidate_pairs = calculate_candidate_pairs(
            accommodation_waypoints, 
            transport_waypoints, 
            max_distance
        )
        
        # Normalize candidate pairs to match connection storage
        normalized_candidates = set()
        for wp1_id, wp2_id in candidate_pairs:
            normalized_candidates.add(tuple(sorted([wp1_id, wp2_id])))
        
        total_candidates = len(normalized_candidates)
        remaining_candidates = total_candidates - unique_connections
        
        # Calculate completion percentage
        completion_pct = (unique_connections / total_candidates * 100) if total_candidates > 0 else 0
        
        print(f"\nConnection Statistics:")
        print(f"  - Total candidate pairs: {total_candidates:,}")
        print(f"  - Existing connections: {unique_connections:,}")
        print(f"  - Remaining connections: {remaining_candidates:,}")
        print(f"  - Completion: {completion_pct:.1f}%")
        print(f"  - Total connection records: {total_connection_records:,}")
        
        # Get feasible connections
        feasible_stmt = select(func.count(Connection.id)).where(
            Connection.from_waypoint_id.in_(all_waypoint_ids),
            Connection.is_feasible == True  # noqa: E712
        )
        feasible_count = db.execute(feasible_stmt).scalar() or 0
        
        print(f"  - Feasible connections: {feasible_count:,}")
        
        # Calculate time to completion
        if remaining_candidates > 0:
            days_to_complete = remaining_candidates / daily_api_quota
            
            # Calculate per-minute rate limit impact
            requests_per_minute = builder._settings.ors_requests_per_minute
            max_per_day_by_rate_limit = requests_per_minute * 60 * 24  # requests/min * 60 min/hr * 24 hr/day
            actual_daily_capacity = min(daily_api_quota, max_per_day_by_rate_limit)
            
            # Time if we use full daily quota
            days_at_full_quota = remaining_candidates / daily_api_quota
            
            # Time if limited by per-minute rate
            days_at_rate_limit = remaining_candidates / max_per_day_by_rate_limit if max_per_day_by_rate_limit > 0 else float('inf')
            
            print(f"\n{'='*60}")
            print(f"Time to 100% Completion:")
            print(f"{'='*60}")
            print(f"  - Daily API quota: {daily_api_quota:,} requests/day")
            print(f"  - Per-minute rate limit: {requests_per_minute} requests/min")
            print(f"  - Max capacity (rate limit): {max_per_day_by_rate_limit:,} requests/day")
            print(f"  - Effective daily capacity: {actual_daily_capacity:,} requests/day")
            print(f"\n  - Days needed (at full quota): {days_at_full_quota:.1f}")
            print(f"  - Weeks needed: {days_at_full_quota / 7:.1f}")
            print(f"  - Months needed: {days_at_full_quota / 30:.1f}")
            
            if max_per_day_by_rate_limit < daily_api_quota:
                print(f"\n  ⚠️  WARNING: Per-minute rate limit ({requests_per_minute}/min) restricts")
                print(f"     daily capacity to {max_per_day_by_rate_limit:,} requests/day.")
                print(f"     Actual time needed: {days_at_rate_limit:.1f} days")
            
            print(f"\nNote: Assumes 1 API call per connection.")
            print(f"      Actual time may vary based on API failures and retries.")
        else:
            print(f"\n{'='*60}")
            print("Graph is 100% complete!")
            print(f"{'='*60}")
        
        # Show breakdown by connection type
        print(f"\n{'='*60}")
        print("Connection Type Breakdown:")
        print(f"{'='*60}")
        
        # Transport -> Accommodation
        transport_to_acc = 0
        for transport_wp in transport_waypoints:
            for accommodation_wp in accommodation_waypoints:
                distance = calculate_straight_line_distance_km(
                    transport_wp.longitude, transport_wp.latitude,
                    accommodation_wp.longitude, accommodation_wp.latitude
                )
                if distance <= max_distance:
                    pair = tuple(sorted([transport_wp.id, accommodation_wp.id]))
                    if pair in normalized_candidates:
                        # Check if exists
                        stmt = select(Connection).where(
                            ((Connection.from_waypoint_id == transport_wp.id) & 
                             (Connection.to_waypoint_id == accommodation_wp.id)) |
                            ((Connection.from_waypoint_id == accommodation_wp.id) & 
                             (Connection.to_waypoint_id == transport_wp.id))
                        )
                        if db.execute(stmt).first():
                            transport_to_acc += 1
        
        # Accommodation -> Accommodation
        from itertools import combinations
        acc_to_acc = 0
        for wp1, wp2 in combinations(accommodation_waypoints, 2):
            distance = calculate_straight_line_distance_km(
                wp1.longitude, wp1.latitude,
                wp2.longitude, wp2.latitude
            )
            if distance <= max_distance:
                pair = tuple(sorted([wp1.id, wp2.id]))
                if pair in normalized_candidates:
                    stmt = select(Connection).where(
                        ((Connection.from_waypoint_id == wp1.id) & 
                         (Connection.to_waypoint_id == wp2.id)) |
                        ((Connection.from_waypoint_id == wp2.id) & 
                         (Connection.to_waypoint_id == wp1.id))
                    )
                    if db.execute(stmt).first():
                        acc_to_acc += 1
        
        # Accommodation -> Transport (same as Transport -> Accommodation, so skip)
        
        total_transport_acc_pairs = len(transport_waypoints) * len(accommodation_waypoints)
        total_acc_acc_pairs = len(list(combinations(accommodation_waypoints, 2)))
        
        # Count pairs within distance
        transport_acc_within = sum(1 for t in transport_waypoints for a in accommodation_waypoints 
                                   if calculate_straight_line_distance_km(t.longitude, t.latitude, a.longitude, a.latitude) <= max_distance)
        acc_acc_within = len([p for p in combinations(accommodation_waypoints, 2) 
                              if calculate_straight_line_distance_km(p[0].longitude, p[0].latitude, p[1].longitude, p[1].latitude) <= max_distance])
        
        print(f"  - Transport ↔ Accommodation pairs: {total_transport_acc_pairs:,} (within distance: {transport_acc_within:,})")
        print(f"  - Accommodation ↔ Accommodation pairs: {total_acc_acc_pairs:,} (within distance: {acc_acc_within:,})")
        
    finally:
        builder.close()
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_graph_completeness.py <region_name> [daily_api_quota]")
        print("Example: python analyze_graph_completeness.py cornwall 4000")
        sys.exit(1)
    
    region_name = sys.argv[1]
    daily_quota = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    
    analyze_region(region_name, daily_quota)

