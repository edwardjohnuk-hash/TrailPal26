"""Analyze what percentage of routes contain road and street way types."""

from collections import defaultdict
from sqlalchemy import select
from trail_pal.db.database import SessionLocal
from trail_pal.db.models import Connection


def analyze_road_street_percentage():
    """Analyze the percentage of routes with road/street way types."""
    db = SessionLocal()
    
    try:
        # Get all connections with route metadata
        stmt = select(Connection)
        connections = list(db.execute(stmt).scalars().all())
        
        total_connections = len(connections)
        connections_with_metadata = 0
        connections_with_road = 0
        connections_with_street = 0
        connections_with_road_or_street = 0
        
        # Track distance breakdown
        total_distance_all = 0.0
        total_road_distance = 0.0
        total_street_distance = 0.0
        
        # Track all way types
        waytype_totals = defaultdict(float)
        waytype_counts = defaultdict(int)
        
        for conn in connections:
            if conn.route_metadata and 'surface_breakdown' in conn.route_metadata:
                connections_with_metadata += 1
                breakdown = conn.route_metadata['surface_breakdown']
                waytypes = breakdown.get('waytypes', {})
                total_distance = breakdown.get('total_distance_km', 0)
                
                total_distance_all += total_distance
                
                # Check for road and street
                road_distance = waytypes.get('road', 0)
                street_distance = waytypes.get('street', 0)
                
                if road_distance > 0:
                    connections_with_road += 1
                    total_road_distance += road_distance
                
                if street_distance > 0:
                    connections_with_street += 1
                    total_street_distance += street_distance
                
                if road_distance > 0 or street_distance > 0:
                    connections_with_road_or_street += 1
                
                # Track all way types
                for waytype, distance in waytypes.items():
                    waytype_totals[waytype] += distance
                    if distance > 0:
                        waytype_counts[waytype] += 1
        
        print(f"\n{'='*60}")
        print("ROUTE WAY TYPE ANALYSIS")
        print(f"{'='*60}\n")
        
        print(f"Total connections in database: {total_connections}")
        print(f"Connections with surface breakdown data: {connections_with_metadata}")
        print(f"Total distance analyzed: {total_distance_all:.2f} km\n")
        
        print(f"{'='*60}")
        print("ROAD & STREET ANALYSIS")
        print(f"{'='*60}\n")
        
        if connections_with_metadata > 0:
            road_pct = (connections_with_road / connections_with_metadata) * 100
            street_pct = (connections_with_street / connections_with_metadata) * 100
            road_or_street_pct = (connections_with_road_or_street / connections_with_metadata) * 100
            
            print(f"Routes containing 'road' way type:")
            print(f"  - Count: {connections_with_road} / {connections_with_metadata}")
            print(f"  - Percentage: {road_pct:.1f}%")
            print(f"  - Total road distance: {total_road_distance:.2f} km")
            if total_distance_all > 0:
                print(f"  - Road as % of total distance: {(total_road_distance/total_distance_all)*100:.1f}%")
            
            print(f"\nRoutes containing 'street' way type:")
            print(f"  - Count: {connections_with_street} / {connections_with_metadata}")
            print(f"  - Percentage: {street_pct:.1f}%")
            print(f"  - Total street distance: {total_street_distance:.2f} km")
            if total_distance_all > 0:
                print(f"  - Street as % of total distance: {(total_street_distance/total_distance_all)*100:.1f}%")
            
            print(f"\nRoutes containing EITHER road OR street:")
            print(f"  - Count: {connections_with_road_or_street} / {connections_with_metadata}")
            print(f"  - Percentage: {road_or_street_pct:.1f}%")
            
            combined_distance = total_road_distance + total_street_distance
            if total_distance_all > 0:
                print(f"  - Combined road+street distance: {combined_distance:.2f} km")
                print(f"  - Combined as % of total distance: {(combined_distance/total_distance_all)*100:.1f}%")
        
        print(f"\n{'='*60}")
        print("ALL WAY TYPES BREAKDOWN")
        print(f"{'='*60}\n")
        
        # Sort by total distance
        sorted_waytypes = sorted(waytype_totals.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Way Type':<20} {'Distance (km)':<15} {'% of Total':<12} {'# Routes':<10}")
        print("-" * 60)
        
        for waytype, distance in sorted_waytypes:
            pct = (distance / total_distance_all * 100) if total_distance_all > 0 else 0
            count = waytype_counts[waytype]
            count_pct = (count / connections_with_metadata * 100) if connections_with_metadata > 0 else 0
            print(f"{waytype:<20} {distance:>10.2f} km   {pct:>8.1f}%     {count:>5} ({count_pct:.1f}%)")
        
        print(f"\n{'='*60}")
        
    finally:
        db.close()


if __name__ == "__main__":
    analyze_road_street_percentage()


