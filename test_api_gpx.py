#!/usr/bin/env python3
"""Example script to call the Trail Pal API and save a GPX file."""

import requests
import sys

# API configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = None  # Set this if you have API keys configured, or leave None if no auth required

def generate_gpx(region="cornwall", days=3, start_waypoint=None, output_file="itinerary.gpx"):
    """Generate a GPX file from the API and save it to disk."""
    
    url = f"{API_BASE_URL}/itineraries/export"
    
    # Prepare request body
    payload = {
        "region": region,
        "days": days,
        "prefer_accommodation": True,
        "randomize": True,
    }
    
    if start_waypoint:
        payload["start_waypoint_name"] = start_waypoint
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    print(f"Calling API: {url}")
    print(f"Request: {payload}")
    
    try:
        # Make the API call
        response = requests.post(url, json=payload, headers=headers)
        
        # Check for errors
        if response.status_code == 200:
            # Save the GPX file
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✓ Successfully saved GPX file to: {output_file}")
            print(f"  File size: {len(response.content)} bytes")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API server.")
        print("  Make sure the API server is running:")
        print("  python -m trail_pal.api")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "itinerary.gpx"
    
    # Generate a 3-day Cornwall itinerary
    success = generate_gpx(
        region="cornwall",
        days=3,
        output_file=output_file
    )
    
    if success:
        print(f"\nYou can now open {output_file} in a GPS app or mapping software!")
    else:
        sys.exit(1)








