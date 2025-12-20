"""Tests for Trail Pal configuration."""

import pytest


def test_settings_defaults():
    """Test that settings have sensible defaults."""
    from trail_pal.config import get_settings

    settings = get_settings()

    assert settings.min_daily_distance_km == 10.0
    assert settings.max_daily_distance_km == 20.0
    assert settings.default_trip_days == 3
    assert settings.ors_requests_per_minute == 40


def test_waypoint_types():
    """Test waypoint type constants."""
    from trail_pal.db.models import WaypointType

    assert WaypointType.CAMPSITE == "campsite"
    assert WaypointType.HOSTEL == "hostel"
    assert WaypointType.GUEST_HOUSE == "guest_house"
    assert WaypointType.VIEWPOINT == "viewpoint"
    assert WaypointType.PEAK == "peak"


def test_region_bounds():
    """Test that Cornwall region bounds are defined."""
    from trail_pal.services.osm_client import REGION_BOUNDS, get_region_bounds

    assert "cornwall" in REGION_BOUNDS

    bounds = get_region_bounds("cornwall")
    assert bounds.south < bounds.north
    assert bounds.west < bounds.east
    assert 49.0 < bounds.south < 51.0  # Cornwall is around 50°N
    assert -6.0 < bounds.west < -4.0  # Cornwall is around -5°W


def test_straight_line_distance():
    """Test Haversine distance calculation."""
    from trail_pal.services.ors_client import calculate_straight_line_distance_km

    # Test with known coordinates (approximately)
    # London to Paris is roughly 344 km
    distance = calculate_straight_line_distance_km(
        lon1=-0.1278, lat1=51.5074,  # London
        lon2=2.3522, lat2=48.8566,  # Paris
    )
    assert 340 < distance < 350


def test_available_regions():
    """Test that available regions are listed."""
    from trail_pal.services.waypoint_seeder import list_available_regions

    regions = list_available_regions()
    assert "cornwall" in regions

