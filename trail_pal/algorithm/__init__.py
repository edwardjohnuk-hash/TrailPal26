"""Algorithm module for Trail Pal."""

from trail_pal.algorithm.itinerary_generator import (
    DayRoute,
    Itinerary,
    ItineraryGenerator,
    ItineraryOptions,
    generate_itineraries,
)
from trail_pal.algorithm.onthefly_generator import (
    OnTheFlyGenerator,
    generate_itineraries_onthefly,
)

__all__ = [
    "DayRoute",
    "Itinerary",
    "ItineraryGenerator",
    "ItineraryOptions",
    "generate_itineraries",
    "OnTheFlyGenerator",
    "generate_itineraries_onthefly",
]

