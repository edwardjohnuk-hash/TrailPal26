# Trail Pal

A hiking itinerary generator for multi-day trails. Currently focused on Cornwall, England, with plans to expand to other regions.

## Features

- Generate 3-day hiking itineraries with 10-20km daily distances
- Waypoints include campsites, hostels, and points of interest
- Routes follow trails and footpaths, avoiding roads
- Pre-computed feasibility graph for instant itinerary generation

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL with PostGIS extension
- OpenRouteService API key (free tier available)

### Setup

1. Clone the repository and navigate to the project directory

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

3. Set up PostgreSQL with PostGIS:
```sql
CREATE DATABASE trail_pal;
\c trail_pal
CREATE EXTENSION postgis;
```

4. Copy `.env.example` to `.env` and configure your settings:
```bash
cp .env.example .env
# Edit .env with your database credentials and API keys
```

5. Run database migrations:
```bash
alembic upgrade head
```

## Usage

### Seed Waypoints

Fetch campsites, hostels, and POIs from OpenStreetMap:

```bash
trail-pal seed --region cornwall
```

### Build Feasibility Graph

Compute valid hiking connections between waypoints:

```bash
trail-pal build-graph --region cornwall
```

### Generate Itinerary

Create a 3-day hiking itinerary:

```bash
trail-pal generate --region cornwall --days 3
trail-pal generate --start "Penzance" --days 3
```

### Export Route

Export an itinerary to GPX format:

```bash
trail-pal export --itinerary-id <id> --format gpx --output route.gpx
```

## Project Structure

```
trail_pal/
├── trail_pal/
│   ├── config.py           # Settings & API configuration
│   ├── db/
│   │   ├── database.py     # SQLAlchemy + PostGIS setup
│   │   └── models.py       # Region, Waypoint, Connection models
│   ├── services/
│   │   ├── osm_client.py   # Overpass API client
│   │   ├── ors_client.py   # OpenRouteService client
│   │   ├── waypoint_seeder.py
│   │   └── graph_builder.py
│   ├── algorithm/
│   │   └── itinerary_generator.py
│   └── cli.py              # Command-line interface
├── alembic/                # Database migrations
└── tests/
```

## API Rate Limits

| API | Free Tier Limits | Notes |
|-----|-----------------|-------|
| Overpass (OSM) | Fair use | Cached locally |
| OpenRouteService | 40 req/min, 2000/day | Pre-computed graph |

## License

MIT License

