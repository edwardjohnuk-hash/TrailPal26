# Trail Pal API Usage Guide

## Starting the API Server

First, make sure the API server is running:

```bash
# Activate your virtual environment
source venv/bin/activate

# Start the API server
python -m trail_pal.api
```

The server will start on `http://localhost:8000` by default.

You can also view the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Generating a GPX File via API

### Using cURL

```bash
# Basic request - generate a 3-day Cornwall itinerary
curl -X POST "http://localhost:8000/itineraries/export" \
  -H "Content-Type: application/json" \
  -d '{
    "region": "cornwall",
    "days": 3,
    "prefer_accommodation": true,
    "randomize": true
  }' \
  --output my_route.gpx
```

### Using Python (requests library)

```python
import requests

response = requests.post(
    "http://localhost:8000/itineraries/export",
    json={
        "region": "cornwall",
        "days": 3,
        "prefer_accommodation": True,
        "randomize": True
    }
)

if response.status_code == 200:
    with open("my_route.gpx", "wb") as f:
        f.write(response.content)
    print("GPX file saved!")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Using the Test Script

A test script is provided for convenience:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the test script
python test_api_gpx.py my_route.gpx
```

### Request Parameters

The `/itineraries/export` endpoint accepts the following JSON body:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `region` | string | Yes | - | Region name (e.g., "cornwall") |
| `days` | integer | No | 3 | Number of hiking days (1-14) |
| `start_waypoint_name` | string | No | null | Optional starting waypoint name (partial match) |
| `prefer_accommodation` | boolean | No | true | Prefer waypoints with accommodation |
| `randomize` | boolean | No | true | Return random itinerary (false for top-scored) |

### Response

On success, the API returns:
- **Status Code**: 200 OK
- **Content-Type**: `application/gpx+xml`
- **Body**: GPX file content
- **Headers**: Includes `Content-Disposition` with suggested filename

On error, the API returns:
- **Status Code**: 400, 401, 403, or 404
- **Content-Type**: `application/json`
- **Body**: `{"detail": "error message"}`

### Example: Generate a 5-day itinerary starting from a specific waypoint

```bash
curl -X POST "http://localhost:8000/itineraries/export" \
  -H "Content-Type: application/json" \
  -d '{
    "region": "cornwall",
    "days": 5,
    "start_waypoint_name": "Penzance",
    "prefer_accommodation": true,
    "randomize": false
  }' \
  --output penzance_5day.gpx
```

### Example: Generate JSON itinerary data (without GPX export)

If you want to get the itinerary data as JSON instead of GPX, use the `/itineraries/generate` endpoint:

```bash
curl -X POST "http://localhost:8000/itineraries/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "region": "cornwall",
    "days": 3,
    "max_results": 5
  }'
```

## API Authentication

If you have configured API keys in your `.env` file, you'll need to include the `X-API-Key` header:

```bash
curl -X POST "http://localhost:8000/itineraries/export" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"region": "cornwall", "days": 3}' \
  --output my_route.gpx
```

If no API keys are configured, the API allows unauthenticated access (development mode).

## Viewing the GPX File

Once you have the GPX file, you can:

1. **Open in a GPS app**: Import the file into apps like:
   - Garmin Connect
   - AllTrails
   - Gaia GPS
   - Komoot

2. **View on a map**: Upload to:
   - https://www.gpsvisualizer.com/
   - https://www.alltrails.com/ (upload GPX)
   - Google Earth (File → Open → select GPX file)

3. **Use in mapping software**:
   - QGIS
   - CalTopo
   - BaseCamp (Garmin)








