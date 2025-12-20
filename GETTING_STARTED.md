# Getting Started with Trail Pal (Beginner's Guide)

This guide assumes you have no programming experience. Follow each step carefully.

---

## Step 1: Install Homebrew (Mac's Package Manager)

Homebrew helps you install software on your Mac. Open the **Terminal** app:
- Press `Cmd + Space`, type "Terminal", and press Enter

Copy and paste this entire command into Terminal, then press Enter:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow any instructions it gives you. This may take a few minutes.

---

## Step 2: Install Python 3.11

In Terminal, run:

```bash
brew install python@3.11
```

Verify it worked:
```bash
python3.11 --version
```

You should see something like "Python 3.11.x"

---

## Step 3: Install PostgreSQL with PostGIS

PostGIS is a database that can store geographic locations.

```bash
brew install postgresql@15 postgis
```

Start PostgreSQL:
```bash
brew services start postgresql@15
```

---

## Step 4: Create the Database

Run these commands one at a time:

```bash
createdb trail_pal
```

```bash
psql trail_pal -c "CREATE EXTENSION postgis;"
```

If you see "CREATE EXTENSION", it worked!

---

## Step 5: Get Your Free OpenRouteService API Key

1. Go to: https://openrouteservice.org/dev/#/signup
2. Click "Sign Up" and create a free account
3. After signing in, go to your Dashboard
4. Click "Request a Token" (or you may already have one)
5. Copy the long string of letters and numbers - this is your API key
6. Save this somewhere safe - you'll need it in Step 7

---

## Step 6: Set Up Trail Pal

Navigate to the project folder and set up Python:

```bash
cd ~/projects/trail_pal
```

Create a fresh environment with Python 3.11:
```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## Step 7: Configure Your Settings

Create your configuration file:

```bash
cp .env.example .env
```

Now open the `.env` file to edit it:
```bash
open -e .env
```

In the text editor that opens, change these lines:

**Before:**
```
DATABASE_URL=postgresql://user:password@localhost:5432/trail_pal
ORS_API_KEY=your_api_key_here
```

**After:**
```
DATABASE_URL=postgresql://localhost:5432/trail_pal
ORS_API_KEY=paste_your_api_key_from_step_5_here
```

Save the file (Cmd + S) and close the editor.

---

## Step 8: Set Up the Database Tables

```bash
source venv/bin/activate
trail-pal init
```

You should see "Database initialized successfully!"

---

## Step 9: Download Waypoints for Cornwall

This fetches campsites, hostels, and points of interest from OpenStreetMap:

```bash
trail-pal seed --region cornwall
```

This may take 30-60 seconds. You'll see statistics about how many waypoints were found.

---

## Step 10: Build the Route Network

This calculates which waypoints are within hiking distance of each other:

```bash
trail-pal build-graph --region cornwall
```

**Note:** This step can take a long time (potentially hours) because it needs to calculate hiking routes between many pairs of waypoints, and the free API has rate limits. You can stop it with `Ctrl + C` and resume later - it will skip pairs it already calculated.

---

## Step 11: Generate Your Hiking Itinerary!

Once you have some connections built, try:

```bash
trail-pal generate --region cornwall --days 3
```

This will show you suggested 3-day hiking routes!

---

## Quick Reference: Common Commands

Always run `source venv/bin/activate` first when opening a new Terminal window.

| What you want to do | Command |
|---------------------|---------|
| See available regions | `trail-pal regions` |
| View waypoint statistics | `trail-pal waypoint-stats --region cornwall` |
| View route network stats | `trail-pal graph-stats --region cornwall` |
| Generate a 3-day trip | `trail-pal generate --region cornwall --days 3` |
| Export route to GPS file | `trail-pal export --region cornwall -o my_hike.gpx` |
| Get help | `trail-pal --help` |

---

## Troubleshooting

### "command not found: trail-pal"
Make sure you activated the virtual environment:
```bash
cd ~/projects/trail_pal
source venv/bin/activate
```

### Database connection errors
Make sure PostgreSQL is running:
```bash
brew services start postgresql@15
```

### "No itineraries found"
The graph needs more connections. Run `trail-pal build-graph --region cornwall` and let it run longer.

### API rate limit errors
The free OpenRouteService tier allows 40 requests per minute. The tool automatically waits between requests, but if you see rate limit errors, just wait a minute and try again.

---

## Need More Help?

Feel free to ask! Common things people want to know:
- How to add more regions beyond Cornwall
- How to customize the hiking distance per day
- How to view routes on a map

