// State
let currentItinerary = null;
let currentRequestParams = null;
let map = null;
let routeLayers = [];

// Feedback state
let feedbackRating = 0;
let feedbackSubmitted = false;

// DOM elements
const form = document.getElementById('itinerary-form');
const formSection = document.getElementById('form-section');
const regionSelect = document.getElementById('region');
const daysInput = document.getElementById('days');
const startWaypointInput = document.getElementById('start-waypoint');
const waypointSuggestions = document.getElementById('waypoint-suggestions');
const preferAccommodationCheckbox = document.getElementById('prefer-accommodation');
const generateBtn = document.getElementById('generate-btn');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resultsSection = document.getElementById('results-section');
const itinerarySummary = document.getElementById('itinerary-summary');
const daysList = document.getElementById('days-list');
const downloadGpxBtn = document.getElementById('download-gpx-btn');
const generateNewBtn = document.getElementById('generate-new-btn');
const mapSection = document.getElementById('map-section');
const mapContainer = document.getElementById('map');

// Feedback DOM elements
const feedbackSection = document.getElementById('feedback-section');
const feedbackStep1 = document.getElementById('feedback-step-1');
const feedbackStep2 = document.getElementById('feedback-step-2');
const feedbackStep3 = document.getElementById('feedback-step-3');
const starRating = document.getElementById('star-rating');
const ratingLabel = document.getElementById('rating-label');
const backToRatingBtn = document.getElementById('back-to-rating-btn');
const submitFeedbackBtn = document.getElementById('submit-feedback-btn');

// Autocomplete state
let autocompleteTimeout = null;
let selectedWaypoint = null;
let currentSuggestions = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadRegions();
    form.addEventListener('submit', handleFormSubmit);
    downloadGpxBtn.addEventListener('click', handleDownloadGpx);
    
    // Feedback event listeners
    if (starRating) {
        setupStarRating();
    }
    if (backToRatingBtn) {
        backToRatingBtn.addEventListener('click', showFeedbackStep1);
    }
    if (submitFeedbackBtn) {
        submitFeedbackBtn.addEventListener('click', handleSubmitFeedback);
    }
    
    generateNewBtn.addEventListener('click', handleGenerateNew);
    
    // Setup autocomplete for waypoint input
    setupWaypointAutocomplete();
    
    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
        const autocompleteContainer = startWaypointInput.closest('.autocomplete-container');
        if (autocompleteContainer && !autocompleteContainer.contains(e.target)) {
            hideSuggestions();
        }
    });
});

// Load available regions
async function loadRegions() {
    try {
        const response = await fetch('/regions');
        if (!response.ok) {
            throw new Error('Failed to load regions');
        }
        const regions = await response.json();
        
        regionSelect.innerHTML = '<option value="">Select a region...</option>';
        regions.forEach(region => {
            const option = document.createElement('option');
            option.value = region.name;
            // Special case for Cornwall - just show "Cornwall"
            if (region.name.toLowerCase() === 'cornwall') {
                option.textContent = 'Cornwall';
            } else {
                option.textContent = `${region.name} (${region.country})`;
            }
            regionSelect.appendChild(option);
        });
        
        // Auto-select Cornwall if available, otherwise select if only one region
        const cornwall = regions.find(r => r.name.toLowerCase() === 'cornwall');
        if (cornwall) {
            regionSelect.value = cornwall.name;
        } else if (regions.length === 1) {
            regionSelect.value = regions[0].name;
        }
    } catch (error) {
        console.error('Error loading regions:', error);
        regionSelect.innerHTML = '<option value="">Error loading regions</option>';
    }
}

// Setup waypoint autocomplete
function setupWaypointAutocomplete() {
    startWaypointInput.addEventListener('input', handleWaypointInput);
    startWaypointInput.addEventListener('keydown', handleWaypointKeydown);
    startWaypointInput.addEventListener('focus', () => {
        if (startWaypointInput.value.trim() && currentSuggestions.length > 0) {
            showSuggestions(currentSuggestions);
        }
    });
}

// Handle waypoint input with debouncing
function handleWaypointInput(e) {
    const query = e.target.value.trim();
    
    // Clear previous timeout
    if (autocompleteTimeout) {
        clearTimeout(autocompleteTimeout);
    }
    
    // Reset selection when user types
    selectedWaypoint = null;
    
    if (!query) {
        hideSuggestions();
        return;
    }
    
    // Get current region
    const region = regionSelect.value;
    if (!region) {
        hideSuggestions();
        return;
    }
    
    // Debounce the search
    autocompleteTimeout = setTimeout(() => {
        searchWaypoints(region, query);
    }, 300);
}

// Handle keyboard navigation in autocomplete
function handleWaypointKeydown(e) {
    const suggestions = waypointSuggestions.querySelectorAll('.autocomplete-suggestion');
    const selected = waypointSuggestions.querySelector('.autocomplete-suggestion.selected');
    let selectedIndex = -1;
    
    if (selected) {
        selectedIndex = Array.from(suggestions).indexOf(selected);
    }
    
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (selectedIndex < suggestions.length - 1) {
            if (selected) selected.classList.remove('selected');
            suggestions[selectedIndex + 1].classList.add('selected');
            suggestions[selectedIndex + 1].scrollIntoView({ block: 'nearest' });
        } else if (suggestions.length > 0) {
            suggestions[0].classList.add('selected');
        }
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (selectedIndex > 0) {
            if (selected) selected.classList.remove('selected');
            suggestions[selectedIndex - 1].classList.add('selected');
            suggestions[selectedIndex - 1].scrollIntoView({ block: 'nearest' });
        } else if (selected) {
            selected.classList.remove('selected');
        }
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (selected) {
            selectWaypoint(selected);
        } else if (suggestions.length > 0) {
            selectWaypoint(suggestions[0]);
        }
    } else if (e.key === 'Escape') {
        hideSuggestions();
    }
}

// Search waypoints
async function searchWaypoints(region, query) {
    try {
        // Always include all waypoints in search results regardless of checkbox
        const response = await fetch(`/regions/${region}/waypoints/search?q=${encodeURIComponent(query)}&limit=10&include_all=true`);
        if (!response.ok) {
            throw new Error('Failed to search waypoints');
        }
        const waypoints = await response.json();
        currentSuggestions = waypoints;
        showSuggestions(waypoints);
    } catch (error) {
        console.error('Error searching waypoints:', error);
        hideSuggestions();
    }
}

// Show suggestions
function showSuggestions(waypoints) {
    if (waypoints.length === 0) {
        hideSuggestions();
        return;
    }
    
    waypointSuggestions.innerHTML = '';
    waypoints.forEach(waypoint => {
        const suggestion = document.createElement('div');
        suggestion.className = 'autocomplete-suggestion';
        suggestion.innerHTML = `
            <div class="autocomplete-suggestion-name">${escapeHtml(waypoint.name)}</div>
            <div class="autocomplete-suggestion-type">${escapeHtml(waypoint.waypoint_type.replace(/_/g, ' '))}</div>
        `;
        suggestion.addEventListener('click', () => selectWaypoint(suggestion, waypoint));
        suggestion.addEventListener('mouseenter', () => {
            waypointSuggestions.querySelectorAll('.autocomplete-suggestion').forEach(s => s.classList.remove('selected'));
            suggestion.classList.add('selected');
        });
        waypointSuggestions.appendChild(suggestion);
    });
    
    waypointSuggestions.classList.add('show');
}

// Hide suggestions
function hideSuggestions() {
    waypointSuggestions.classList.remove('show');
    waypointSuggestions.innerHTML = '';
}

// Select a waypoint
function selectWaypoint(suggestionElement, waypoint = null) {
    if (!waypoint) {
        // Extract waypoint from current suggestions
        const index = Array.from(waypointSuggestions.querySelectorAll('.autocomplete-suggestion')).indexOf(suggestionElement);
        waypoint = currentSuggestions[index];
    }
    
    if (waypoint) {
        selectedWaypoint = waypoint;
        startWaypointInput.value = waypoint.name;
        // Set days to 3 when a waypoint is selected (for 3-day itinerary)
        daysInput.value = 3;
        hideSuggestions();
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const region = regionSelect.value;
    const days = parseInt(daysInput.value);
    // Use selected waypoint name if available, otherwise use input value
    const startWaypoint = selectedWaypoint ? selectedWaypoint.name : (startWaypointInput.value.trim() || null);
    const preferAccommodation = preferAccommodationCheckbox.checked;
    
    // If user explicitly provided a waypoint name (selected or typed), allow any waypoint type
    // Otherwise, default to train stations/towns (allow_any_start: false)
    const allowAnyStart = startWaypoint !== null;
    
    // Store request params for GPX download
    currentRequestParams = {
        region,
        days,
        start_waypoint_name: startWaypoint,
        prefer_accommodation: preferAccommodation,
        max_results: 1,
        randomize: true,
        allow_any_start: allowAnyStart
    };
    
    // Show loading, hide errors and results
    loadingDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    resultsSection.style.display = 'none';
    generateBtn.disabled = true;
    
    try {
        const response = await fetch('/itineraries/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentRequestParams)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate itinerary');
        }
        
        const data = await response.json();
        
        if (data.count === 0 || !data.itineraries || data.itineraries.length === 0) {
            throw new Error('No itineraries found');
        }
        
        currentItinerary = data.itineraries[0];
        displayItinerary(currentItinerary);
        
    } catch (error) {
        console.error('Error generating itinerary:', error);
        errorDiv.textContent = `Error: ${error.message}`;
        errorDiv.style.display = 'block';
    } finally {
        loadingDiv.style.display = 'none';
        generateBtn.disabled = false;
    }
}

// Display itinerary
function displayItinerary(itinerary) {
    // Display summary
    const totalHours = Math.floor(itinerary.total_duration_minutes / 60);
    const totalMinutes = itinerary.total_duration_minutes % 60;
    
    // Format region name for display
    const regionDisplay = itinerary.region.toLowerCase() === 'cornwall' 
        ? 'Cornwall' 
        : itinerary.region.charAt(0).toUpperCase() + itinerary.region.slice(1);
    
    itinerarySummary.innerHTML = `
        <div class="summary-item"><strong>Region:</strong> ${regionDisplay}</div>
        <div class="summary-item"><strong>Total Distance:</strong> ${itinerary.total_distance_km.toFixed(1)} km</div>
        <div class="summary-item"><strong>Total Duration:</strong> ${totalHours}h ${totalMinutes}min</div>
        <div class="summary-item"><strong>Total Elevation Gain:</strong> ${itinerary.total_elevation_gain_m.toFixed(0)}m</div>
    `;
    
    // Build consistent waytype color mapping across all days
    const allWaytypes = {};
    itinerary.days.forEach(day => {
        if (day.surface_stats && day.surface_stats.waytypes) {
            Object.entries(day.surface_stats.waytypes).forEach(([type, distance]) => {
                allWaytypes[type] = (allWaytypes[type] || 0) + distance;
            });
        }
    });
    // Sort by total distance across all days and assign color indices
    const waytypeColorMap = {};
    Object.entries(allWaytypes)
        .sort((a, b) => b[1] - a[1])
        .forEach(([type], index) => {
            waytypeColorMap[type] = index;
        });

    // Display days
    daysList.innerHTML = '';
    itinerary.days.forEach(day => {
        const dayCard = document.createElement('div');
        dayCard.className = 'day-card';
        
        const durationHours = Math.floor(day.duration_minutes / 60);
        const durationMinutes = day.duration_minutes % 60;
        
        let waytypeHtml = '';
        if (day.surface_stats && day.surface_stats.total_distance_km > 0) {
            const waytypes = day.surface_stats.waytypes;
            const totalKm = day.surface_stats.total_distance_km;
            
            // Sort way types by distance for this day
            const waytypeEntries = Object.entries(waytypes)
                .sort((a, b) => b[1] - a[1]);
            
            if (waytypeEntries.length > 0) {
                // Build the stacked bar segments using consistent colors
                let barSegments = '';
                let legendItems = '';
                
                waytypeEntries.forEach(([type, distance]) => {
                    const colorIndex = waytypeColorMap[type] || 0;
                    const percentage = (distance / totalKm) * 100;
                    const typeDisplay = type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    barSegments += `<div class="waytype-segment waytype-${colorIndex}" style="width: ${percentage}%" title="${typeDisplay}: ${percentage.toFixed(1)}%"></div>`;
                    legendItems += `<div class="waytype-legend-item"><span class="waytype-dot waytype-${colorIndex}"></span>${typeDisplay} <span class="waytype-percent">${percentage.toFixed(0)}%</span></div>`;
                });
                
                waytypeHtml = `
                    <div class="waytype-stats">
                        <div class="waytype-bar">${barSegments}</div>
                        <div class="waytype-legend">${legendItems}</div>
                    </div>`;
            }
        }
        
        dayCard.innerHTML = `
            <h3>Day ${day.day_number}</h3>
            <div class="day-info">
                <div class="day-info-item">
                    <strong>Start:</strong>
                    ${day.start.name} (${day.start.waypoint_type})
                </div>
                <div class="day-info-item">
                    <strong>End:</strong>
                    ${day.end.name} (${day.end.waypoint_type})
                </div>
                <div class="day-info-item">
                    <strong>Distance:</strong>
                    ${day.distance_km.toFixed(1)} km
                </div>
                <div class="day-info-item">
                    <strong>Duration:</strong>
                    ${durationHours}h ${durationMinutes}min
                </div>
                ${day.elevation_gain_m ? `
                <div class="day-info-item">
                    <strong>Elevation:</strong>
                    +${day.elevation_gain_m.toFixed(0)}m / -${(day.elevation_loss_m || 0).toFixed(0)}m
                </div>
                ` : ''}
            </div>
                ${waytypeHtml}
            `;
        
        daysList.appendChild(dayCard);
    });
    
    // Hide form, show results
    formSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Automatically load the map
    loadMapForItinerary();
    
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Handle GPX download
async function handleDownloadGpx() {
    if (!currentItinerary) {
        alert('No itinerary to download. Please generate an itinerary first.');
        return;
    }
    
    downloadGpxBtn.disabled = true;
    downloadGpxBtn.textContent = 'Downloading...';
    
    // Build request with itinerary data to ensure consistent results
    const exportRequest = {
        region: currentItinerary.region,
        itinerary_id: currentItinerary.id,
        days: currentItinerary.days.map(day => ({
            day_number: day.day_number,
            start_id: day.start.id,
            end_id: day.end.id
        }))
    };
    
    try {
        const response = await fetch('/itineraries/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(exportRequest)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to download GPX');
        }
        
        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `${currentItinerary.region}_itinerary_${currentItinerary.days.length}days.gpx`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }
        
        // Download file
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
    } catch (error) {
        console.error('Error downloading GPX:', error);
        alert(`Error downloading GPX: ${error.message}`);
    } finally {
        downloadGpxBtn.disabled = false;
        downloadGpxBtn.textContent = 'Download GPX';
    }
}

// Load map for itinerary
async function loadMapForItinerary() {
    console.log('Loading map for itinerary');
    
    if (!currentItinerary) {
        console.error('No itinerary to display');
        return;
    }
    
    // Build request with itinerary data to ensure consistent results
    const geometryRequest = {
        region: currentItinerary.region,
        itinerary_id: currentItinerary.id,
        days: currentItinerary.days.map(day => ({
            day_number: day.day_number,
            start_id: day.start.id,
            end_id: day.end.id
        }))
    };
    
    try {
        console.log('Fetching route geometry...', geometryRequest);
        // Fetch route geometry
        const response = await fetch('/itineraries/geometry', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(geometryRequest)
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load route geometry');
        }
        
        const geometryData = await response.json();
        console.log('Geometry data received:', geometryData);
        displayMap(geometryData);
        
    } catch (error) {
        console.error('Error loading map:', error);
    }
}

// Display map with route
function displayMap(geometryData) {
    console.log('Displaying map with geometry data:', geometryData);
    
    // Use requestAnimationFrame to ensure DOM is updated before initializing map
    requestAnimationFrame(() => {
        // Initialize map if not already created
        if (!map) {
            console.log('Initializing new map...');
            try {
                map = L.map('map');
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 19
                }).addTo(map);
                console.log('Map initialized successfully');
            } catch (error) {
                console.error('Error initializing map:', error);
                return;
            }
        }
        
        // Invalidate size after a short delay to ensure container is visible
        setTimeout(() => {
            if (map) {
                map.invalidateSize();
                console.log('Map size invalidated');
                drawRouteOnMap(geometryData);
            }
        }, 200);
    });
}

// Draw route on the map
function drawRouteOnMap(geometryData) {
    if (!map) {
        console.error('Map not initialized');
        return;
    }
    
    // Clear existing route layers
    routeLayers.forEach(layer => {
        map.removeLayer(layer);
    });
    routeLayers = [];
    
    // Colors for different days
    const dayColors = [
        '#3498db', // Blue
        '#27ae60', // Green
        '#e74c3c', // Red
        '#f39c12', // Orange
        '#9b59b6', // Purple
        '#1abc9c', // Turquoise
        '#e67e22', // Dark Orange
        '#34495e', // Dark Blue
        '#16a085', // Dark Turquoise
        '#c0392b', // Dark Red
        '#8e44ad', // Dark Purple
        '#d35400', // Dark Orange
        '#2980b9', // Blue
        '#27ae60'  // Green
    ];
    
    // Collect all coordinates for bounds calculation
    const allCoords = [];
    
    // Draw route for each day
    geometryData.days.forEach(day => {
        const color = dayColors[(day.day_number - 1) % dayColors.length];
        
        // Draw polyline for the route
        if (day.geometry && day.geometry.length > 0) {
            const latlngs = day.geometry.map(coord => [coord[1], coord[0]]); // Convert [lon, lat] to [lat, lon]
            const polyline = L.polyline(latlngs, {
                color: color,
                weight: 4,
                opacity: 0.8
            }).addTo(map);
            
            polyline.bindPopup(`Day ${day.day_number}: ${day.start.name} to ${day.end.name}`);
            routeLayers.push(polyline);
            
            // Add coordinates to bounds
            latlngs.forEach(coord => allCoords.push(coord));
        }
        
        // Add start marker (green)
        const startMarker = L.marker([day.start.latitude, day.start.longitude], {
            icon: L.divIcon({
                className: 'custom-marker start-marker',
                html: `<div style="background-color: #27ae60; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${day.day_number}</div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 15]
            })
        }).addTo(map);
        
        startMarker.bindPopup(`<strong>Day ${day.day_number} Start</strong><br>${day.start.name}<br>(${day.start.waypoint_type})`);
        routeLayers.push(startMarker);
        allCoords.push([day.start.latitude, day.start.longitude]);
        
        // Add end marker (red) - only for the last day or if it's different from next day's start
        const isLastDay = day.day_number === geometryData.days.length;
        const isDifferentFromNext = !isLastDay && 
            (day.end.latitude !== geometryData.days[day.day_number].start.latitude ||
             day.end.longitude !== geometryData.days[day.day_number].start.longitude);
        
        if (isLastDay || isDifferentFromNext) {
            const endMarker = L.marker([day.end.latitude, day.end.longitude], {
                icon: L.divIcon({
                    className: 'custom-marker end-marker',
                    html: `<div style="background-color: #e74c3c; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${day.day_number}</div>`,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                })
            }).addTo(map);
            
            endMarker.bindPopup(`<strong>Day ${day.day_number} End</strong><br>${day.end.name}<br>(${day.end.waypoint_type})`);
            routeLayers.push(endMarker);
            allCoords.push([day.end.latitude, day.end.longitude]);
        }
    });
    
    // Fit map to show all points
    if (allCoords.length > 0) {
        const bounds = L.latLngBounds(allCoords);
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

// Handle generate new
function handleGenerateNew() {
    // Save region selection before reset
    const savedRegion = regionSelect.value;
    
    // Reset form
    form.reset();
    
    // Restore region and set defaults
    regionSelect.value = savedRegion || 'cornwall';
    daysInput.value = 3;
    preferAccommodationCheckbox.checked = true;
    
    // Clear autocomplete state
    selectedWaypoint = null;
    currentSuggestions = [];
    hideSuggestions();
    
    // Clear state
    currentItinerary = null;
    currentRequestParams = null;
    
    // Reset feedback state
    resetFeedbackState();
    
    // Clear map layers
    if (map) {
        routeLayers.forEach(layer => {
            map.removeLayer(layer);
        });
        routeLayers = [];
    }
    
    // Hide results and errors
    resultsSection.style.display = 'none';
    errorDiv.style.display = 'none';
    
    // Show form
    formSection.style.display = 'block';
    formSection.scrollIntoView({ behavior: 'smooth' });
}

// ===== FEEDBACK FUNCTIONS =====

// Rating labels for each star level
const ratingLabels = {
    1: 'Poor',
    2: 'Fair',
    3: 'Good',
    4: 'Very Good',
    5: 'Excellent'
};

// Reset feedback to initial state
function resetFeedbackState() {
    feedbackRating = 0;
    feedbackSubmitted = false;
    resetStarRating();
    resetFeedbackReasons();
    showFeedbackStep1();
}

// Setup star rating interaction
function setupStarRating() {
    const stars = starRating.querySelectorAll('.star');
    
    stars.forEach(star => {
        star.addEventListener('click', () => {
            feedbackRating = parseInt(star.dataset.rating);
            updateStarDisplay(feedbackRating);
            ratingLabel.textContent = ratingLabels[feedbackRating];
            
            // After a short delay, move to step 2
            setTimeout(() => {
                showFeedbackStep2();
            }, 300);
        });
        
        star.addEventListener('mouseenter', () => {
            const hoverRating = parseInt(star.dataset.rating);
            updateStarDisplay(hoverRating);
            ratingLabel.textContent = ratingLabels[hoverRating];
        });
        
        star.addEventListener('mouseleave', () => {
            updateStarDisplay(feedbackRating);
            ratingLabel.textContent = feedbackRating ? ratingLabels[feedbackRating] : '';
        });
    });
}

// Update star display
function updateStarDisplay(rating) {
    const stars = starRating.querySelectorAll('.star');
    stars.forEach(star => {
        const starRatingValue = parseInt(star.dataset.rating);
        if (starRatingValue <= rating) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
}

// Reset star rating display
function resetStarRating() {
    updateStarDisplay(0);
    ratingLabel.textContent = '';
}

// Reset feedback reason checkboxes
function resetFeedbackReasons() {
    const checkboxes = feedbackStep2.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = false);
}

// Show feedback step 1 (rating)
function showFeedbackStep1() {
    feedbackStep1.style.display = 'block';
    feedbackStep2.style.display = 'none';
    feedbackStep3.style.display = 'none';
}

// Show feedback step 2 (reasons)
function showFeedbackStep2() {
    feedbackStep1.style.display = 'none';
    feedbackStep2.style.display = 'block';
    feedbackStep3.style.display = 'none';
}

// Show feedback step 3 (thank you)
function showFeedbackStep3() {
    feedbackStep1.style.display = 'none';
    feedbackStep2.style.display = 'none';
    feedbackStep3.style.display = 'block';
}

// Get selected feedback reasons
function getSelectedReasons() {
    const checkboxes = feedbackStep2.querySelectorAll('input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// Build route summary for feedback
function buildRouteSummary() {
    if (!currentItinerary) return {};
    
    return {
        total_distance_km: currentItinerary.total_distance_km,
        total_duration_minutes: currentItinerary.total_duration_minutes,
        total_elevation_gain_m: currentItinerary.total_elevation_gain_m,
        num_days: currentItinerary.days.length,
        days: currentItinerary.days.map(day => ({
            day_number: day.day_number,
            start_id: day.start.id,
            start_name: day.start.name,
            end_id: day.end.id,
            end_name: day.end.name,
            distance_km: day.distance_km
        }))
    };
}

// Handle submit feedback
async function handleSubmitFeedback() {
    if (!currentItinerary || feedbackRating === 0) {
        alert('Please select a rating first.');
        return;
    }
    
    submitFeedbackBtn.disabled = true;
    submitFeedbackBtn.textContent = 'Submitting...';
    
    const feedbackData = {
        itinerary_id: currentItinerary.id,
        region: currentItinerary.region,
        rating: feedbackRating,
        feedback_reasons: getSelectedReasons(),
        route_summary: buildRouteSummary()
    };
    
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to submit feedback');
        }
        
        // Success - mark as submitted and show thank you
        feedbackSubmitted = true;
        showFeedbackStep3();
        
    } catch (error) {
        console.error('Error submitting feedback:', error);
        alert(`Error submitting feedback: ${error.message}`);
    } finally {
        submitFeedbackBtn.disabled = false;
        submitFeedbackBtn.textContent = 'Submit Feedback';
    }
}
