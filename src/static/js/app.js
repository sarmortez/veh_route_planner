// static/js/app.js
const debugLog = document.getElementById("debug-log");
function logToUI(message) {
    if (debugLog) {
        const p = document.createElement("p");
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        debugLog.appendChild(p);
        debugLog.scrollTop = debugLog.scrollHeight; // Auto-scroll to bottom
    }
    console.log(message); // Also log to browser console
}

logToUI("app.js started");
document.addEventListener("DOMContentLoaded", () => {
    logToUI("DOM fully loaded and parsed");
    const MAPBOX_TOKEN = "********************************************************"; // Replace with your Mapbox Access Token 
    mapboxgl.accessToken = MAPBOX_TOKEN;

    const mapContainer = document.getElementById("map");
    if (!mapContainer) {
        logToUI("Map container not found!");
        console.error("Map container not found!");
        return;
    }
    const map = new mapboxgl.Map({
        container: "map",
        style: "mapbox://styles/mapbox/streets-v11",
        center: [-81.3792, 28.5383], // Orlando coordinates
        zoom: 10
    });
    logToUI("Map initialized for Orlando");

    // Global variables for both P2F and F2P
    let allFetchedGroceryStores = [];
    let allFetchedTransitStops = [];
    let allFetchedCommunityCenters = [];
    
    // P2F specific variables
    let selectedGroceryStores = []; 
    let selectedTransitStops = [];
    let p2fLocationIdToShortNameMap = {}; 
    
    let groceryStoreMarkers = {}; 
    let transitStopMarkers = {}; 
    let communityCenterMarkers = {};
    let activeRouteHighlight = null;
    let activeRowHighlight = null;

    // F2P specific variables
    let selectedF2PGroceryStores = [];
    let selectedCommunityCenters = [];
    let f2pLocationIdToShortNameMap = {};

    // UI elements
    const p2fGroceryStoresDisplay = document.getElementById("p2f-grocery-stores-display"); 
    const f2pGroceryStoresDisplay = document.getElementById("f2p-grocery-stores-display");
    const transitStopsListDiv = document.getElementById("transit-stops-list");
    const communityCentersListDiv = document.getElementById("community-centers-list");
    const runP2FOptimizationButton = document.getElementById("run-p2f-optimization-button");
    const runF2POptimizationButton = document.getElementById("run-f2p-optimization-button");
    const recalibrateButton = document.getElementById("recalibrate-button");
    const recalibrationMessage = document.getElementById("recalibration-message");
    const optimizationMessage = document.getElementById("optimization-message");
    const scheduleTableDiv = document.getElementById("schedule-table");
    const tabButtons = document.querySelectorAll(".tab-button");
    const tabContents = document.querySelectorAll(".tab-content");

    // Initialize recalibrate button as disabled
    if (recalibrateButton) {
        recalibrateButton.disabled = true;
    }

    // Tab switching functionality
    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            const tabId = button.getAttribute("data-tab");
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            
            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove("active");
                if (content.id === `${tabId}-tab`) {
                    content.classList.add("active");
                }
            });
            
            // Clear the schedule when switching tabs
            if (scheduleTableDiv) {
                scheduleTableDiv.innerHTML = "<p>No schedule generated yet, or no trips planned.</p>";
            }
            
            // Clear any route layers on the map
            clearAllRouteLayers();
            
            // Update marker visibility based on active tab
            updateMarkerVisibility(tabId);
        });
    });

    function updateMarkerVisibility(activeTab) {
        // Show/hide markers based on active tab
        Object.values(groceryStoreMarkers).forEach(markerData => {
            const element = markerData.marker.getElement();
            element.style.display = "block"; // Always show grocery stores
        });
        
        Object.values(transitStopMarkers).forEach(markerData => {
            const element = markerData.marker.getElement();
            element.style.display = activeTab === "p2f" ? "block" : "none";
        });
        
        Object.values(communityCenterMarkers).forEach(markerData => {
            const element = markerData.marker.getElement();
            element.style.display = activeTab === "f2p" ? "block" : "none";
        });
    }

    function updateP2FShortNameMapping() {
        p2fLocationIdToShortNameMap = {}; 
        selectedGroceryStores.forEach((store, index) => {
            p2fLocationIdToShortNameMap[store.id] = `GS${index + 1}`;
        });
        selectedTransitStops.forEach((stop, index) => {
            p2fLocationIdToShortNameMap[stop.id] = `TS${index + 1}`;
        });
        logToUI("P2F short name mapping updated");
    }

    function updateF2PShortNameMapping() {
        f2pLocationIdToShortNameMap = {}; 
        selectedF2PGroceryStores.forEach((store, index) => {
            f2pLocationIdToShortNameMap[store.id] = `GS${index + 1}`;
        });
        selectedCommunityCenters.forEach((center, index) => {
            f2pLocationIdToShortNameMap[center.id] = `CC${index + 1}`;
        });
        logToUI("F2P short name mapping updated");
    }

    function updateP2FRunButtonState() {
        if (runP2FOptimizationButton) {
            runP2FOptimizationButton.disabled = !(selectedGroceryStores.length > 0 && selectedTransitStops.length > 0);
            logToUI(`P2F Run button state updated. Disabled: ${runP2FOptimizationButton.disabled}`);
        }
    }

    function updateF2PRunButtonState() {
        if (runF2POptimizationButton) {
            runF2POptimizationButton.disabled = !(selectedF2PGroceryStores.length > 0 && selectedCommunityCenters.length > 0);
            logToUI(`F2P Run button state updated. Disabled: ${runF2POptimizationButton.disabled}`);
        }
    }

    function createMarkerElement(type, id) {
        const el = document.createElement("div");
        if (type === "grocery") {
            el.className = "marker-gs-all";
        } else if (type === "transit") {
            el.className = "marker-ts-all";
        } else if (type === "community") {
            el.className = "marker-cc-all";
        }
        el.dataset.id = id;
        el.dataset.type = type;
        return el;
    }

    function updateMarkerOpacity(markerElement, isSelected) {
        if (markerElement) {
            markerElement.style.opacity = isSelected ? "1" : "0.4";
        }
    }

    function showMessage(messageElement, message, type) {
        if (!messageElement) return;
        
        // Remove all existing classes
        messageElement.classList.remove("message-success", "message-warning", "message-info");
        
        // Add appropriate class based on type
        if (type === "success") {
            messageElement.classList.add("message-success");
        } else if (type === "warning") {
            messageElement.classList.add("message-warning");
        } else {
            messageElement.classList.add("message-info");
        }
        
        messageElement.textContent = message;
        messageElement.style.display = "block";
    }

    function hideMessage(messageElement) {
        if (messageElement) {
            messageElement.style.display = "none";
            messageElement.textContent = "";
        }
    }

    // Function to highlight a route on the map
    function highlightRoute(routeLayerId) {
        // Reset any previous highlight
        resetRouteHighlight();
        
        // Highlight the new route
        if (map.getLayer(routeLayerId)) {
            map.setPaintProperty(routeLayerId, 'line-width', 6);
            map.setPaintProperty(routeLayerId, 'line-opacity', 1);
            
            // Store the active highlight for later reset
            activeRouteHighlight = routeLayerId;
        }
    }

    // Function to reset route highlight
    function resetRouteHighlight() {
        if (activeRouteHighlight && map.getLayer(activeRouteHighlight)) {
            map.setPaintProperty(activeRouteHighlight, 'line-width', 3);
            map.setPaintProperty(activeRouteHighlight, 'line-opacity', 0.6);
            activeRouteHighlight = null;
        }
    }

    // Function to highlight a table row
    function highlightTableRow(row) {
        // Reset any previous highlight
        resetTableRowHighlight();
        
        // Highlight the new row
        if (row) {
            row.classList.add('highlighted-row');
            activeRowHighlight = row;
        }
    }

    // Function to reset table row highlight
    function resetTableRowHighlight() {
        if (activeRowHighlight) {
            activeRowHighlight.classList.remove('highlighted-row');
            activeRowHighlight = null;
        }
    }

    // Function to highlight a marker
    function highlightMarker(markerId, markerType) {
        let markers;
        if (markerType === 'grocery') {
            markers = groceryStoreMarkers;
        } else if (markerType === 'transit') {
            markers = transitStopMarkers;
        } else if (markerType === 'community') {
            markers = communityCenterMarkers;
        }
        
        const markerData = markers[markerId];
        
        if (markerData && markerData.marker) {
            const el = markerData.marker.getElement();
            if (el) {
                // Add highlight class without changing position or visibility
                el.classList.add('marker-highlight');
                // Scale slightly larger but keep in place
                el.style.width = '20px';
                el.style.height = '20px';
            }
        }
    }

    // Function to reset marker highlight
    function resetMarkerHighlight(markerId, markerType) {
        let markers;
        if (markerType === 'grocery') {
            markers = groceryStoreMarkers;
        } else if (markerType === 'transit') {
            markers = transitStopMarkers;
        } else if (markerType === 'community') {
            markers = communityCenterMarkers;
        }
        
        const markerData = markers[markerId];
        
        if (markerData && markerData.marker) {
            const el = markerData.marker.getElement();
            if (el) {
                // Remove highlight class
                el.classList.remove('marker-highlight');
                // Reset size
                el.style.width = '15px';
                el.style.height = '15px';
            }
        }
    }

    // Function to clear all route layers from the map
    function clearAllRouteLayers() {
        // Get all layers
        const layers = map.getStyle().layers;
        
        // Find and remove route layers
        for (const layer of layers) {
            if (layer.id.includes('route-')) {
                map.removeLayer(layer.id);
            }
        }
        
        // Find and remove route sources
        const sources = map.getStyle().sources;
        for (const sourceId in sources) {
            if (sourceId.includes('route-')) {
                map.removeSource(sourceId);
            }
        }
        
        activeRouteHighlight = null;
    }

    async function loadAndDisplayAllLocations() {
        logToUI("Fetching all locations...");
        try {
            const response = await fetch("/api/all_locations");
            if (!response.ok) throw new Error(`Failed to fetch locations: ${response.status}`);
            const data = await response.json();
            logToUI("Locations fetched successfully");

            allFetchedGroceryStores = data.grocery_stores || [];
            allFetchedTransitStops = data.transit_stops || [];
            allFetchedCommunityCenters = data.community_centers || [];

            // Clear existing markers
            Object.values(groceryStoreMarkers).forEach(m => m.marker.remove());
            groceryStoreMarkers = {};
            Object.values(transitStopMarkers).forEach(m => m.marker.remove());
            transitStopMarkers = {};
            Object.values(communityCenterMarkers).forEach(m => m.marker.remove());
            communityCenterMarkers = {};

            // Create grocery store markers
            allFetchedGroceryStores.forEach(store => {
                const el = createMarkerElement("grocery", store.id);
                const isSelectedP2F = selectedGroceryStores.some(s => s.id === store.id);
                const isSelectedF2P = selectedF2PGroceryStores.some(s => s.id === store.id);
                updateMarkerOpacity(el, isSelectedP2F || isSelectedF2P);
                
                const marker = new mapboxgl.Marker(el)
                    .setLngLat([store.longitude, store.latitude])
                    .addTo(map);
                    
                groceryStoreMarkers[store.id] = { marker: marker, data: store, element: el };
                
                const popup = new mapboxgl.Popup({ offset: 25, closeButton: false, closeOnClick: false })
                    .setText(`GS: ${store.name || store.id} (Lat: ${store.latitude.toFixed(4)}, Lon: ${store.longitude.toFixed(4)})`);
                    
                marker.getElement().addEventListener("mouseenter", () => { 
                    map.getCanvas().style.cursor = "pointer"; 
                    popup.setLngLat([store.longitude, store.latitude]).addTo(map); 
                });
                
                marker.getElement().addEventListener("mouseleave", () => { 
                    map.getCanvas().style.cursor = ""; 
                    popup.remove(); 
                });
                
                marker.getElement().addEventListener("click", (e) => {
                    e.stopPropagation(); // Prevent map click from firing
                    
                    // Determine which tab is active
                    const activeTab = document.querySelector(".tab-button.active").getAttribute("data-tab");
                    
                    if (activeTab === "p2f") {
                        handleP2FGroceryStoreSelection(store);
                    } else if (activeTab === "f2p") {
                        handleF2PGroceryStoreSelection(store);
                    }
                });
            });

            // Create transit stop markers
            allFetchedTransitStops.forEach(stop => {
                const el = createMarkerElement("transit", stop.id);
                const isSelected = selectedTransitStops.some(s => s.id === stop.id);
                updateMarkerOpacity(el, isSelected);
                
                const marker = new mapboxgl.Marker(el)
                    .setLngLat([stop.longitude, stop.latitude])
                    .addTo(map);
                    
                transitStopMarkers[stop.id] = { marker: marker, data: stop, element: el };
                
                const popup = new mapboxgl.Popup({ offset: 25, closeButton: false, closeOnClick: false })
                    .setText(`TS: ${stop.id} (Demand: ${stop.demand}, Lat: ${stop.latitude.toFixed(4)}, Lon: ${stop.longitude.toFixed(4)})`);
                    
                marker.getElement().addEventListener("mouseenter", () => { 
                    map.getCanvas().style.cursor = "pointer"; 
                    popup.setLngLat([stop.longitude, stop.latitude]).addTo(map); 
                });
                
                marker.getElement().addEventListener("mouseleave", () => { 
                    map.getCanvas().style.cursor = ""; 
                    popup.remove(); 
                });
                
                marker.getElement().addEventListener("click", (e) => {
                    e.stopPropagation(); // Prevent map click from firing
                    handleTransitStopSelection(stop);
                });
                
                // Initially hide transit stops if F2P tab is active
                if (document.querySelector(".tab-button.active").getAttribute("data-tab") === "f2p") {
                    marker.getElement().style.display = "none";
                }
            });

            // Create community center markers
            allFetchedCommunityCenters.forEach(center => {
                const el = createMarkerElement("community", center.id);
                const isSelected = selectedCommunityCenters.some(c => c.id === center.id);
                updateMarkerOpacity(el, isSelected);
                
                const marker = new mapboxgl.Marker(el)
                    .setLngLat([center.longitude, center.latitude])
                    .addTo(map);
                    
                communityCenterMarkers[center.id] = { marker: marker, data: center, element: el };
                
                const popup = new mapboxgl.Popup({ offset: 25, closeButton: false, closeOnClick: false })
                    .setText(`CC: ${center.name || center.id} (Demand: ${center.demand}, Type: ${center.type}, Lat: ${center.latitude.toFixed(4)}, Lon: ${center.longitude.toFixed(4)})`);
                    
                marker.getElement().addEventListener("mouseenter", () => { 
                    map.getCanvas().style.cursor = "pointer"; 
                    popup.setLngLat([center.longitude, center.latitude]).addTo(map); 
                });
                
                marker.getElement().addEventListener("mouseleave", () => { 
                    map.getCanvas().style.cursor = ""; 
                    popup.remove(); 
                });
                
                marker.getElement().addEventListener("click", (e) => {
                    e.stopPropagation(); // Prevent map click from firing
                    handleCommunityCenterSelection(center);
                });
                
                // Initially hide community centers if P2F tab is active
                if (document.querySelector(".tab-button.active").getAttribute("data-tab") === "p2f") {
                    marker.getElement().style.display = "none";
                }
            });
            
            logToUI(`Displayed ${allFetchedGroceryStores.length} grocery stores, ${allFetchedTransitStops.length} transit stops, and ${allFetchedCommunityCenters.length} community centers.`);
        } catch (error) {
            logToUI("Error loading or displaying all locations: " + error.message);
            console.error("Error loading or displaying all locations:", error);
            alert("Could not load initial location data.");
        }
    }

    function handleP2FGroceryStoreSelection(store) {
        logToUI("handleP2FGroceryStoreSelection called with: " + store.id);
        const index = selectedGroceryStores.findIndex(s => s.id === store.id);
        if (index > -1) {
            selectedGroceryStores.splice(index, 1);
            logToUI("Removed grocery store from P2F: " + store.id);
            if (groceryStoreMarkers[store.id]) {
                updateMarkerOpacity(groceryStoreMarkers[store.id].element, false);
            }
        } else {
            selectedGroceryStores.push(store);
            logToUI("Added grocery store to P2F: " + store.id);
            if (groceryStoreMarkers[store.id]) {
                updateMarkerOpacity(groceryStoreMarkers[store.id].element, true);
            }
        }
        updateP2FShortNameMapping();
        displaySelectedP2FGroceryStores();
        updateP2FRunButtonState();
        
        // Disable recalibrate button when selection changes
        if (recalibrateButton) {
            recalibrateButton.disabled = true;
        }
        hideMessage(recalibrationMessage);
    }

    function handleF2PGroceryStoreSelection(store) {
        logToUI("handleF2PGroceryStoreSelection called with: " + store.id);
        const index = selectedF2PGroceryStores.findIndex(s => s.id === store.id);
        if (index > -1) {
            selectedF2PGroceryStores.splice(index, 1);
            logToUI("Removed grocery store from F2P: " + store.id);
            if (groceryStoreMarkers[store.id]) {
                updateMarkerOpacity(groceryStoreMarkers[store.id].element, false);
            }
        } else {
            selectedF2PGroceryStores.push(store);
            logToUI("Added grocery store to F2P: " + store.id);
            if (groceryStoreMarkers[store.id]) {
                updateMarkerOpacity(groceryStoreMarkers[store.id].element, true);
            }
        }
        updateF2PShortNameMapping();
        displaySelectedF2PGroceryStores();
        updateF2PRunButtonState();
        
        hideMessage(optimizationMessage);
    }

    function handleTransitStopSelection(stop) {
        logToUI("handleTransitStopSelection called with: " + stop.id);
        const index = selectedTransitStops.findIndex(s => s.id === stop.id);
        if (index > -1) {
            selectedTransitStops.splice(index, 1);
            logToUI("Removed transit stop: " + stop.id);
            if (transitStopMarkers[stop.id]) {
                updateMarkerOpacity(transitStopMarkers[stop.id].element, false);
            }
        } else {
            selectedTransitStops.push(stop);
            logToUI("Added transit stop: " + stop.id);
            if (transitStopMarkers[stop.id]) {
                updateMarkerOpacity(transitStopMarkers[stop.id].element, true);
            }
        }
        updateP2FShortNameMapping();
        displaySelectedTransitStops();
        updateP2FRunButtonState();
        
        // Disable recalibrate button when selection changes
        if (recalibrateButton) {
            recalibrateButton.disabled = true;
        }
        hideMessage(recalibrationMessage);
    }

    function handleCommunityCenterSelection(center) {
        logToUI("handleCommunityCenterSelection called with: " + center.id);
        const index = selectedCommunityCenters.findIndex(c => c.id === center.id);
        if (index > -1) {
            selectedCommunityCenters.splice(index, 1);
            logToUI("Removed community center: " + center.id);
            if (communityCenterMarkers[center.id]) {
                updateMarkerOpacity(communityCenterMarkers[center.id].element, false);
            }
        } else {
            selectedCommunityCenters.push(center);
            logToUI("Added community center: " + center.id);
            if (communityCenterMarkers[center.id]) {
                updateMarkerOpacity(communityCenterMarkers[center.id].element, true);
            }
        }
        updateF2PShortNameMapping();
        displaySelectedCommunityCenters();
        updateF2PRunButtonState();
        
        hideMessage(optimizationMessage);
    }

    function displaySelectedP2FGroceryStores() {
        if (!p2fGroceryStoresDisplay) return;
        p2fGroceryStoresDisplay.innerHTML = "";
        if (selectedGroceryStores.length === 0) {
            p2fGroceryStoresDisplay.innerHTML = "<p>None selected.</p>";
            return;
        }
        selectedGroceryStores.forEach(store => {
            const storeDiv = document.createElement("div");
            storeDiv.className = "selected-item-sidebar store-item"; 
            const shortName = p2fLocationIdToShortNameMap[store.id] || store.id;
            storeDiv.textContent = `${shortName}: ${store.name || store.id}`;
            storeDiv.dataset.id = store.id;

            storeDiv.addEventListener("mouseover", () => {
                highlightMarker(store.id, 'grocery');
            });
            
            storeDiv.addEventListener("mouseout", () => {
                resetMarkerHighlight(store.id, 'grocery');
            });
            
            p2fGroceryStoresDisplay.appendChild(storeDiv);
        });
    }

    function displaySelectedF2PGroceryStores() {
        if (!f2pGroceryStoresDisplay) return;
        f2pGroceryStoresDisplay.innerHTML = "";
        if (selectedF2PGroceryStores.length === 0) {
            f2pGroceryStoresDisplay.innerHTML = "<p>None selected.</p>";
            return;
        }
        selectedF2PGroceryStores.forEach(store => {
            const storeDiv = document.createElement("div");
            storeDiv.className = "selected-item-sidebar store-item"; 
            const shortName = f2pLocationIdToShortNameMap[store.id] || store.id;
            storeDiv.textContent = `${shortName}: ${store.name || store.id}`;
            storeDiv.dataset.id = store.id;

            storeDiv.addEventListener("mouseover", () => {
                highlightMarker(store.id, 'grocery');
            });
            
            storeDiv.addEventListener("mouseout", () => {
                resetMarkerHighlight(store.id, 'grocery');
            });
            
            f2pGroceryStoresDisplay.appendChild(storeDiv);
        });
    }

    function displaySelectedTransitStops() {
        if (!transitStopsListDiv) return;
        transitStopsListDiv.innerHTML = "";
        if (selectedTransitStops.length === 0) {
            transitStopsListDiv.innerHTML = "<p>None selected.</p>";
            return;
        }
        selectedTransitStops.forEach(stop => {
            const stopDiv = document.createElement("div");
            stopDiv.className = "selected-item-sidebar stop-item";
            const shortName = p2fLocationIdToShortNameMap[stop.id] || stop.id;
            stopDiv.textContent = `${shortName}: Demand ${stop.demand}`;
            stopDiv.dataset.id = stop.id;

            stopDiv.addEventListener("mouseover", () => {
                highlightMarker(stop.id, 'transit');
            });
            
            stopDiv.addEventListener("mouseout", () => {
                resetMarkerHighlight(stop.id, 'transit');
            });
            
            transitStopsListDiv.appendChild(stopDiv);
        });
    }

    function displaySelectedCommunityCenters() {
        if (!communityCentersListDiv) return;
        communityCentersListDiv.innerHTML = "";
        if (selectedCommunityCenters.length === 0) {
            communityCentersListDiv.innerHTML = "<p>None selected.</p>";
            return;
        }
        selectedCommunityCenters.forEach(center => {
            const centerDiv = document.createElement("div");
            centerDiv.className = "selected-item-sidebar center-item";
            const shortName = f2pLocationIdToShortNameMap[center.id] || center.id;
            // Round up demand and calculate cubic feet
            const demandRounded = Math.ceil(center.demand);
            const cubicFeet = demandRounded * 18; // 1 household = 18 cubic feet
            centerDiv.textContent = `${shortName}: ${center.name} (${demandRounded} households, ${cubicFeet} cubic ft)`;
            centerDiv.dataset.id = center.id;

            centerDiv.addEventListener("mouseover", () => {
                highlightMarker(center.id, 'community');
            });
            
            centerDiv.addEventListener("mouseout", () => {
                resetMarkerHighlight(center.id, 'community');
            });
            
            communityCentersListDiv.appendChild(centerDiv);
        });
    }

    function displayP2FSchedule(data) {
        if (!scheduleTableDiv) return;
        
        if (!data || !data.bus_details || data.bus_details.length === 0) {
            scheduleTableDiv.innerHTML = "<p>No schedule generated or no trips planned.</p>";
            return;
        }
        
        let scheduleHTML = `
            <div class="summary-section">
                <div class="summary-item">Total Buses: ${data.num_buses}</div>
                <div class="summary-item">Total Weekly Driving Time: ${Math.round(data.total_driving_time)} minutes</div>
                <div class="summary-item">Total Weekly Passengers Serviced: ${Math.round(data.total_passengers_serviced)} people</div>
            </div>
        `;
        
        data.bus_details.forEach(bus => {
            const busId = bus.bus_id;
            const busCapacity = bus.capacity;
            const totalWeeklyDrivingTime = Math.round(bus.total_weekly_driving_time);
            const assignedGroceryStoreId = bus.assigned_grocery_store_id;
            
            // Find the grocery store name
            let groceryStoreName = "Unknown";
            const groceryStore = allFetchedGroceryStores.find(gs => gs.id === assignedGroceryStoreId);
            if (groceryStore) {
                groceryStoreName = groceryStore.name || groceryStore.id;
            }
            
            scheduleHTML += `
                <div class="bus-schedule">
                    <h4>Bus ${busId} (Capacity: ${busCapacity}, Weekly Driving: ${totalWeeklyDrivingTime} min)</h4>
                    <p>Assigned to: ${groceryStoreName}</p>
            `;
            
            // Create a table for each day
            for (let day = 0; day < 5; day++) {
                const dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
                const dayName = dayNames[day];
                const dayTrips = bus.daily_schedule[day] || [];
                
                if (dayTrips.length > 0) {
                    scheduleHTML += `
                        <div class="day-schedule">
                            <h5>${dayName}</h5>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Trip</th>
                                        <th>Pax</th>
                                        <th>Duration</th>
                                        <th>Full Round Trip Route</th>
                                    </tr>
                                </thead>
                                <tbody>
                    `;
                    
                    dayTrips.forEach((trip, tripIndex) => {
                        const tripNumber = tripIndex + 1;
                        const passengers = trip.passengers;
                        const duration = Math.round(trip.duration_minutes);
                        
                        // Create route description
                        const pickupStops = trip.pickup_leg.route_description || [];
                        const dropoffStops = trip.dropoff_leg.route_description || [];
                        
                        // Map IDs to short names
                        const mappedPickup = pickupStops.map(id => p2fLocationIdToShortNameMap[id] || id);
                        const mappedDropoff = dropoffStops.map(id => p2fLocationIdToShortNameMap[id] || id);
                        
                        const routeDescription = [...mappedPickup, ...mappedDropoff.slice(1)].join(" → ");
                        
                        // Create unique IDs for this trip's routes
                        const pickupRouteId = `route-${busId}-day${day}-trip${tripIndex}-pickup`;
                        const dropoffRouteId = `route-${busId}-day${day}-trip${tripIndex}-dropoff`;
                        
                        scheduleHTML += `
                            <tr class="trip-row" 
                                data-pickup-route="${pickupRouteId}" 
                                data-dropoff-route="${dropoffRouteId}"
                                data-trip-id="${trip.trip_id}">
                                <td>${tripNumber}</td>
                                <td>${passengers}</td>
                                <td>${duration}</td>
                                <td>${routeDescription}</td>
                            </tr>
                        `;
                    });
                    
                    scheduleHTML += `
                                </tbody>
                            </table>
                        </div>
                    `;
                }
            }
            
            scheduleHTML += `</div>`;
        });
        
        scheduleTableDiv.innerHTML = scheduleHTML;
        
        // Add event listeners for trip rows
        const tripRows = scheduleTableDiv.querySelectorAll(".trip-row");
        tripRows.forEach(row => {
            row.addEventListener("mouseover", () => {
                const pickupRouteId = row.getAttribute("data-pickup-route");
                const dropoffRouteId = row.getAttribute("data-dropoff-route");
                
                // Highlight both routes
                highlightRoute(pickupRouteId);
                highlightRoute(dropoffRouteId);
                
                // Highlight the row
                highlightTableRow(row);
            });
            
            row.addEventListener("mouseout", () => {
                resetRouteHighlight();
                resetTableRowHighlight();
            });
        });
    }

    function displayF2PSchedule(data) {
        if (!scheduleTableDiv) return;
        
        if (!data || !data.truck_details || data.truck_details.length === 0) {
            scheduleTableDiv.innerHTML = "<p>No schedule generated or no trips planned.</p>";
            return;
        }
        
        let scheduleHTML = `
            <div class="summary-section">
                <div class="summary-item">Total Trucks: ${data.num_trucks}</div>
                <div class="summary-item">Total Weekly Driving Time: ${Math.round(data.total_driving_time)} minutes</div>
                <div class="summary-item">Total Weekly Load Delivered: ${Math.round(data.total_cubic_feet_delivered)} cubic ft</div>
            </div>
        `;
        
        data.truck_details.forEach(truck => {
            const truckId = truck.truck_id;
            const truckCapacity = truck.capacity;
            const totalWeeklyTime = Math.round(truck.total_weekly_time);
            const assignedGroceryStoreId = truck.assigned_grocery_store_id;
            
            // Find the grocery store name
            let groceryStoreName = "Unknown";
            const groceryStore = allFetchedGroceryStores.find(gs => gs.id === assignedGroceryStoreId);
            if (groceryStore) {
                groceryStoreName = groceryStore.name || groceryStore.id;
            }
            
            scheduleHTML += `
                <div class="truck-schedule">
                    <h4>Truck ${truckId} (Capacity: ${truckCapacity} cubic ft, Weekly Time: ${totalWeeklyTime} min)</h4>
                    <p>Assigned to: ${groceryStoreName}</p>
            `;
            
            // Create a table for each day
            for (let day = 0; day < 5; day++) {
                const dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
                const dayName = dayNames[day];
                const dayTrips = truck.daily_schedule[day] || [];
                
                if (dayTrips.length > 0) {
                    scheduleHTML += `
                        <div class="day-schedule">
                            <h5>${dayName}</h5>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Trip</th>
                                        <th>Cubic Feet</th>
                                        <th>Duration</th>
                                        <th>Full Trip Route</th>
                                    </tr>
                                </thead>
                                <tbody>
                    `;
                    
                    // Create route description for all legs
                    let routeDescription = "";
                    let routeIds = [];

                    dayTrips.forEach((trip, tripIndex) => {
                        const tripNumber = tripIndex + 1;
                        const cubicFeet = trip.cubic_feet;
                        const duration = Math.round(trip.duration_minutes);
                        
                        // Map IDs to short names
                        const mappedLocations = trip.route_description.map(id => f2pLocationIdToShortNameMap[id] || id);
                            
                        // if (tripIndex > 0) {
                        //     routeDescription += " → ";
                        // }
                        
                        routeDescription = mappedLocations.join(" → ");

                        // Create unique ID for this trips's route
                        const routeId = `route-${truckId}-day${day}-trip${tripIndex}`;
                        routeIds.push(routeId);
                        
                        scheduleHTML += `
                            <tr class="trip-row" 
                                data-route-ids="${routeIds.join(',')}"
                                data-trip-id="${trip.trip_id}">
                                <td>${tripNumber}</td>
                                <td>${cubicFeet}</td>
                                <td>${duration}</td>
                                <td>${routeDescription}</td>
                            </tr>
                        `;
                    });
                    
                    scheduleHTML += `
                                </tbody>
                            </table>
                        </div>
                    `;
                }
            }
            
            scheduleHTML += `</div>`;
        });
        
        scheduleTableDiv.innerHTML = scheduleHTML;
        
        // Add event listeners for trip rows
        const tripRows = scheduleTableDiv.querySelectorAll(".trip-row");
        tripRows.forEach(row => {
            row.addEventListener("mouseover", () => {
                const routeIds = row.getAttribute("data-route-ids").split(',');
                
                // Highlight all routes for this trip
                routeIds.forEach(routeId => {
                    highlightRoute(routeId);
                });
                
                // Highlight the row
                highlightTableRow(row);
            });
            
            row.addEventListener("mouseout", () => {
                resetRouteHighlight();
                resetTableRowHighlight();
            });
        });
    }

    function addRouteToMap(routeId, geometry, color) {
        if (map.getSource(routeId)) {
            map.removeSource(routeId);
        }
        
        if (map.getLayer(routeId)) {
            map.removeLayer(routeId);
        }
        
        map.addSource(routeId, {
            'type': 'geojson',
            'data': {
                'type': 'Feature',
                'properties': {},
                'geometry': geometry
            }
        });
        
        map.addLayer({
            'id': routeId,
            'type': 'line',
            'source': routeId,
            'layout': {
                'line-join': 'round',
                'line-cap': 'round'
            },
            'paint': {
                'line-color': color,
                'line-width': 3,
                'line-opacity': 0.6
            }
        });
    }

    async function runP2FOptimization() {
        if (selectedGroceryStores.length === 0 || selectedTransitStops.length === 0) {
            alert("Please select at least one grocery store and one transit stop.");
            return;
        }
        
        try {
            // Clear previous routes
            clearAllRouteLayers();
            
            // Disable run button during optimization
            if (runP2FOptimizationButton) {
                runP2FOptimizationButton.disabled = true;
                runP2FOptimizationButton.textContent = "Optimizing...";
            }
            
            // Get bus capacity
            const busCapacity = parseInt(document.getElementById("bus-capacity").value) || 15;
            
            // Prepare data for API
            const requestData = {
                selected_grocery_stores: selectedGroceryStores,
                selected_transit_stops: selectedTransitStops,
                bus_capacity: busCapacity
            };
            
            // Call optimization API
            const response = await fetch("/optimize", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Optimization failed");
            }
            
            const data = await response.json();
            
            // Add routes to map
            if (data.bus_details) {
                data.bus_details.forEach(bus => {
                    for (let day = 0; day < 5; day++) {
                        const dayTrips = bus.daily_schedule[day] || [];
                        dayTrips.forEach((trip, tripIndex) => {
                            // Add pickup leg route
                            if (trip.pickup_leg && trip.pickup_leg.geometry) {
                                const pickupRouteId = `route-${bus.bus_id}-day${day}-trip${tripIndex}-pickup`;
                                addRouteToMap(pickupRouteId, trip.pickup_leg.geometry, "#3388ff"); // Blue for pickup
                            }
                            
                            // Add dropoff leg route
                            if (trip.dropoff_leg && trip.dropoff_leg.geometry) {
                                const dropoffRouteId = `route-${bus.bus_id}-day${day}-trip${tripIndex}-dropoff`;
                                addRouteToMap(dropoffRouteId, trip.dropoff_leg.geometry, "#ff3388"); // Pink for dropoff
                            }
                        });
                    }
                });
            }
            
            // Display schedule
            displayP2FSchedule(data);
            
            // Enable recalibrate button
            if (recalibrateButton) {
                recalibrateButton.disabled = false;
            }
            
        } catch (error) {
            console.error("Optimization error:", error);
            alert("Optimization failed: " + error.message);
        } finally {
            // Re-enable run button
            if (runP2FOptimizationButton) {
                runP2FOptimizationButton.disabled = false;
                runP2FOptimizationButton.textContent = "Run P2F Optimization";
            }
        }
    }

    async function runF2POptimization() {
        if (selectedF2PGroceryStores.length === 0 || selectedCommunityCenters.length === 0) {
            alert("Please select at least one grocery store and one community center.");
            return;
        }
        
        try {
            // Clear previous routes
            clearAllRouteLayers();
            
            // Disable run button during optimization
            if (runF2POptimizationButton) {
                runF2POptimizationButton.disabled = true;
                runF2POptimizationButton.textContent = "Optimizing...";
            }
            
            // Disable recalibrate button during optimization
            const recalibrateF2PButton = document.getElementById("recalibrate-f2p-button");
            if (recalibrateF2PButton) {
                recalibrateF2PButton.disabled = true;
            }
            
            // Hide any previous messages
            const f2pRecalibrationMessage = document.getElementById("f2p-recalibration-message");
            hideMessage(f2pRecalibrationMessage);
            hideMessage(optimizationMessage);
            
            // Get truck capacity
            const truckCapacity = parseInt(document.getElementById("truck-capacity").value) || 270;
            
            // Prepare data for API
            const requestData = {
                selected_grocery_stores: selectedF2PGroceryStores,
                selected_community_centers: selectedCommunityCenters,
                truck_capacity: truckCapacity
            };
            
            // Call optimization API
            const response = await fetch("/optimize_f2p", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "F2P Optimization failed");
            }
            
            const data = await response.json();
            
            // Add routes to map
            if (data.truck_details) {
                data.truck_details.forEach(truck => {
                    for (let day = 0; day < 5; day++) {
                        const dayTrips = truck.daily_schedule[day] || [];
                        dayTrips.forEach((trip, tripIndex) => {
                            // Add all trip routes
                            if (trip.geometry) {
                                const routeId = `route-${truck.truck_id}-day${day}-trip${tripIndex}`;
                                // Use different colors for different trips
                                const colors = ["#3388ff", "#ff3388", "#33ff88", "#ff8833", "#8833ff"];
                                const color = colors[tripIndex % colors.length];
                                addRouteToMap(routeId, trip.geometry, color);
                            }
                        });
                    }
                });
            }
            
            // Display schedule
            displayF2PSchedule(data);
            
            // Show success message
            showMessage(optimizationMessage, "F2P Optimization completed successfully!", "success");
            
            // Enable recalibrate button after successful optimization
            if (recalibrateF2PButton) {
                recalibrateF2PButton.disabled = false;
            }
            
        } catch (error) {
            console.error("F2P Optimization error:", error);
            alert("F2P Optimization failed: " + error.message);
            showMessage(optimizationMessage, "F2P Optimization failed: " + error.message, "warning");
        } finally {
            // Re-enable run button
            if (runF2POptimizationButton) {
                runF2POptimizationButton.disabled = false;
                runF2POptimizationButton.textContent = "Run F2P Optimization";
            }
        }
    }

    async function recalibrateP2F() {
        try {
            // Disable recalibrate button during recalibration
            if (recalibrateButton) {
                recalibrateButton.disabled = true;
                recalibrateButton.textContent = "Recalibrating...";
            }
            
            // Call recalibration API
            const response = await fetch("/recalibrate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Recalibration failed");
            }
            
            const data = await response.json();
            
            // Clear previous routes
            clearAllRouteLayers();
            
            // Add routes to map
            if (data.bus_details) {
                data.bus_details.forEach(bus => {
                    for (let day = 0; day < 5; day++) {
                        const dayTrips = bus.daily_schedule[day] || [];
                        dayTrips.forEach((trip, tripIndex) => {
                            // Add pickup leg route
                            if (trip.pickup_leg && trip.pickup_leg.geometry) {
                                const pickupRouteId = `route-${bus.bus_id}-day${day}-trip${tripIndex}-pickup`;
                                addRouteToMap(pickupRouteId, trip.pickup_leg.geometry, "#3388ff"); // Blue for pickup
                            }
                            
                            // Add dropoff leg route
                            if (trip.dropoff_leg && trip.dropoff_leg.geometry) {
                                const dropoffRouteId = `route-${bus.bus_id}-day${day}-trip${tripIndex}-dropoff`;
                                addRouteToMap(dropoffRouteId, trip.dropoff_leg.geometry, "#ff3388"); // Pink for dropoff
                            }
                        });
                    }
                });
            }
            
            // Display schedule
            displayP2FSchedule(data);
            
            // Show recalibration message
            if (data.recalibration_status === "improved") {
                showMessage(recalibrationMessage, data.recalibration_message, "success");
            } else if (data.recalibration_status === "no_improvement") {
                showMessage(recalibrationMessage, data.recalibration_message, "warning");
            } else {
                showMessage(recalibrationMessage, data.recalibration_message, "info");
            }
            
        } catch (error) {
            console.error("Recalibration error:", error);
            alert("Recalibration failed: " + error.message);
            showMessage(recalibrationMessage, "Recalibration failed: " + error.message, "warning");
        } finally {
            // Re-enable recalibrate button
            if (recalibrateButton) {
                recalibrateButton.disabled = false;
                recalibrateButton.textContent = "Recalibrate";
            }
        }
    }

    async function recalibrateF2P() {
        try {
            // Disable recalibrate button during recalibration
            const recalibrateF2PButton = document.getElementById("recalibrate-f2p-button");
            if (recalibrateF2PButton) {
                recalibrateF2PButton.disabled = true;
                recalibrateF2PButton.textContent = "Recalibrating...";
            }
            
            const f2pRecalibrationMessage = document.getElementById("f2p-recalibration-message");
            hideMessage(f2pRecalibrationMessage);
            
            // Call recalibration API
            const response = await fetch("/recalibrate_f2p", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "F2P Recalibration failed");
            }
            
            const data = await response.json();
            
            // Clear previous routes
            clearAllRouteLayers();
            
            // Add routes to map
            if (data.truck_details) {
                data.truck_details.forEach(truck => {
                    for (let day = 0; day < 5; day++) {
                        const dayTrips = truck.daily_schedule[day] || [];
                        dayTrips.forEach((trip, tripIndex) => {
                            // Add all trip routes
                            if (trip.geometry) {
                                const routeId = `route-${truck.truck_id}-day${day}-trip${tripIndex}`;
                                // Use different colors for different trips
                                const colors = ["#3388ff", "#ff3388", "#33ff88", "#ff8833", "#8833ff"];
                                const color = colors[tripIndex % colors.length];
                                addRouteToMap(routeId, trip.geometry, color);
                            }
                        });
                    }
                });
            }
            
            // Display schedule
            displayF2PSchedule(data);
            
            // Show recalibration message
            if (data.recalibration_status === "improved") {
                showMessage(f2pRecalibrationMessage, data.recalibration_message, "success");
            } else if (data.recalibration_status === "no_improvement") {
                showMessage(f2pRecalibrationMessage, data.recalibration_message, "warning");
            } else {
                showMessage(f2pRecalibrationMessage, data.recalibration_message, "info");
            }
            
        } catch (error) {
            console.error("F2P Recalibration error:", error);
            alert("F2P Recalibration failed: " + error.message);
            const f2pRecalibrationMessage = document.getElementById("f2p-recalibration-message");
            showMessage(f2pRecalibrationMessage, "F2P Recalibration failed: " + error.message, "warning");
        } finally {
            // Re-enable recalibrate button
            const recalibrateF2PButton = document.getElementById("recalibrate-f2p-button");
            if (recalibrateF2PButton) {
                recalibrateF2PButton.disabled = false;
                recalibrateF2PButton.textContent = "Recalibrate";
            }
        }
    }

    // Initialize the application
    loadAndDisplayAllLocations();
    
    // Set up event listeners for optimization buttons
    if (runP2FOptimizationButton) {
        runP2FOptimizationButton.addEventListener("click", runP2FOptimization);
    }
    
    if (runF2POptimizationButton) {
        runF2POptimizationButton.addEventListener("click", runF2POptimization);
    }
    
    if (recalibrateButton) {
        recalibrateButton.addEventListener("click", recalibrateP2F);
    }
    
    // Set up event listener for F2P recalibrate button
    const recalibrateF2PButton = document.getElementById("recalibrate-f2p-button");
    if (recalibrateF2PButton) {
        recalibrateF2PButton.addEventListener("click", recalibrateF2P);
    }
    
    // Initialize the active tab
    const activeTab = document.querySelector(".tab-button.active").getAttribute("data-tab");
    updateMarkerVisibility(activeTab);
});
