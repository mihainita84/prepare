# PREPARE | Romania Wildfire Portal v11

Streamlit prototype for **PREPARE WP4 – Forest fires and related risks**.

## v11 changes

### Fire-spread simulator moved to Stage 2
The simulator now appears **after the full current-situation assessment**: weather/soil moisture, forest cover/biomass, EFFIS/GWIS products, thematic gallery, WorldCover composition, 72-hour forecast and FIRMS detections.

### Propagation bug fixed
The previous version stored arrival times as `float32` and compared them with heap times using exact equality. This caused propagation to stop after the ignition cell and its first neighbour ring. v11 uses `float64` arrival times and tolerant stale-queue checks.

### More realistic spatial propagation
- adaptive grid: **50 m** for AOI ≤ 1,500 ha, **75 m** for 1,500–3,500 ha, **100 m** above 3,500 ha;
- **16 directional propagation links** instead of only 8 neighbours;
- directional wind multiplier;
- directional upslope/downhill multiplier from Copernicus DEM;
- WorldCover fuel coefficients;
- ESA CCI biomass modifier;
- fine-fuel moisture scenario multiplier;
- control-line crossing checks prevent long directional links from jumping over a drawn barrier.

### 24-hour simulations
Simulation horizon can now be set from **0.5 to 24 hours**.

### Play animation
After running a scenario, the first result tab contains a Leaflet TimeDimension animation with a **Play button**. The footprint is cumulative: once a grid cell is reached it remains visible while later arrival-time cells are added.

Animation time step is automatically chosen:
- ≤ 6 h: 15 min
- 6–12 h: 30 min
- > 12 h: 60 min

The second result tab keeps the manual time slider for inspecting a specific moment.

## Important limitation
This remains an **experimental screening / scenario-comparison model**, not a validated operational fire-behaviour or firefighter deployment model. Use it to compare scenarios and identify potentially critical sectors, not to define safety zones, escape routes or exact crew locations.
