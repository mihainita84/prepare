# PREPARE | Romania Wildfire Portal v10

Streamlit prototype for **WP4 – Forest fires and related risks** in PREPARE.

## v10: Fire Spread Scenario Simulator

After the normal AOI analysis, the portal now opens a second-stage **Fire Spread Scenario Simulator**.

### Inputs used by the simulator

- ESA WorldCover 2021 fuel / land-cover classes
- ESA CCI Biomass v7.0 (2024) as a moderate fuel-loading modifier
- Copernicus DEM GLO-30 from the public AWS COG archive for terrain / slope
- current Open-Meteo wind speed and **wind direction**
- user-adjustable fine-fuel-moisture scenario value
- optional user-drawn hypothetical control lines

### Interaction

The user can:

1. use the most recent FIRMS hotspot as the fire seed;
2. use the AOI centroid;
3. click a new ignition/seed point inside the AOI;
4. change wind speed;
5. change wind direction;
6. change fine-fuel moisture;
7. change simulation horizon from 0.5 to 12 hours;
8. draw one or more blue hypothetical control lines;
9. change line width and hypothetical effectiveness;
10. rerun the scenario;
11. use a time slider to see the front at successive times.

### Outputs

- modelled fire-arrival map
- area reached at the selected display time
- area reached by the scenario horizon
- maximum modelled spread distance
- modelled head-fire bearing
- maximum scenario rate of spread
- AOI-boundary warning
- comparison of a drawn control-line scenario with the same simulation without the line

### Model status

This is an **experimental directional least-cost / travel-time screening model**. It is inspired by the main variables used in established surface-fire behaviour modelling (fuel, fuel moisture, wind and slope), but it is **not a calibrated operational Rothermel / BehavePlus implementation**.

The internal baseline rate-of-spread coefficients are for scenario comparison and should be locally calibrated before any operational research use.

The simulator must **not** be used to prescribe firefighter positions, safety zones or escape routes. Those require incident-command assessment and field information.

## Existing portal modules retained

- OpenStreetMap + NASA FIRMS active fires
- maximum AOI 5,000 ha / 50 km²
- ESA WorldCover 2021
- ESA CCI Biomass 2024
- EFFIS European Fuel Map
- GWIS MODIS hotspots
- GWIS ECMWF FWI / FFMC / DC / ISI / BUI
- GWIS NFDRS IC / ROS
- AOI danger-class interpretation
- weather and soil moisture
- EFFIS burned-area intersection
- HTML / JSON / GeoJSON downloads
- Romanian / English interface

## Windows installation

First run:

`setup_and_run.bat`

Later runs:

`run.bat`

The application creates and uses its own `.venv` and does not depend on the global `C:\Python312\Scripts` folder.
