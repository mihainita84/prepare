# PREPARE | Romania Wildfire Portal v13

Streamlit prototype for **PREPARE WP4 – Forest fires and related risks**.

## v13 – spread-model correction

This release keeps the working v12 control-line and MP4/GIF animation modules and corrects two issues observed during testing.

### 1. Streamlit error after video generation

The error was in the application code, not in Streamlit. `arrival_overlay_png()` created a PNG in an in-memory buffer but did not return the bytes. On the next Streamlit rerun, `_build_spread_result_map()` received `None` and `base64.b64encode(None)` raised a `TypeError`.

v13 explicitly returns `buf.getvalue()`.

### 2. Stronger landscape control of fire propagation

The experimental travel-time model now gives stronger spatial control to the input layers:

- **ESA WorldCover 2021**
  - unknown/no-data cells no longer receive a generic positive rate of spread;
  - built-up, water and snow/ice remain non-burnable;
  - baseline screening ROS coefficients were reduced, especially for tree cover and cropland.
- **ESA CCI Biomass 2024**
  - used only as a woody-fuel continuity/structure modifier for tree and shrub classes;
  - low AGB strongly reduces spread potential;
  - medium/high AGB saturates near a neutral factor instead of increasing ROS without bound;
  - missing CCI data are treated as neutral rather than as true zero biomass.
- **Copernicus DEM GLO-30**
  - directional elevation change affects every propagation edge;
  - upslope travel is accelerated and downslope travel is reduced;
  - data-coverage diagnostics are shown in the Streamlit UI.
- **Fuel continuity**
  - the 16-direction neighbourhood can no longer jump across a one-cell water/built-up/no-fuel gap;
  - control-line geometry remains checked separately as an exact buffered vector barrier.
- **ROS plausibility caps**
  - class-specific caps prevent multiplication of wind + slope from producing unrealistically large rates in low-resolution cells.

The exported MP4/GIF also visually modulates woody cells with the CCI biomass factor and uses a DEM hillshade so the two inputs are visible in the animation background.

## Important scientific limitation

ESA CCI AGB is **woody above-ground biomass**, not direct fine surface-fuel load. It is therefore used here as a landscape structure/continuity modifier and should not be interpreted as a calibrated fuel load for Rothermel/BehavePlus calculations.

This remains an experimental screening/scenario-comparison model. It is not an operational fire-behaviour or firefighter-deployment model.

## Run on Windows

First run:

`setup_and_run.bat`

Later runs:

`run.bat`

The project uses its own `.venv`.
