# PREPARE | Romania Wildfire Portal v12

Streamlit prototype for **PREPARE WP4 – Forest fires and related risks**.

## v12 changes

### Control lines are true propagation barriers

The fire-spread graph still uses 16 directional connections, but control lines are no longer checked only as raster cell-centre masks.

1. Each user-drawn line is transformed to the simulator projected CRS.
2. The line is buffered to the selected real width (for example 200 m).
3. Every candidate propagation edge is represented as a vector segment between source and destination cell centres.
4. The segment is intersected with the buffered control strip.
5. In **Complete barrier** mode, any intersecting propagation edge is rejected.
6. In **Partial barrier** mode, the crossing portion receives an explicit travel-time penalty according to the selected effectiveness.

This prevents long 16-direction graph links from numerically jumping across a wide control line. Fire can still go around the *ends* of a line if the geometry does not close the sector.

The simulator reports diagnostic values such as:

- barrier cells,
- blocked propagation crossings,
- penalized crossings,
- control mode and width.

### Video export

The Play tab now includes **Animation export**:

- MP4/H.264 export,
- animated GIF export,
- selectable frame interval,
- selectable playback FPS.

The exported animation uses the analysed ESA WorldCover grid as background, cumulative fire arrival cells, ignition point and the blue control strip. OpenStreetMap tiles are not embedded in the exported file.

Dependencies added:

- `imageio`
- `imageio-ffmpeg`

## Run

First Windows run:

`setup_and_run.bat`

Later runs:

`run.bat`

The application uses its own `.venv`.
