# Shared Test Footage Directory (`test_footage/`)

This shared folder contains staged test video clips used across AI detection, behavior analysis, and backend ingestion pipelines.

## Expected Video Formats & Specifications

- **Container / Codec:** MP4 (`.mp4`), H.264 video stream
- **Resolution:** 1080p (`1920x1080`) or 720p (`1280x720`)
- **Framerate:** 15 to 30 FPS
- **Color Space:** BGR (8-bit per channel)

## Standard Naming Conventions

| Category | Example Filename | Purpose |
|---|---|---|
| Daylight Surveillance | `day_crossing_01.mp4` | Human & vehicle detection, PAR clothing color clustering |
| Night / Low-Light | `night_low_light_01.mp4` | Low-light detection recall, Zero-DCE enhancement & IR fallback |
| License Plate Test | `plate_vehicle_01.mp4` | Indian vehicle license plate localization and PaddleOCR extraction |
| Geo-Fence Crossing | `geofence_breach_01.mp4` | Ground-contact foot coordinate tracking & rule engine breach |
| Degraded Visibility | `fog_dust_01.mp4` | Contrast collapse and `tau_visibility` degradation handling |

*Note: Large media files should remain local or gitignored if exceeding repository size constraints.*
