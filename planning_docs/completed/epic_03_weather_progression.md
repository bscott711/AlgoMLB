# Implementation Plan - Epic 3: Open-Meteo Weather Progression & Forecast Surprise

## 1. Overview

Map environmental conditions from 1st pitch (T0) through the 9th inning (T4) to capture the "environmental arc" and identify market-surprise signals by comparing actual conditions to opening-odds forecasts.

## 2. Database Schema

Create a new table `openmeteo_weather_progression` to store hourly slices and derived features.

### SQLAlchemy Model (`src/algomlb/db/models.py`)

```python
from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

class OpenMeteoWeatherProgressionORM(Base):
    __tablename__ = "openmeteo_weather_progression"

    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("game_results.game_id"), primary_key=True)
    
    # T0-T4 Hourly Progression: Raw values
    # Temperature (F), Wind Speed (MPH), Wind Dir (DEG)
    temp_t0_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t0: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t0: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    
    temp_t1_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t1: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t1: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    
    temp_t2_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t2: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t2: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    
    temp_t3_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t3: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t3: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    
    temp_t4_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_t4: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_t4: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    
    # Supplemental first pitch (T0) details
    humidity_t0: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    precip_t0_mm: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    pressure_t0_hpa: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    cloud_cover_t0_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))

    # Derived Progression Aggregates
    temp_delta_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))   # T3 - T0
    temp_min_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_speed_max_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_dir_variance_deg: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    headwind_t0_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    headwind_t3_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    headwind_delta_game: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    crosswind_t0_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    wind_shift_gt_45deg: Mapped[Optional[bool]] = mapped_column(Boolean)
    temp_drop_gt_10f: Mapped[Optional[bool]] = mapped_column(Boolean)
    precip_any_game: Mapped[Optional[bool]] = mapped_column(Boolean)

    # T-24h Opening Forecast
    forecast_temp_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_wind_speed_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_wind_dir_deg: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_headwind_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_crosswind_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_precip_prob: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_cloud_cover_pct: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    forecast_source: Mapped[Optional[str]] = mapped_column(String(50))

    # Market Surprise Deltas (Actual - Forecast)
    delta_temp_f: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    delta_wind_speed_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    delta_headwind_mph: Mapped[Optional[float]] = mapped_column(Numeric(5, 1))
    delta_precip_mm: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))

    era5_model_used: Mapped[Optional[str]] = mapped_column(String(50))
    fetched_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
```

## 3. Ingestion Workflow (`src/algomlb/ingestion/openmeteo_ingester.py`)

### Input Data

- Stadium Key, Lat/Lon, Game Date, First Pitch Hour (Local), Timezone.
- Home Plate Bearing (from Epic 1).

### Logic Steps

1. **Archive Fetch**: Fetch ERA5 hourly history.
2. **Forecast Fetch**: Fetch Historical Forecast for T-24h. If unavailable for older games (pre-2022), fallback to ERA5 proxy (as requested).
3. **Calculation**:
   - Apply `wind_components()` to T0-T4.
   - Compute `circular_std()` of wind directions.
   - Calculate aggregates (Min/Max/Delta).
   - Evaluate boolean flags (`wind_shift_gt_45deg`, etc.).
4. **Market Surprises**: Compare forecast state at first pitch with actual T0 state.

## 4. Verification Plan

- **Unit Tests**:
  - Mock multi-point array responses from Open-Meteo.
  - Verify `temp_drop_gt_10f` logic specifically near the boundary.
  - Verify circular statistics for wind variance.
- **Backfill Strategy**:
  - Implement seasonal backfill in `orchestrator.py` or separate script.
  - Rate-limiting for Open-Meteo API (free tier limits).
