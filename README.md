Triathlon Training Analysis Pipeline

A reproducible data pipeline for analyzing multi-year triathlon training data exported from Apple Health.  
Compares training data from the 5 month window before previous 2 Ironman 70.3 triathlons and compares training hours, hr-based load,
average workout hr, and cumulative hours for current race build.

This project parses Apple Health XML exports, resolves duplicate cycling workouts (Strava + Apple Watch double-logging), 
merges accurate distance and heart rate data into canonical workout records, engineers training load metrics, and produces 
longitudinal build comparisons and dashboards.

export.xml
    ↓
Workout + WorkoutStatistics extraction
    ↓
Heart rate record extraction
    ↓
Efficient time-window HR → workout assignment
    ↓
Cycling duplicate clustering (overlap-based)
    ↓
Merge Strava distance + Apple Watch HR
    ↓
Feature engineering (HR intensity + load proxy)
    ↓
Weekly aggregation by build cycle
    ↓
Outputs:
    - Enriched workout dataset
    - Weekly comparison table
    - 2×2 dashboard visualization

Data: data/export.xml
Run: run_pipeline.py
