# AlgoMLB: Architecture Overview

AlgoMLB is a professional-grade machine learning and analytics platform for Major League Baseball. It is designed to ingest, process, and analyze vast amounts of Statcast-level data to provide predictive insights and financial strategy management.

## System Philosophy

The project adheres to several core engineering principles:

1.  **Strict Layering**: A unidirectional "Import Ladder" prevents circular dependencies and ensures a stable dependency graph.
2.  **Type Safety**: Pydantic is used extensively for data validation atEvery boundary (API, DB, and internal logic).
3.  **Auditability**: Structured JSONL logging and an `AgentResult` protocol ensure that both humans and automated agents can monitor system health.
4.  **Resilience**: Comprehensive handling of null values (NaN/None) and edge cases is built into the domain and UI layers.

## The Import Ladder

To ensure long-term maintainability, imports must only flow downwards. A module at a higher level can import from lower levels, but never vice-versa.

```text
[7] UI        (Streamlit dashboard)
 ╿
[6] CLI       (Command-line tools)
 ╿
[5] ML        (Features, models, quant service)
 ╿
[4] INGESTION (API clients, web scrapers, orchestrators)
 ╿
[3] DB        (SQLAlchemy ORM, repository pattern)
 ╿
[2] DOMAIN    (Pydantic models, constants, physics)
 ╿
[1] CORE/CONFIG (Logging, settings, global utilities)
```

## Module Responsibilities

| Layer | Module | Responsibility |
| :--- | :--- | :--- |
| **Interface** | `algomlb.ui` | Interactive data visualization and real-time monitoring. |
| **Interface** | `algomlb.cli` | System entry points for manual syncing, training, and data management. |
| **Intelligence** | `algomlb.ml` | Mathematical modeling, feature engineering, and market "edge" calculation. |
| **Data Acquisition** | `algomlb.ingestion` | Extracting data from Statcast, Gumbo (MLB API), Odds APIs, and Umpire Scorecards. |
| **Persistence** | `algomlb.db` | Mapping domain objects to PostgreSQL and handling bulk I/O operations. |
| **Logic** | `algomlb.domain` | Defining the "shape" of baseball data and universal constants. |
| **Foundation** | `algomlb.core` | Logging, standardized agent output, and low-level I/O. |
| **Foundation** | `algomlb.config` | Unified configuration management from environment and YAML sources. |

## Data Flow

1.  **Ingestion**: Orchestrators fetch raw data from external APIs.
2.  **Normalization**: Data is validated against `domain` models.
3.  **Persistence**: The `repository` maps domain models to ORM models and saves them to the database.
4.  **Processing**: The ML layer reads from the DB to compute rolling features, quant metrics, and model predictions.
5.  **Consumption**: The CLI or UI displays the processed insights to the user.
