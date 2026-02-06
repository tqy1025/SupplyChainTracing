# Implicit Relationship Identification

This module implements the static endpoint matching framework to uncover "Implicit Relationships" as described in **Section 3.3**.

## Core Logic

- **`1.Implicit_Identify.py`**: 
  - **Endpoint Matching**: Aggregates devices into candidate interaction groups based on shared IP addresses, vanity hostnames, or canonical hostnames.
  - **Explicit Relationship Pruning**: Filters out pairs of devices belonging to the same brand to isolate cross-brand dependencies.
  - **Explicit Generic Public Services**: Filters out pairs of devices belonging to generic public services (using the curated blocklist discussed in **Section 3.3.5**).
  - **Implicit Link Isolation**: Identifies pairs of different brands that intersect at a specific service endpoint (e.g., a shared PaaS platform), representing a "Commoditized Interface."

## Key Terminology
- **Implicit Relationship**: Direct evidence of shared cloud infrastructure revealed through identical, non-public network endpoints.
