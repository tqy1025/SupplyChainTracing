# SupplyChainTracing


## Project Structure

This repository is organized into five main directories, following the logical flow of our measurement study: from data preparation to final relationship validation.

### 📂 1. Data
Contains the metadata and auxiliary records required for device identification and hostname mapping.
*   `device_list.csv`: The master list of IoT devices analyzed in this study, including brand information.
*   `device_type.csv`: Classification of devices into functional categories (e.g., Camera, Hub, Plug).
*   `device_rdns.rar`: Pre-resolved Reverse DNS (rDNS) records used for Hostname Mapping.

### 📂 2. Flow_preprocess
Scripts for transforming raw network traffic into structured, semantic-labeled flows.
*   Includes modules for flow extraction, rigorous filtering, and dual-layer hostname mapping (Vanity and Canonical).

### 📂 3. Implicit_Identification
Implementation of the static endpoint matching framework.
*   Detects **Implicit Relationships** by identifying shared service endpoints across different brands and performs cross-layer deduplication.

### 📂 4. Latent_Detection
Implementation of behavioral similarity analysis using the **PS-DTW** metric.
*   Contains scripts for hierarchical clustering and the three-pattern validation strategy (Co-occurrence, Composite Emergence, and Brand Corroboration).

### 📂 5. Result
The primary outputs of our framework, categorized by the type of relationship detected.

#### 📁 Implicit_Results
*   `Targets_Implicit_DedupByFlow.csv`: List of verified shared endpoints (IPs/Domains) that establish implicit supply chain ties.

#### 📁 Latent_Results
*   `4.1.Clusters_Validated_by_Manual.csv`: Clusters that have been corroborated through manual audit of business relationships.（Brand Corroboration）
*   `6.FINAL.Consolidated_..._v2.4.csv`: Validated component clusters, validated through multi-rule matching. (Rule Matching)
