# Latent Relationship Detection

This directory implements the behavioral analysis pipeline using our proposed **PS-DTW** metric to uncover shared components hidden behind diverging endpoints (**Section 3.4**).

## Execution Steps

### I. Clustering Pipeline (Section 3.4.3)
1. **`1.In_Device_Flow_Cluster.py`**: (Stage I) Compresses redundant intra-device flows to extract unique behavioral "exemplars" using Hierarchical Agglomerative Clustering (HAC).
2. **`2.Global_Flow_Cluster.py`**: (Stage II) Applies HAC with the **Packet-Semantic Weighted Dynamic Time Warping (PS-DTW)** similarity metric to discover cross-device behavioral similarities.

### II. Filtering and Refinement
3. **`3.Exclude_implicit_Generic.py`**: Prunes candidate clusters that are either already explained by implicit relationships or correspond to generic public services (using the curated blocklist discussed in **Section 3.3.5**).

### III. Candidate Verification (Section 3.4.4)
4. **`4.Brand_Corroboration.py`**: Cross-references clusters with documented business relationships (OEM agreements, strategic partnerships). （An LLM version is provided here; in the experiment, this was done manually.）
5. **Validation Rules**:
   - **`5.1.multi_cluster_cooccurrence.py`**: Checks if the same set of devices recur in multiple distinct functional clusters.
   - **`5.2.composite_pattern.py`**: Identifies "suite" clusters that can be reconstructed by smaller, validated functional units.
   - **`5.3.known_relationship.py`**: Reinforces behavioral findings using known explicit/implicit ties.

### IV. Results Generation
6. **`6.rule_matching_statistics.py`**: Calculates final validation yields and generates statistics.

## Core Algorithm
**PS-DTW**: A customized DTW algorithm that assigns penalties and discounts based on packet semantic zones (Control signaling, General payload, and MTU signatures).
