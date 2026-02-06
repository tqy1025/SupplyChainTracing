# Flow Preprocessing

This directory contains scripts for transforming raw network traffic into structured, semantic-labeled flows, as described in **Section 3.2** of the paper.

## Workflow

1. **`1.Flow_extract.py`**: Parses raw PCAP files and reconstructs session-level bidirectional TCP/UDP flows. 
2. **`2.Flow_filter.py`**: Applies the four rigorous filtering criteria:
   - LAN traffic exclusion.
   - Minimum length constraint (flows with < 5 packets are filtered).
   - Protocol filtering (removing DNS control flows while retaining application-layer data).
   - Device filtering (excluding smart hubs).
3. **`3.Get_Vanity_hostnames.py`**: Performs hostname mapping by inspecting captured DNS query/response pairs to identify the "Vanity Hostname" (e.g., `api.brand.com`) requested by each device.
4. **`4.GET_Canonica_hostnames.py`**: Conducts Reverse DNS (rDNS) lookups on destination IP addresses to identify "Canonical Hostnames" (e.g., `*.amazonaws.com`), exposing the underlying infrastructure provider.

## Output
The processed dataset contains filtered flows with a dual-layer semantic mapping (Vanity and Canonical hostnames), serving as the foundation for both implicit and latent relationship detection.
