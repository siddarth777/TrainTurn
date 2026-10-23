#!/usr/bin/env bash
# Download the public source data into data/raw/ (not redistributed with this repository).
# RSTGCN: research use, cite Chowdhury et al., IEEE T-ITS 2026. DataMeet: CC0.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/raw/rstgcn data/raw/datameet
get() { curl -sS -L --retry 8 --retry-all-errors --retry-delay 3 -o "$1" "$2"; }
get data/raw/rstgcn/Indian-Railway-Network-and-Delays.zip \
    https://raw.githubusercontent.com/KoyenaChowdhury/RSTGCN/HEAD/Indian-Railway-Network-and-Delays.zip
get data/raw/datameet/trains.json https://raw.githubusercontent.com/datameet/railways/master/trains.json
get data/raw/datameet/stations.json https://raw.githubusercontent.com/datameet/railways/master/stations.json
(cd data/raw/rstgcn && unzip -o -q Indian-Railway-Network-and-Delays.zip \
    -x "Indian-Railway-Network-and-Delays/train_delays_Sep2024.json")
sha256sum -c - <<'SUMS'
f8a6fe9e70972acb8ad98bbe5b7eb594c5b6b3475e21efd6d3b9e19049bdaf91  data/raw/rstgcn/Indian-Railway-Network-and-Delays.zip
e434d9c56016ccdf2291ccd446586ef87a7fab101084efa5f68871ab27800167  data/raw/datameet/trains.json
9bd5e1da3a859e5359a95f40b6009aa468efb177da4faac57ffdcd7daeb4df18  data/raw/datameet/stations.json
SUMS
