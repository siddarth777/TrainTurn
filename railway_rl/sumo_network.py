"""Generate an original two-corridor junction; SUMO handles physical rail movement."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from .station import Station


def binary(name: str) -> str:
    local = Path(sys.executable).parent / name
    found = str(local) if local.is_file() else shutil.which(name)
    if found is None:
        raise RuntimeError(f"{name} not installed: uv pip install eclipse-sumo")
    return found


def network_file(station: Station, cache: Path) -> Path:
    digest = hashlib.sha256(json.dumps(asdict(station), sort_keys=True).encode()).hexdigest()[:16]
    root = cache / digest
    output = root / "station.net.xml"
    if output.exists():
        return output
    root.mkdir(parents=True, exist_ok=True)
    nodes, edges, connections = ET.Element("nodes"), ET.Element("edges"), ET.Element("connections")
    for name, x, y, kind in [("originA", -250, 50, "priority"), ("originB", -250, -50, "priority"),
                             ("in", 0, 0, "rail_signal"), ("out", 1300, 0, "rail_signal"),
                             ("sinkA", 1550, 50, "priority"), ("sinkB", 1550, -50, "priority")]:
        ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y), type=kind)
    def edge(name, source, target, length, allow="rail rail_electric rail_fast"):
        ET.SubElement(edges, "edge", id=name, **{"from": source, "to": target},
                      numLanes="1", speed="20", length=str(length), allow=allow)
    for corridor in ("A", "B"):
        # The boundary reservoir is explicit in the adapter; this long edge permits insertion of a full train.
        edge(f"approach{corridor}", f"origin{corridor}", "in", 800)
        edge(f"downstream{corridor}", "out", f"sink{corridor}", 800)
    for p in station.platforms:
        y = (p.id - (len(station.platforms) + 1) / 2) * 60
        for prefix, x in (("left", 250), ("right", 1050)):
            ET.SubElement(nodes, "node", id=f"{prefix}{p.id}", x=str(x), y=str(y), type="rail_signal")
        permission = "rail rail_electric rail_fast" if p.electrified else "rail"
        edge(f"entry{p.id}", "in", f"left{p.id}", 250, permission)
        edge(f"platform{p.id}", f"left{p.id}", f"right{p.id}", p.length * 25 + 40, permission)
        edge(f"exit{p.id}", f"right{p.id}", "out", 250, permission)
        for corridor in p.approaches:
            ET.SubElement(connections, "connection", **{"from": f"approach{corridor}", "to": f"entry{p.id}"})
        ET.SubElement(connections, "connection", **{"from": f"entry{p.id}", "to": f"platform{p.id}"})
        ET.SubElement(connections, "connection", **{"from": f"platform{p.id}", "to": f"exit{p.id}"})
        for corridor in p.departures:
            ET.SubElement(connections, "connection", **{"from": f"exit{p.id}", "to": f"downstream{corridor}"})
    for name, tree in (("nodes", nodes), ("edges", edges), ("connections", connections)):
        ET.ElementTree(tree).write(root / f"{name}.xml", encoding="utf-8", xml_declaration=True)
    command = [binary("netconvert"), "--node-files", str(root / "nodes.xml"), "--edge-files", str(root / "edges.xml"),
               "--connection-files", str(root / "connections.xml"), "--output-file", str(output),
               "--no-turnarounds", "true", "--no-warnings", "true"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"netconvert failed: {result.stderr}")
    return output
