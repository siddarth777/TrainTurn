"""Seven-station, double-track rail network built around the original junction.

Traffic runs in both directions. Each line (A, B) on each side of the junction has
an eastbound and a westbound track with its own signalled blocks; each minor
station has a main track and loops for each direction. Junction platforms are
bidirectional: eastbound trains enter at the west throat (node ``in``) and leave at
the east throat (``out``), westbound trains the reverse.

Junction corridors (the labels the trained agent observes):

=========  =================================  ================================
corridor   as an arrival                      as a departure
=========  =================================  ================================
``A``/``B``  eastbound from west line A/B     eastbound to east line A/B
``C``/``D``  westbound from east line A/B     westbound to west line A/B
=========  =================================  ================================

Eastbound junction edges keep the names, lengths and node positions of
``railway_rl.sumo_network`` (``approachX``, ``entry{p}``, ``platform{p}``,
``exit{p}``, ``downstreamX``). Westbound trains run on SUMO bidirectional twins of
the same physical tracks (``-exit{p}``, ``-platform{p}``, ``-entry{p}``), so SUMO's
rail signals also protect against opposing movements. Every junction connection
is pinned to the internal-lane speed of the trained single-station network
(westbound connections take the speed of their eastbound mirror), so the extra
edges at the throats do not change junction running times.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from railway_rl.station import Station
from railway_rl.sumo_network import binary

RAIL = "rail rail_electric rail_fast"
GEOMETRY_VERSION = 4  # bump when drawn coordinates change (lengths/topology are in the digest)
J_APPROACH, J_TURNOUT, J_DOWNSTREAM = 800, 250, 800
EASTBOUND, WESTBOUND = ("A", "B"), ("C", "D")
CORRIDOR_LINE = {"A": ("A", "e"), "B": ("B", "e"), "C": ("A", "w"), "D": ("B", "w")}
LINE_CORRIDOR = {(line, d): c for c, (line, d) in CORRIDOR_LINE.items()}


@dataclass(frozen=True)
class MinorStation:
    id: str
    name: str
    tracks: int  # per direction; track 1 is that direction's main line

    def __post_init__(self) -> None:
        if not 2 <= self.tracks <= 4:
            raise ValueError("Minor stations need 2-4 tracks per direction (track 1 is the main line)")


@dataclass(frozen=True)
class Branch:
    line: str
    side: str  # "west" or "east" of the junction
    stations: tuple[str, ...]   # west -> east
    links_m: tuple[int, ...]    # west -> east; one more than stations

    def __post_init__(self) -> None:
        if self.side not in ("west", "east") or self.line not in ("A", "B"):
            raise ValueError("Branches are west/east of the junction on line A or B")
        if len(self.links_m) != len(self.stations) + 1 or min(self.links_m) < 1000:
            raise ValueError("Each branch needs one link (>= 1 km) more than it has stations")


@dataclass(frozen=True)
class NetworkLayout:
    stations: tuple[MinorStation, ...]
    branches: tuple[Branch, ...]
    block_length: int
    station_entry_length: int
    station_track_length: int
    station_exit_length: int

    def __post_init__(self) -> None:
        names = [s.id for s in self.stations]
        used = [s for b in self.branches for s in b.stations]
        if len(set(names)) != len(names) or sorted(used) != sorted(names):
            raise ValueError("Every minor station must appear on exactly one branch")
        for side in ("west", "east"):
            if sorted(b.line for b in self.branches if b.side == side) != ["A", "B"]:
                raise ValueError(f"The {side} side needs exactly one A and one B branch")
        if self.station_track_length < 640:
            raise ValueError("Minor-station tracks must hold a 24-coach (600 m) train")

    # ----- naming ---------------------------------------------------------------------
    def station(self, sid: str) -> MinorStation:
        return next(s for s in self.stations if s.id == sid)

    def branch(self, side: str, line: str) -> Branch:
        return next(b for b in self.branches if b.side == side and b.line == line)

    def blocks(self, side: str, line: str, direction: str, link: int) -> list[str]:
        """Block edges of one link in travel order for that direction."""
        length = self.branch(side, line).links_m[link]
        count = max(1, round(length / self.block_length))
        names = [f"ln{side[0].upper()}{line}{direction}{link}_{i}" for i in range(count)]
        return names if direction == "e" else names[::-1]

    @staticmethod
    def track_edges(sid: str, direction: str, track: int) -> list[str]:
        return [f"{sid}_{direction}in{track}", f"{sid}_{direction}pf{track}", f"{sid}_{direction}out{track}"]

    def chain(self, side: str, line: str, direction: str) -> list[tuple]:
        """Travel-order items: ("link", index) and ("station", id)."""
        b = self.branch(side, line)
        items = []
        for i, sid in enumerate(b.stations):
            items += [("link", i), ("station", sid)]
        items.append(("link", len(b.stations)))
        return items if direction == "e" else items[::-1]

    def path(self, side: str, line: str, direction: str, tracks: dict[str, int]) -> list[str]:
        edges = []
        for kind, value in self.chain(side, line, direction):
            edges += (self.blocks(side, line, direction, value) if kind == "link"
                      else self.track_edges(value, direction, tracks[value]))
        return edges

    @staticmethod
    def approach_side(corridor: str) -> str:
        return "west" if corridor in EASTBOUND else "east"

    @staticmethod
    def departure_side(corridor: str) -> str:
        return "east" if corridor in EASTBOUND else "west"

    def _side_items(self, corridor: str, approach: bool):
        line, d = CORRIDOR_LINE[corridor]
        side = self.approach_side(corridor) if approach else self.departure_side(corridor)
        return side, line, d, self.chain(side, line, d)

    def approach_path(self, corridor: str, tracks: dict[str, int]) -> list[str]:
        side, line, d, _ = self._side_items(corridor, True)
        return self.path(side, line, d, tracks)

    def departure_path(self, corridor: str, tracks: dict[str, int]) -> list[str]:
        side, line, d, _ = self._side_items(corridor, False)
        return self.path(side, line, d, tracks)

    def stations_on(self, corridor: str, approach: bool) -> list[str]:
        return [v for k, v in self._side_items(corridor, approach)[3] if k == "station"]

    def links_on(self, corridor: str, approach: bool) -> list[float]:
        side, line, _, items = self._side_items(corridor, approach)
        return [self.branch(side, line).links_m[v] for k, v in items if k == "link"]

    def feed_edges(self, corridor: str) -> list[str]:
        """Blocks between the last station before the junction and the junction gate."""
        side, line, d, items = self._side_items(corridor, True)
        return self.blocks(side, line, d, items[-1][1])

    def gate_edge(self, corridor: str) -> str:
        return self.feed_edges(corridor)[-1]

    def first_departure_edge(self, corridor: str) -> str:
        side, line, d, items = self._side_items(corridor, False)
        return self.blocks(side, line, d, items[0][1])[0]

    def decision_edges(self) -> dict[str, tuple[str, str]]:
        """Last block before each (station, direction); the track is chosen while a train is on it."""
        result = {}
        for b in self.branches:
            for d in ("e", "w"):
                items = self.chain(b.side, b.line, d)
                for i, (kind, value) in enumerate(items):
                    if kind == "station":
                        result[self.blocks(b.side, b.line, d, items[i - 1][1])[-1]] = (value, d)
        return result

    def line_edges(self, corridor: str) -> set[str]:
        """Every edge a train approaching the junction on this corridor can occupy."""
        side, line, d, items = self._side_items(corridor, True)
        edges = set()
        for kind, value in items:
            if kind == "link":
                edges |= set(self.blocks(side, line, d, value))
            else:
                for k in range(1, self.station(value).tracks + 1):
                    edges |= set(self.track_edges(value, d, k))
        return edges

    @staticmethod
    def junction_path(arrival: str, platform: int, departure: str) -> list[str]:
        if arrival in EASTBOUND:
            middle = [f"entry{platform}", f"platform{platform}", f"exit{platform}"]
        else:
            middle = [f"-exit{platform}", f"-platform{platform}", f"-entry{platform}"]
        return [f"approach{arrival}", *middle, f"downstream{departure}"]

    @staticmethod
    def platform_edges(platform: int) -> tuple[str, str]:
        return f"platform{platform}", f"-platform{platform}"

    def digest(self, station: Station) -> str:
        payload = json.dumps({"layout": asdict(self), "junction": asdict(station), "bidirectional": True},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ----- SUMO network ----------------------------------------------------------------
    def build(self, station: Station, cache: Path | None = None) -> Path:
        cache = cache or Path(tempfile.gettempdir()) / f"railway-sumo-multistation-g{GEOMETRY_VERSION}"
        root = cache / self.digest(station)
        output = root / "network.net.xml"
        if output.exists():
            return output
        root.mkdir(parents=True, exist_ok=True)
        nodes, edges, connections = ET.Element("nodes"), ET.Element("edges"), ET.Element("connections")
        placed: set[str] = set()
        bidi: list[str] = []

        def node(name, x, y, kind="rail_signal"):
            if name not in placed:
                placed.add(name)
                ET.SubElement(nodes, "node", id=name, x=f"{x:.1f}", y=f"{y:.1f}", type=kind)

        def edge(name, source, target, length, **extra):
            ET.SubElement(edges, "edge", id=name, **{"from": source, "to": target}, numLanes="1",
                          speed="20", length=f"{length:.1f}", allow=RAIL, **extra)

        def connect(a, b):
            ET.SubElement(connections, "connection", **{"from": a, "to": b})

        # Junction. The eastbound part is identical to the trained single-station network.
        n = len(station.platforms)
        node("in", 0, 0)
        node("out", 1300, 0)
        for line, sign in (("A", 1), ("B", -1)):
            east_c, west_c = LINE_CORRIDOR[(line, "e")], LINE_CORRIDOR[(line, "w")]
            node(f"gate{east_c}", -250, 50 * sign)
            node(f"gateOut{east_c}", 1550, 50 * sign)
            # Westbound gates sit far out so their lines meet the throats at shallow angles.
            node(f"gateE{west_c}", 2000, 120 * sign)
            node(f"gateW{west_c}", -700, 120 * sign)
            edge(f"approach{east_c}", f"gate{east_c}", "in", J_APPROACH)
            edge(f"downstream{east_c}", "out", f"gateOut{east_c}", J_DOWNSTREAM)
            edge(f"approach{west_c}", f"gateE{west_c}", "out", J_APPROACH)
            edge(f"downstream{west_c}", "in", f"gateW{west_c}", J_DOWNSTREAM)
        for p in station.platforms:
            y = (p.id - (n + 1) / 2) * 60
            node(f"left{p.id}", 250, y)
            node(f"right{p.id}", 1050, y)
            length = p.length * 25 + 40
            # Shared physical tracks: SUMO creates the reverse twins ("-entry1", ...).
            edge(f"entry{p.id}", "in", f"left{p.id}", J_TURNOUT, spreadType="center")
            edge(f"platform{p.id}", f"left{p.id}", f"right{p.id}", length, spreadType="center")
            edge(f"exit{p.id}", f"right{p.id}", "out", J_TURNOUT, spreadType="center")
            bidi += [f"entry{p.id}", f"platform{p.id}", f"exit{p.id}"]
            for c in p.approaches:
                if c in EASTBOUND:
                    connect(f"approach{c}", f"entry{p.id}")
            connect(f"entry{p.id}", f"platform{p.id}")
            connect(f"platform{p.id}", f"exit{p.id}")
            for c in p.departures:
                if c in EASTBOUND:
                    connect(f"exit{p.id}", f"downstream{c}")

        # Lines. Positions run west -> east for both directions; edges point in travel direction.
        scale, width, fan = 0.12, 440.0, 110.0
        for b in self.branches:
            sign = 1 if b.line == "A" else -1
            visual = sum(b.links_m) * scale + width * len(b.stations)
            for d in ("e", "w"):
                corridor = LINE_CORRIDOR[(b.line, d)]
                y = (50.0 if d == "e" else 120.0) * sign
                step = 16.0 * sign
                if b.side == "west":
                    x = (-250.0 if d == "e" else -700.0) - visual
                    west_end = f"originW{b.line}" if d == "e" else f"sinkW{b.line}"
                    east_end = f"gate{corridor}" if d == "e" else f"gateW{corridor}"
                    node(west_end, x, y, "priority")
                else:
                    x = 1550.0 if d == "e" else 2000.0
                    west_end = f"gateOut{corridor}" if d == "e" else f"gateE{corridor}"
                    east_end = f"sinkE{b.line}" if d == "e" else f"originE{b.line}"
                for link in range(len(b.stations) + 1):
                    count = max(1, round(b.links_m[link] / self.block_length))
                    seg = b.links_m[link] / count
                    prev = west_end if link == 0 else f"{b.stations[link - 1]}_{d}b"
                    for i in range(count):
                        name = f"ln{b.side[0].upper()}{b.line}{d}{link}_{i}"
                        x += seg * scale
                        if i < count - 1:
                            target = f"sig_{b.side[0]}{b.line}{d}{link}_{i}"
                        elif link < len(b.stations):
                            target = f"{b.stations[link]}_{d}a"
                        else:
                            target = east_end
                        node(target, x, y, "priority" if target.startswith(("sink", "origin")) else "rail_signal")
                        edge(name, prev, target, seg) if d == "e" else edge(name, target, prev, seg)
                        prev = target
                    if link < len(b.stations):
                        sid = b.stations[link]
                        node(f"{sid}_{d}b", x + width, y)
                        for k in range(1, self.station(sid).tracks + 1):
                            ty = y + (k - 1) * step
                            node(f"{sid}_{d}l{k}", x + fan, ty)
                            node(f"{sid}_{d}r{k}", x + width - fan, ty)
                            a, left, right, z = f"{sid}_{d}a", f"{sid}_{d}l{k}", f"{sid}_{d}r{k}", f"{sid}_{d}b"
                            ins, pf, out = self.track_edges(sid, d, k)
                            if d == "e":
                                edge(ins, a, left, self.station_entry_length)
                                edge(pf, left, right, self.station_track_length)
                                edge(out, right, z, self.station_exit_length)
                            else:
                                edge(ins, z, right, self.station_entry_length)
                                edge(pf, right, left, self.station_track_length)
                                edge(out, left, a, self.station_exit_length)
                        x += width
                items = self.chain(b.side, b.line, d)
                for i, (kind, value) in enumerate(items):
                    if kind == "link":
                        blocks = self.blocks(b.side, b.line, d, value)
                        for first, second in zip(blocks, blocks[1:]):
                            connect(first, second)
                    else:
                        before = self.blocks(b.side, b.line, d, items[i - 1][1])[-1]
                        after = self.blocks(b.side, b.line, d, items[i + 1][1])[0]
                        for k in range(1, self.station(value).tracks + 1):
                            ins, pf, out = self.track_edges(value, d, k)
                            connect(before, ins)
                            connect(ins, pf)
                            connect(pf, out)
                            connect(out, after)
                if (b.side == "west") == (d == "e"):   # this track approaches the junction
                    connect(self.gate_edge(corridor), f"approach{corridor}")
                else:                                  # this track leaves the junction
                    connect(f"downstream{corridor}", self.first_departure_edge(corridor))

        for name, tree in (("nodes", nodes), ("edges", edges), ("connections", connections)):
            ET.ElementTree(tree).write(root / f"{name}.xml", encoding="utf-8", xml_declaration=True)
        (root / "bidi.txt").write_text("\n".join(bidi) + "\n")
        netconvert = binary("netconvert")
        # Pass 1: create the bidirectional twins and export plain XML.
        self._run([netconvert, "--node-files", str(root / "nodes.xml"), "--edge-files", str(root / "edges.xml"),
                   "--connection-files", str(root / "connections.xml"),
                   "--railway.topology.all-bidi.input-file", str(root / "bidi.txt"),
                   "--plain-output-prefix", str(root / "plain"), "--no-turnarounds", "true",
                   "--no-warnings", "true", "--output-file", str(root / "pass1.net.xml")])
        # Pass 2: add westbound junction connections and pin all junction speeds.
        speeds = trained_junction_speeds(station)
        tree = ET.parse(root / "plain.con.xml")
        con = tree.getroot()
        for p in station.platforms:
            for c in station.platforms[0].approaches:
                if c in WESTBOUND:
                    ET.SubElement(con, "connection", **{"from": f"approach{c}", "to": f"-exit{p.id}"})
            ET.SubElement(con, "connection", **{"from": f"-exit{p.id}", "to": f"-platform{p.id}"})
            ET.SubElement(con, "connection", **{"from": f"-platform{p.id}", "to": f"-entry{p.id}"})
            for c in p.departures:
                if c in WESTBOUND:
                    ET.SubElement(con, "connection", **{"from": f"-entry{p.id}", "to": f"downstream{c}"})
        for element in con.iter("connection"):
            key = (element.get("from"), element.get("to"))
            if key in speeds:
                element.set("fromLane", "0")
                element.set("toLane", "0")
                element.set("speed", f"{speeds[key]:.2f}")
        tree.write(root / "pinned.con.xml", encoding="utf-8", xml_declaration=True)
        command = [netconvert, "--node-files", str(root / "plain.nod.xml"), "--edge-files", str(root / "plain.edg.xml"),
                   "--connection-files", str(root / "pinned.con.xml"), "--no-turnarounds", "true",
                   "--no-warnings", "true", "--output-file", str(output)]
        if (root / "plain.tll.xml").exists():
            command[1:1] = ["--tllogic-files", str(root / "plain.tll.xml")]
        self._run(command)
        return output

    @staticmethod
    def _run(command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"netconvert failed: {result.stderr}")


def trained_junction_speeds(station: Station) -> dict[tuple[str, str], float]:
    """Internal-lane speeds of the trained single-station network, mirrored for westbound."""
    from dataclasses import replace
    from railway_rl.sumo_network import network_file
    base = replace(station, platforms=tuple(replace(p, approaches=EASTBOUND, departures=EASTBOUND)
                                            for p in station.platforms))
    net = ET.parse(network_file(base, Path(tempfile.gettempdir()) / "railway-sumo-networks-v1")).getroot()
    lanes = {lane.get("id"): float(lane.get("speed")) for e in net.iter("edge") for lane in e.iter("lane")}
    trained = {(c.get("from"), c.get("to")): lanes[c.get("via")] for c in net.iter("connection") if c.get("via")}
    mirror = {}
    for p in station.platforms:
        i = p.id
        for east, west in zip(EASTBOUND, WESTBOUND):
            if (f"approach{east}", f"entry{i}") in trained:
                mirror[(f"approach{west}", f"-exit{i}")] = trained[(f"approach{east}", f"entry{i}")]
            if (f"exit{i}", f"downstream{east}") in trained:
                mirror[(f"-entry{i}", f"downstream{west}")] = trained[(f"exit{i}", f"downstream{east}")]
        mirror[(f"-exit{i}", f"-platform{i}")] = trained[(f"platform{i}", f"exit{i}")]
        mirror[(f"-platform{i}", f"-entry{i}")] = trained[(f"entry{i}", f"platform{i}")]
    return {**trained, **mirror}


def layout_from_config(network: dict) -> NetworkLayout:
    stations = tuple(MinorStation(sid, s["name"], int(s["tracks"])) for sid, s in network["stations"].items())
    branches = tuple(Branch(line, side, tuple(spec["stations"]), tuple(int(x) for x in spec["links_m"]))
                     for side in ("west", "east") for line, spec in network[side].items())
    return NetworkLayout(stations, branches, int(network["block_length"]), int(network["station_entry_length"]),
                         int(network["station_track_length"]), int(network["station_exit_length"]))
