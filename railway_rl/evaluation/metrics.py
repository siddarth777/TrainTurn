from collections import defaultdict
from ..simulator import Simulator


def episode_metrics(sim: Simulator, horizon: int) -> tuple[dict, list[dict]]:
    """Return measured aggregates and explicit per-train completion/censoring records."""
    sim.assert_safety()
    rows = []
    for train in sim.scenario.trains:
        state = sim.states[train.id]
        delay = state.waiting + state.departure_waiting
        departure = state.station_departure if state.station_departure is not None else state.departure
        rows.append({"train": train.id, "category": train.category, "priority": train.priority,
                     "completed": state.departure is not None, "incoming_delay": sim.noise[train.id].incoming_delay,
                     "station_delay": delay, "weighted_delay": delay * train.priority,
                     "waiting_minutes": state.waiting, "upstream_minutes": state.upstream_waiting,
                     "departure_waiting": state.departure_waiting,
                     "departure_lateness": max(0, departure - train.scheduled_departure) if state.departure is not None else None,
                     "scheduled_arrival": train.scheduled_arrival, "signal_arrival": state.signal_arrival,
                     "entry": state.entry, "departure": departure, "network_exit": state.departure, "platform": state.platform})
    completed = [r for r in rows if r["completed"]]
    classes = defaultdict(list)
    for row in rows:
        classes[row["category"]].append(row["station_delay"])
    start = min(t.scheduled_arrival for t in sim.scenario.trains)
    window = max(1, horizon - start)
    occupied = {p.id: 0 for p in sim.station.platforms}
    for event in getattr(sim, "platform_occupancy", sim.occupancy):
        end = event["end"] if event["end"] is not None else sim.time
        occupied[event["platform"]] += max(0, min(end, horizon) - max(start, event["start"]))
    summary = {
        "weighted_station_delay": sum(r["weighted_delay"] for r in rows),
        "mean_station_delay": sum(r["station_delay"] for r in rows) / len(rows),
        "mean_departure_lateness_completed": sum(r["departure_lateness"] for r in completed) / len(completed) if completed else None,
        "max_departure_lateness_completed": max((r["departure_lateness"] for r in completed), default=None),
        "max_station_delay": max(r["station_delay"] for r in rows),
        "waiting_minutes": sum(r["waiting_minutes"] for r in rows),
        "upstream_minutes": sum(r["upstream_minutes"] for r in rows),
        "waiting_events": sum(r["waiting_minutes"] > 0 for r in rows),
        "congestion_events": sim.congestion_events, "platform_changes": sim.platform_changes,
        "processed_fraction": len(completed) / len(rows), "unfinished": len(rows) - len(completed),
        "platform_utilization": sum(occupied.values()) / (len(occupied) * window),
        "safety_violations": 0, "simulated_minutes": sim.time,
        "holds": sim.holds, "exceptional_waits": sim.forced_waits,
    }
    summary.update({f"delay_{name}": sum(values) / len(values) for name, values in classes.items()})
    summary.update({f"utilization_p{p}": minutes / window for p, minutes in occupied.items()})
    return summary, rows
