"""
Flood knowledge graph + Graph RAG.

Nodes:  building, shelter, road, risk level, facility type, guidance topic
Edges:  building -NEAREST_SHELTER-> shelter      (distance_km)
        building -ACCESS_VIA-> road               (nearest road, its closure time)
        building -HAS_RISK-> risk level
        building -IS_A-> facility type
        facility type / risk level -GUIDED_BY-> guidance topic (knowledge-base doc)

Graph RAG walks two hops out from a building to collect structured facts
(e.g. how loaded its shelter is, whether its access road closes before the
water arrives) and the guidance topics to retrieve, then hands those to the
text retriever. This is what lets an alert say "your road closes in 1 hour,
leave by boat" -- a fact that lives in the graph, not in any single document.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

AVG_HOUSEHOLD_SIZE = 4.0  # assumed people per household (rural Bangladesh is roughly 4); tune with local data

TYPE_TOPICS = {
    "household": ["01_household_evacuation", "04_livestock_and_assets"],
    "clinic": ["03_medical_facility_evacuation"],
    "school": ["02_shelter_operations"],
    "market_stall": ["04_livestock_and_assets"],
}
RISK_TOPICS = {
    "CRITICAL": ["01_household_evacuation", "05_roads_and_boats"],
    "HIGH": ["01_household_evacuation", "06_warning_levels"],
    "WATCH": ["06_warning_levels"],
    "SAFE": ["06_warning_levels"],
}


@dataclass
class KnowledgeGraph:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: dict[str, list[tuple[str, str, dict]]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node_id: str, **attrs):
        self.nodes.setdefault(node_id, {}).update(attrs)

    def add_edge(self, src: str, rel: str, dst: str, **attrs):
        self.edges[src].append((rel, dst, attrs))

    def neighbors(self, node_id: str, rel: str | None = None):
        return [(r, d, a) for r, d, a in self.edges.get(node_id, []) if rel is None or r == rel]

    def incoming(self, node_id: str, rel: str):
        return [s for s, outs in self.edges.items() for r, d, _ in outs if d == node_id and r == rel]

    def stats(self) -> dict:
        return {"nodes": len(self.nodes), "edges": sum(len(v) for v in self.edges.values())}


def build_graph(pipeline_result: dict) -> KnowledgeGraph:
    g = KnowledgeGraph()
    scenario = pipeline_result["scenario"]
    impacts = pipeline_result["building_impacts"]
    roads = {r.id: r for r in pipeline_result["road_impacts"]}

    for _, s in scenario.shelters.iterrows():
        g.add_node(f"shelter:{s['id']}", kind="shelter", name=s["name"], capacity=int(s["capacity"]))
    for road in scenario.roads:
        ri = roads.get(road["id"])
        g.add_node(f"road:{road['id']}", kind="road", name=road["name"],
                   status=ri.status if ri else None,
                   hours_to_inaccessible=ri.hours_to_inaccessible if ri else None)
    for level, topics in RISK_TOPICS.items():
        g.add_node(f"risk:{level}", kind="risk_level")
        for t in topics:
            g.add_edge(f"risk:{level}", "GUIDED_BY", f"topic:{t}")
    for btype, topics in TYPE_TOPICS.items():
        g.add_node(f"type:{btype}", kind="facility_type")
        for t in topics:
            g.add_edge(f"type:{btype}", "GUIDED_BY", f"topic:{t}")
    for t in {t for ts in list(TYPE_TOPICS.values()) + list(RISK_TOPICS.values()) for t in ts}:
        g.add_node(f"topic:{t}", kind="guidance_topic")

    road_pts = {r["id"]: np.asarray(r["points"], dtype=float) for r in scenario.roads}
    for b in impacts:
        bid = f"building:{b.id}"
        g.add_node(bid, kind="building", type=b.type, hours_to_flood=b.hours_to_flood,
                   risk_level=b.risk_level, priority=b.priority)
        g.add_edge(bid, "IS_A", f"type:{b.type}")
        g.add_edge(bid, "HAS_RISK", f"risk:{b.risk_level}")
        if b.nearest_shelter_id:
            g.add_edge(bid, "NEAREST_SHELTER", f"shelter:{b.nearest_shelter_id}",
                       distance_km=b.distance_to_shelter_km)
        if road_pts:
            here = np.array([b.row, b.col], dtype=float)
            rid, dist = min(((rid, float(np.min(np.linalg.norm(pts - here, axis=1))))
                             for rid, pts in road_pts.items()), key=lambda x: x[1])
            g.add_edge(bid, "ACCESS_VIA", f"road:{rid}", grid_distance=round(dist, 1))
    return g


def graph_context(g: KnowledgeGraph, building_id: str) -> dict:
    """Two-hop Graph RAG traversal for one building -> facts + topics."""
    bid = building_id if building_id.startswith("building:") else f"building:{building_id}"
    if bid not in g.nodes:
        return {"facts": [], "topics": []}
    b = g.nodes[bid]
    facts, topics = [], []

    for _, sid, attrs in g.neighbors(bid, "NEAREST_SHELTER"):
        s = g.nodes[sid]
        assigned = [x for x in g.incoming(sid, "NEAREST_SHELTER")
                    if g.nodes[x].get("risk_level") in ("CRITICAL", "HIGH")]
        people = round(len(assigned) * AVG_HOUSEHOLD_SIZE)
        load = people / s["capacity"] if s.get("capacity") else None
        facts.append(f"Nearest shelter is {s['name']}, {attrs.get('distance_km')} km away, "
                     f"capacity {s['capacity']} people.")
        if load is not None:
            facts.append(f"About {people} people from {len(assigned)} at-risk buildings are assigned to "
                         f"{s['name']} ({round(100 * load)}% of capacity).")
            if load >= 0.8:
                others = [n for n, a in g.nodes.items() if a.get("kind") == "shelter" and n != sid]
                if others:
                    facts.append(f"{s['name']} is near capacity; the alternative shelter is "
                                 f"{g.nodes[others[0]]['name']}.")

    for _, rid, _ in g.neighbors(bid, "ACCESS_VIA"):
        r = g.nodes[rid]
        hb, hr = b.get("hours_to_flood"), r.get("hours_to_inaccessible")
        if hr is not None:
            facts.append(f"Access road {r['name']} becomes impassable in about {hr} hours.")
            if hb is not None and hr < hb:
                facts.append("The road closes before the building floods, so leave early or arrange a boat.")
        else:
            facts.append(f"Access road {r['name']} is expected to stay passable.")

    for rel in ("IS_A", "HAS_RISK"):
        for _, node, _ in g.neighbors(bid, rel):
            for _, tid, _ in g.neighbors(node, "GUIDED_BY"):
                t = tid.split(":", 1)[1]
                if t not in topics:
                    topics.append(t)
    return {"facts": facts, "topics": topics}
