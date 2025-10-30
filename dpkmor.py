#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import math
import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Any, Set


# -------------------- Graph --------------------
class Graph:
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: Dict[str, List[Tuple[str, float]]] = {}

    def add_edge(self, u: str, v: str, w: float = 1.0) -> None:
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        self.adj[u].append((v, float(w)))
        if not self.directed:
            self.adj[v].append((u, float(w)))

    def neighbors(self, u: str) -> Iterable[Tuple[str, float]]:
        return self.adj.get(u, [])

    def nodes(self) -> Iterable[str]:
        return self.adj.keys()

    def __len__(self):
        return len(self.adj)


# -------------------- IO: edges --------------------
def read_edge_list(path: str, delimiter: Optional[str] = None, directed: bool = False,
                   comment: str = "#", ignore_first_col: bool = True) -> Graph:
    graph = Graph(directed=directed)
    sep = None
    if delimiter:
        if delimiter == "space":
            sep = None
        elif delimiter == "tab":
            sep = "\t"
        elif delimiter == "comma":
            sep = ","
        else:
            sep = delimiter

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            parts = line.split(sep) if sep is not None else line.split()
            if ignore_first_col:
                if len(parts) < 4:
                    continue
                u = parts[1]; v = parts[2]; w_str = parts[3]
            else:
                if len(parts) < 3:
                    continue
                u = parts[0]; v = parts[1]; w_str = parts[2]
            try:
                w = float(w_str)
            except (ValueError, TypeError):
                w = 1.0
            graph.add_edge(str(u), str(v), w)
    return graph


# -------------------- Requests --------------------
@dataclass
class Request:
    name: str
    pickup: str
    dropoff: str
    demand: int = 1
    wait_limit: float = math.inf


def parse_requests_file(path: str) -> List[Request]:
    reqs: List[Request] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in (line.replace(",", " ").split())]
            if len(parts) < 2:
                continue
            if len(parts) >= 5:
                name = parts[0]; pickup = parts[1]; dropoff = parts[2]
                try:
                    demand = int(parts[3])
                except ValueError:
                    demand = 1
                try:
                    wait_limit = float(parts[4])
                except ValueError:
                    wait_limit = math.inf
            elif len(parts) == 4:
                name = f"r{ln}"; pickup = parts[0]; dropoff = parts[1]
                try:
                    demand = int(parts[2])
                except ValueError:
                    demand = 1
                try:
                    wait_limit = float(parts[3])
                except ValueError:
                    wait_limit = math.inf
            else:  # len == 2
                name = f"r{ln}"; pickup = parts[0]; dropoff = parts[1]; demand = 1; wait_limit = math.inf
            reqs.append(Request(name=name, pickup=str(pickup), dropoff=str(dropoff),
                                demand=demand, wait_limit=wait_limit))
    return reqs


# -------------------- Shortest Path --------------------
def dijkstra_shortest_path(graph: Graph, source: str, target: str) -> Tuple[List[str], float]:
    if source not in graph.adj or target not in graph.adj:
        return [], math.inf
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}
    pq: List[Tuple[float, str]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == target:
            break
        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("Dijkstra 不支持负权")
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return [], math.inf
    path: List[str] = []
    cur: Optional[str] = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, dist[target]


# -------------------- 路径工具函数 --------------------
def path_cost(graph: Graph, path: List[str]) -> float:
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        u = path[i]; v = path[i + 1]
        found = False
        for nb, cost in graph.neighbors(u):
            if nb == v:
                total += cost
                found = True
                break
        if not found:
            return math.inf
    return total


def is_reasonable_path(path: List[str], max_len: int = 60) -> bool:
    if not path:
        return False
    if len(path) > max_len:
        return False
    for i in range(1, len(path) - 1):
        if path[i - 1] == path[i + 1]:
            return False
    return True


# -------------------- 终端集最短路 --------------------
def pairwise_shortest(graph: Graph, terms: List[str]) -> Tuple[List[List[float]], List[List[List[str]]]]:
    n = len(terms)
    dist = [[math.inf] * n for _ in range(n)]
    paths: List[List[List[str]]] = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0.0
                paths[i][j] = [terms[i]]
            else:
                p, c = dijkstra_shortest_path(graph, terms[i], terms[j])
                dist[i][j] = c
                paths[i][j] = p
    return dist, paths


def reconstruct_route(seq: List[int], paths_table: List[List[List[str]]], terms: List[str]) -> List[str]:
    if not seq:
        return []
    route: List[str] = [terms[seq[0]]]
    for a, b in zip(seq, seq[1:]):
        seg = paths_table[a][b]
        if not seg:
            return []
        route.extend(seg[1:])
    return route


# -------------------- DP 插入（束搜索） --------------------
def select_candidates(graph: Graph, source: str, target: str, requests: List[Request], limit: int) -> List[Request]:
    # 以 detour = S->P + P->D + D->T - S->T 排序
    _, base_cost = dijkstra_shortest_path(graph, source, target)
    scored: List[Tuple[float, Request]] = []
    for r in requests:
        _, c_sp = dijkstra_shortest_path(graph, source, r.pickup)
        _, c_pd = dijkstra_shortest_path(graph, r.pickup, r.dropoff)
        _, c_dt = dijkstra_shortest_path(graph, r.dropoff, target)
        det = c_sp + c_pd + c_dt - base_cost
        if math.isinf(det) or math.isinf(c_pd):
            continue
        scored.append((det, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:limit]]


def check_wait_limits_on_seq(seq: List[int], dist: List[List[float]], m: int, reqs: List[Request]) -> bool:
    pos: Dict[int, int] = {}
    for idx, node in enumerate(seq):
        pos[node] = idx
    for i, r in enumerate(reqs):
        pu = 1 + i
        do = 1 + m + i
        if pu in pos and do in pos and pos[do] > pos[pu]:
            cost_on_route = 0.0
            for a, b in zip(seq[pos[pu]:pos[do]], seq[pos[pu]:pos[do] + 1]):
                cost_on_route += dist[a][b]
            shortest = dist[pu][do]
            if math.isinf(shortest):
                return False
            if cost_on_route - shortest > r.wait_limit:
                return False
    return True


def dpkmor_dp(graph: Graph, source: str, target: str, requests: List[Request],
              k: int, capacity: int, dp_limit: int = 10, beam: int = 64, max_steps: int = 60000,
              len_limit: int = 60, debug: bool = False) -> List[Tuple[List[str], float, int, List[str]]]:
    # 候选请求
    cand = select_candidates(graph, source, target, requests, dp_limit)
    if debug:
        print(f"候选筛选: {len(cand)}/{len(requests)}")
    m = len(cand)
    terms: List[str] = [source] + [r.pickup for r in cand] + [r.dropoff for r in cand] + [target]
    dist, paths_table = pairwise_shortest(graph, terms)
    n_terms = len(terms)
    if math.isinf(dist[0][n_terms - 1]):
        # 基础不可达
        base_path, base_cost = dijkstra_shortest_path(graph, source, target)
        return [] if not base_path else [(base_path, base_cost, 0, [])]

    # 状态：(cost, -served, last, maskP, maskD, load, seq)
    # maskP/maskD: m 位
    pq: List[Tuple[float, int, int, int, int, int, List[int]]] = []
    heapq.heappush(pq, (0.0, 0, 0, 0, 0, 0, [0]))
    visited: Dict[Tuple[int, int, int, int], float] = {}

    results: List[Tuple[List[str], float, int, List[str]]] = []
    steps = 0
    pushed = 0

    # 预先构造名称列表
    names = [r.name for r in cand]

    while pq and steps < max_steps:
        cost, nserved_neg, last, maskP, maskD, load, seq = heapq.heappop(pq)
        nserved = -nserved_neg
        steps += 1

        if debug and steps <= 5:
            # 仅前几步打印，避免刷屏
            print(f"[DBG] pop step={steps}, last={last}, load={load}, served={nserved}, seq_len={len(seq)}")

        key = (last, maskP, maskD, load)
        if visited.get(key, math.inf) <= cost:
            continue
        visited[key] = cost

        # 随时可以尝试结束到 T
        if last != n_terms - 1 and not math.isinf(dist[last][n_terms - 1]):
            end_seq = seq + [n_terms - 1]
            if check_wait_limits_on_seq(end_seq, dist, m, cand):
                route = reconstruct_route(end_seq, paths_table, terms)
                if route and is_reasonable_path(route, len_limit):
                    # 统计已完成送达
                    served_names: List[str] = []
                    for i, r in enumerate(cand):
                        if (((maskP >> i) & 1) == 1) and (((maskD >> i) & 1) == 1):
                            served_names.append(r.name)
                    total_cost = cost + dist[last][n_terms - 1]
                    results.append((route, total_cost, len(served_names), served_names))

        # 生成可扩展动作
        actions: List[Tuple[int, int, int]] = []  # (next_idx, new_load, type) type: 0=pick,1=drop
        # 尝试 pickup
        for i, r in enumerate(cand):
            if (((maskP >> i) & 1) == 0):
                idx = 1 + i
                if not math.isinf(dist[last][idx]) and load + r.demand <= capacity:
                    actions.append((idx, load + r.demand, 0))
        # 尝试 dropoff
        for i, r in enumerate(cand):
            picked = (((maskP >> i) & 1) == 1)
            dropped = (((maskD >> i) & 1) == 1)
            if picked and not dropped:
                idx = 1 + m + i
                if not math.isinf(dist[last][idx]):
                    actions.append((idx, load - r.demand if load - r.demand >= 0 else 0, 1))

        # 排序动作：优先靠近 T
        actions.sort(key=lambda a: dist[a[0]][n_terms - 1])
        if debug and steps <= 5:
            print(f"[DBG] actions={len(actions)} (pick/drop top3): {[(a[0], a[2]) for a in actions[:3]]}")
        # 取 beam 个
        for nxt, nload, typ in actions[:beam]:
            edge = dist[last][nxt]
            if math.isinf(edge):
                continue
            new_cost = cost + edge
            new_maskP, new_maskD = maskP, maskD
            if typ == 0:
                i = nxt - 1
                new_maskP |= (1 << i)
            else:
                i = nxt - (1 + m)
                new_maskD |= (1 << i)

            new_seq = seq + [nxt]
            # 立即检查：如果刚完成了某个乘客的D，校验个人绕行
            if typ == 1:
                # 找到该乘客的 P 在 new_seq 中的位置
                try:
                    pi = new_seq.index(1 + i)
                    di = new_seq.index(1 + m + i)
                except ValueError:
                    pi = -1; di = -1
                if pi != -1 and di != -1 and di > pi:
                    pd_cost = 0.0
                    for a, b in zip(new_seq[pi:di], new_seq[pi:di + 1]):
                        pd_cost += dist[a][b]
                    shortest = dist[1 + i][1 + m + i]
                    if math.isinf(shortest) or (pd_cost - shortest) > cand[i].wait_limit:
                        continue

            # 剪枝：重复状态
            state_key = (nxt, new_maskP, new_maskD, nload)
            if visited.get(state_key, math.inf) <= new_cost:
                continue

            served_count = 0
            for j in range(m):
                if ((new_maskP >> j) & 1) and ((new_maskD >> j) & 1):
                    served_count += 1

            # 分数：服务人数优先，其次成本+启发
            h = dist[nxt][n_terms - 1]
            heapq.heappush(pq, (new_cost, -served_count, nxt, new_maskP, new_maskD, nload, new_seq))
            pushed += 1
        if debug and steps <= 5:
            print(f"[DBG] pq_size={len(pq)} after push")
    if debug:
        print(f"扩展步骤: {steps}, 压入队列: {pushed}, 结果数: {len(results)}")
    
    # 去重与排序
    dedup: List[Tuple[List[str], float, int, List[str]]] = []
    # seen: Set[Tuple[str, ...]] = set()
    # for route, c, served, names_served in results:
    #     if not route:
    #         continue
    #     key = tuple(route)
    #     if key in seen:
    #         continue
    #     seen.add(key)
    #     dedup.append((route, c, served, names_served))
    best_by_path: Dict[Tuple[str, ...], Tuple[List[str], float, int, List[str]]] = {}
    for route, c, served, names_served in results:
        if not route:
            continue
        key = tuple(route)
        prev = best_by_path.get(key)
        if (prev is None) or (served > prev[2]) or (served == prev[2] and c < prev[1]):
            best_by_path[key] = (route, c, served, names_served)
    dedup = list(best_by_path.values())
    dedup.sort(key=lambda x: (-x[2], x[1]))
    return dedup[:k]


# -------------------- CLI & Runner --------------------
def parse_kv_list(kv_list: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kv_list:
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        try:
            if '.' in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            if v.lower() in {'true', 'false'}:
                out[k] = (v.lower() == 'true')
            else:
                out[k] = v
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DPKMOR - 基于DP/束搜索的多请求插入（与KPURN接口一致）")
    parser.add_argument("--input", required=True, help="边列表文件路径")
    parser.add_argument("--delimiter", default="space", choices=["space", "tab", "comma"])
    parser.add_argument("--ignore-first-col", action="store_true", help="忽略行首编号")
    parser.add_argument("--directed", action="store_true", help="有向图")
    parser.add_argument("--source", required=True, help="源点")
    parser.add_argument("--target", required=True, help="终点")
    parser.add_argument("--requests-file", help="请求文件路径")
    parser.add_argument("--param", nargs="*", default=[], help="key=value 参数，如 k=3 capacity=4 dp_limit=10 beam=64")
    args = parser.parse_args(argv)

    # 读取图
    graph = read_edge_list(args.input,
                           delimiter=(None if args.delimiter == "space" else args.delimiter),
                           directed=args.directed,
                           ignore_first_col=args.ignore_first_col)

    # 读取请求
    requests: List[Request] = []
    if args.requests_file:
        requests = parse_requests_file(args.requests_file)
        print(f"已加载 {len(requests)} 个乘客请求")
        for req in requests:
            print(f"  - {req.name}: {req.pickup} -> {req.dropoff} (需求:{req.demand}, 等待限制:{req.wait_limit:.1f}s)")

    # 解析参数
    params = parse_kv_list(args.param)
    k = int(params.get('k', 3))
    capacity = int(params.get('capacity', 6))
    dp_limit = int(params.get('dp_limit', 10))
    beam = int(params.get('beam', 64))
    max_steps = int(params.get('max_steps', 60000))
    len_limit = int(params.get('len_limit', 60))

    # 运行
    start_time = time.time()
    # 使用 DP 插入直接生成路径（不再像 KPURN 那样分步插入）
    debug = bool(params.get('debug', False))
    results = dpkmor_dp(graph, args.source, args.target, requests, k, capacity, dp_limit, beam, max_steps, len_limit, debug)
    end_time = time.time()

    if not results:
        print("未找到可行路径")
        return 1

    print(f"\n算法用时: {end_time - start_time:.2f}秒")
    print(f"找到 {len(results)} 条合理路径:")
    print("=" * 60)

    for i, (path, cost, served, names_served) in enumerate(results, 1):
        print(f"Top{i}: 服务{served}请求, 代价{cost:.2f}, 路径长度{len(path)}")
        print(f"      路径: {' -> '.join(path)}")
        if names_served:
            # 补足显示 pickup/dropoff
            name_to_req = {r.name: r for r in requests}
            items = []
            for n in names_served:
                r = name_to_req.get(n)
                if r:
                    items.append(f"{r.name}({r.pickup}→{r.dropoff})")
            if items:
                print(f"      服务请求: {', '.join(items)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())