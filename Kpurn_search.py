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
                u = parts[1];
                v = parts[2];
                w_str = parts[3]
            else:
                if len(parts) < 3:
                    continue
                u = parts[0];
                v = parts[1];
                w_str = parts[2]
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
                name = parts[0];
                pickup = parts[1];
                dropoff = parts[2]
                try:
                    demand = int(parts[3])
                except ValueError:
                    demand = 1
                try:
                    wait_limit = float(parts[4])
                except ValueError:
                    wait_limit = math.inf
            elif len(parts) == 4:
                name = f"r{ln}";
                pickup = parts[0];
                dropoff = parts[1]
                try:
                    demand = int(parts[2])
                except ValueError:
                    demand = 1
                try:
                    wait_limit = float(parts[3])
                except ValueError:
                    wait_limit = math.inf
            else:  # len == 2
                name = f"r{ln}";
                pickup = parts[0];
                dropoff = parts[1];
                demand = 1;
                wait_limit = math.inf
            reqs.append(Request(name=name, pickup=str(pickup), dropoff=str(dropoff),
                                demand=demand, wait_limit=wait_limit))
    return reqs


# -------------------- Dijkstra's Shortest Path --------------------
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
    """计算路径代价"""
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        found = False
        for neighbor, cost in graph.neighbors(u):
            if neighbor == v:
                total += cost
                found = True
                break
        if not found:
            return math.inf
    return total


def path_subcost(graph: Graph, path: List[str], i: int, j: int) -> float:
    """计算路径中下标区间 [i, j] 的成本。若无效则返回 inf。"""
    if i < 0 or j >= len(path) or i >= j:
        return 0.0 if i == j else math.inf
    return path_cost(graph, path[i:j + 1])


def per_request_ok(path: List[str], request: Request, graph: Graph) -> bool:
    """判断路径能否在个人绕行限制内服务该请求。"""
    pairs = _served_pairs(path, [request])
    if request.name not in pairs:
        return False
    pi, di = pairs[request.name]
    pd_cost_on_new = path_subcost(graph, path, pi, di)
    _, pd_shortest = dijkstra_shortest_path(graph, request.pickup, request.dropoff)
    if math.isinf(pd_shortest) or math.isinf(pd_cost_on_new):
        return False
    pd_detour = pd_cost_on_new - pd_shortest
    return pd_detour <= request.wait_limit


def select_feasible_subset(path: List[str], requests: List[Request], capacity: int, graph: Graph) -> Set[str]:
    """在给定路径上，选择一个容量可行的服务请求子集，尽量数量多（启发式）。
    策略：按索引扫描，drop先于pick；当超载时，移除当前在车中“需求最大、且最晚下车”的请求。
    返回被保留服务的请求名集合。
    """
    served_pairs = _served_pairs(path, requests)
    # 仅保留个人绕行合规的候选
    candidate = {r.name: r for r in requests if r.name in served_pairs and per_request_ok(path, r, graph)}
    if not candidate:
        return set()

    # 事件表
    drop_at: Dict[int, List[str]] = {}
    pick_at: Dict[int, List[str]] = {}
    for rname, (pi, di) in served_pairs.items():
        if rname not in candidate:
            continue
        drop_at.setdefault(di, []).append(rname)
        pick_at.setdefault(pi, []).append(rname)

    # 活动集合：存储 (rname, demand, drop_idx)
    active: List[Tuple[str, int, int]] = []
    total = 0
    kept: Set[str] = set()

    for i in range(len(path)):
        # 先下车
        for rname in drop_at.get(i, []) or []:
            # 下车仅对仍在 active 的起作用
            for idx in range(len(active)):
                if active[idx][0] == rname:
                    _, d, _ = active.pop(idx)
                    total = max(0, total - d)
                    break
        # 再上车
        for rname in pick_at.get(i, []) or []:
            r = candidate[rname]
            _, di = served_pairs[rname]
            active.append((rname, r.demand, di))
            kept.add(rname)
            total += r.demand
            # 如超载，移除最差项直到可行
            if total > capacity:
                while total > capacity and active:
                    # 选需求最大，若并列则选 drop 最晚
                    worst_idx = max(range(len(active)), key=lambda k: (active[k][1], active[k][2]))
                    wname, wd, _ = active.pop(worst_idx)
                    if wname in kept:
                        kept.remove(wname)
                    total -= wd

    return kept


def is_reasonable_path(path: List[str]) -> bool:
    """检查路径是否合理"""
    if len(path) > 30:  # 路径太长
        return False

    # 检查是否有明显的来回重复 (A -> B -> A 模式)
    for i in range(1, len(path) - 1):
        if path[i - 1] == path[i + 1]:
            return False

    # 检查单个节点重复次数
    from collections import Counter
    node_counts = Counter(path)
    for count in node_counts.values():
        if count > 3:  # 任何节点出现超过3次
            return False

    return True


def remove_unnecessary_repeats(path: List[str]) -> List[str]:
    """移除不必要的重复节点"""
    if len(path) <= 2:
        return path

    cleaned = [path[0]]
    for i in range(1, len(path)):
        # 避免 A -> B -> A 模式
        if i < len(path) - 1 and path[i - 1] == path[i + 1]:
            continue
        # 避免连续重复
        if path[i] != cleaned[-1]:
            cleaned.append(path[i])

    return cleaned


# -------------------- 平衡的插入策略 --------------------
def balanced_insert(graph: Graph, base_path: List[str], request: Request, capacity: int) -> Optional[
    Tuple[List[str], float]]:
    """
    平衡的插入策略：使用最短路径连接，但避免重复
    """
    base_cost = path_cost(graph, base_path)
    best = None
    n = len(base_path)

    if request.pickup not in graph.adj or request.dropoff not in graph.adj:
        return None

    # 选择插入锚点：路径较短尝试全位置，否则抽样
    if n <= 50:
        positions = list(range(n))
    else:
        step = max(1, n // 8)
        positions = list(range(0, n, step))
        if (n - 1) not in positions:
            positions.append(n - 1)

    for i in positions:
        for j in positions:
            if i >= j:
                continue

            # 计算从base_path[i]到pickup的最短路径
            path1, cost1 = dijkstra_shortest_path(graph, base_path[i], request.pickup)
            if not path1: continue

            # 计算从pickup到dropoff的最短路径
            path2, cost2 = dijkstra_shortest_path(graph, request.pickup, request.dropoff)
            if not path2: continue

            # 计算从dropoff到base_path[j]的最短路径
            path3, cost3 = dijkstra_shortest_path(graph, request.dropoff, base_path[j])
            if not path3: continue

            # 构建新路径
            new_path = (
                    base_path[:i + 1] +
                    path1[1:] +  # 去掉重复的base_path[i]
                    path2[1:] +  # 去掉重复的pickup
                    path3[1:] +  # 去掉重复的dropoff
                    base_path[j + 1:]
            )

            # 清理路径
            new_path = remove_unnecessary_repeats(new_path)

            # 检查路径合理性
            if not is_reasonable_path(new_path):
                continue

            new_cost = path_cost(graph, new_path)
            if new_cost == math.inf:
                continue

            # 乘客个体绕行约束：新路径中该乘客的上车到下车的成本相对最短路的增量不得超过 wait_limit
            pairs = _served_pairs(new_path, [request])
            if request.name not in pairs:
                continue
            pi, di = pairs[request.name]
            pd_cost_on_new = path_subcost(graph, new_path, pi, di)
            _, pd_shortest = dijkstra_shortest_path(graph, request.pickup, request.dropoff)
            if math.isinf(pd_shortest) or math.isinf(pd_cost_on_new):
                continue
            pd_detour = pd_cost_on_new - pd_shortest
            if pd_detour > request.wait_limit:
                continue

            # 容量检查（仅对当前请求即可；二次阶段会对全体请求检查）
            if capacity_feasible(new_path, [request], capacity) and (best is None or new_cost < best[1]):
                best = (new_path, new_cost)

    return best


# -------------------- 平衡的KPURN算法 --------------------
def kpurn_search_balanced(graph: Graph, source: str, target: str,
                          requests: List[Request], capacity: int, k: int = 3,
                          beam: int = 5, max_iters: int = 3) -> List[Tuple[List[str], float, int]]:
    """
    平衡的KPURN算法 - 在路径质量和多样性之间平衡
    """
    print("正在计算基础路径...")
    # 步骤1: 找到基础最短路径
    base_path, base_cost = dijkstra_shortest_path(graph, source, target)
    if not base_path:
        print("错误: 无法找到从源点到终点的路径")
        return []

    results = []
    base_keep = select_feasible_subset(base_path, requests, capacity, graph)
    results.append((base_path, base_cost, len(base_keep)))
    print(f"基础路径: 服务{len(base_keep)}请求, 代价{base_cost:.2f}, 长度{len(base_path)}")

    # 步骤2: 对每个请求尝试插入
    unserved = [req for req in requests if req.name not in select_feasible_subset(base_path, requests, capacity, graph)]
    print(f"未服务请求: {[req.name for req in unserved]}")

    for request in unserved:
        inserted = balanced_insert(graph, base_path, request, capacity)
        if inserted:
            new_path, new_cost = inserted
            keep = select_feasible_subset(new_path, requests, capacity, graph)
            results.append((new_path, new_cost, len(keep)))
            print(f"插入 {request.name}: 服务{len(keep)}请求, 代价{new_cost:.2f}, 长度{len(new_path)}")

    # 步骤3: 束搜索（多轮组合插入），尽量服务更多乘客
    improved_results: List[Tuple[List[str], float, int]] = list(results)
    frontier: List[Tuple[List[str], float, int]] = list(results)
    seen_paths: Set[Tuple[str, ...]] = {tuple(p) for p, _, _ in results}

    for it in range(max_iters):
        next_candidates: List[Tuple[List[str], float, int]] = []
        for path, cost, served in frontier:
            if served == len(requests):
                continue
            remaining_requests = [req for req in requests if not is_request_served(path, req)]
            for req in remaining_requests:
                inserted = balanced_insert(graph, path, req, capacity)
                if not inserted:
                    continue
                new_path, new_cost = inserted
                keep = select_feasible_subset(new_path, requests, capacity, graph)
                key = tuple(new_path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                next_candidates.append((new_path, new_cost, len(keep)))
                print(f"组合插入 {req.name}: 服务{len(keep)}请求, 代价{new_cost:.2f}, 长度{len(new_path)}")

        if not next_candidates:
            break
        # 排序并裁剪束宽度
        next_candidates.sort(key=lambda x: (-x[2], x[1]))
        next_candidates = next_candidates[:max(beam, k)]
        improved_results.extend(next_candidates)
        frontier = next_candidates

    # 步骤4: 过滤、排序和去重（使用新的去重集合，避免与上一步的探索去重混淆）
    valid_results: List[Tuple[List[str], float, int]] = []
    final_seen: Set[Tuple[str, ...]] = set()

    for path, cost, served in improved_results:
        # 使用可服务子集而非强制全服务；保持路径合理性
        if is_reasonable_path(path):
            key = tuple(path)
            if key not in final_seen:
                valid_results.append((path, cost, served))
                final_seen.add(key)

    # 按服务请求数降序，代价升序排序
    valid_results.sort(key=lambda x: (-x[2], x[1]))

    return valid_results[:k]


def count_served_requests(path: List[str], requests: List[Request]) -> int:
    """计算路径服务的请求数量"""
    served = 0
    for req in requests:
        if is_request_served(path, req):
            served += 1
    return served


def is_request_served(path: List[str], request: Request) -> bool:
    """检查请求是否被路径服务：第一次到达pickup，之后第一次到达dropoff"""
    pickup_idx: Optional[int] = None
    for i, node in enumerate(path):
        if pickup_idx is None:
            if node == request.pickup:
                pickup_idx = i
        else:
            if node == request.dropoff:
                return True
    return False


def _served_pairs(path: List[str], requests: List[Request]) -> Dict[str, Tuple[int, int]]:
    """返回所有被路径服务的请求的(上车索引, 下车索引)。"""
    pos_by_node: Dict[str, List[int]] = {}
    for idx, node in enumerate(path):
        pos_by_node.setdefault(node, []).append(idx)

    served: Dict[str, Tuple[int, int]] = {}
    for req in requests:
        p_list = pos_by_node.get(req.pickup)
        d_list = pos_by_node.get(req.dropoff)
        if not p_list or not d_list:
            continue
        for pi in p_list:
            di = next((dj for dj in d_list if dj > pi), None)
            if di is not None:
                served[req.name] = (pi, di)
                break
    return served


def capacity_feasible(path: List[str], requests: List[Request], capacity: int) -> bool:
    """全局容量检查：同一索引先下车再上车。"""
    served = _served_pairs(path, requests)
    drop_at: Dict[int, List[Request]] = {}
    pick_at: Dict[int, List[Request]] = {}
    rmap: Dict[str, Request] = {r.name: r for r in requests}
    for rname, (pi, di) in served.items():
        r = rmap[rname]
        drop_at.setdefault(di, []).append(r)
        pick_at.setdefault(pi, []).append(r)

    load = 0
    for i in range(len(path)):
        for r in drop_at.get(i, []):
            load = max(0, load - r.demand)
        for r in pick_at.get(i, []):
            load += r.demand
            if load > capacity:
                return False
    return True


# -------------------- CLI 接口 --------------------
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
    parser = argparse.ArgumentParser(description="KPURN算法 - 平衡版")
    parser.add_argument("--input", required=True, help="边列表文件路径")
    parser.add_argument("--delimiter", default="space", choices=["space", "tab", "comma"])
    parser.add_argument("--ignore-first-col", action="store_true", help="忽略行首编号")
    parser.add_argument("--directed", action="store_true", help="有向图")
    parser.add_argument("--source", required=True, help="源点")
    parser.add_argument("--target", required=True, help="终点")
    parser.add_argument("--requests-file", help="请求文件路径")
    parser.add_argument("--param", nargs="*", default=[], help="key=value 参数，如 k=3 capacity=4")
    args = parser.parse_args(argv)

    # 读取图
    graph = read_edge_list(args.input,
                           delimiter=(None if args.delimiter == "space" else args.delimiter),
                           directed=args.directed,
                           ignore_first_col=args.ignore_first_col)

    # 读取请求
    requests = []
    if args.requests_file:
        requests = parse_requests_file(args.requests_file)
        print(f"已加载 {len(requests)} 个乘客请求")
        for req in requests:
            print(f"  - {req.name}: {req.pickup} -> {req.dropoff} (需求:{req.demand}, 等待限制:{req.wait_limit:.1f}s)")

    # 解析参数
    params = parse_kv_list(args.param)
    k = params.get('k', 3)
    capacity = params.get('capacity', 6)

    # 运行算法
    start_time = time.time()
    results = kpurn_search_balanced(graph, args.source, args.target, requests, capacity, k)
    end_time = time.time()

    # 输出结果
    if not results:
        print("未找到可行路径")
        return 1

    print(f"\n算法用时: {end_time - start_time:.2f}秒")
    print(f"找到 {len(results)} 条合理路径:")
    print("=" * 60)

    for i, (path, cost, served) in enumerate(results, 1):
        # 基于容量与个人绕行限制，选取真正可服务的子集，用于展示
        keep_names = select_feasible_subset(path, requests, capacity, graph)
        keep_list = [req for req in requests if req.name in keep_names]

        print(f"Top{i}: 服务{len(keep_list)}请求, 代价{cost:.2f}, 路径长度{len(path)}")
        print(f"      路径: {' -> '.join(path)}")

        if keep_list:
            served_reqs = [f"{req.name}({req.pickup}→{req.dropoff})" for req in keep_list]
            print(f"      服务请求: {', '.join(served_reqs)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())