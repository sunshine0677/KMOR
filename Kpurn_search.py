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

    # 限制搜索位置
    positions = list(range(0, n, max(1, n // 5)))
    if n - 1 not in positions:
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

            detour = new_cost - base_cost
            if detour > request.wait_limit:
                continue

            # 容量检查
            load = 0
            feasible = True
            delivered = False
            for node in new_path:
                if node == request.pickup:
                    load += request.demand
                if node == request.dropoff and not delivered:
                    delivered = True
                    load = max(0, load - request.demand)
                if load > capacity:
                    feasible = False
                    break

            if feasible and (best is None or new_cost < best[1]):
                best = (new_path, new_cost)

    return best


# -------------------- 平衡的KPURN算法 --------------------
def kpurn_search_balanced(graph: Graph, source: str, target: str,
                          requests: List[Request], capacity: int, k: int = 3) -> List[Tuple[List[str], float, int]]:
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
    base_served = count_served_requests(base_path, requests)
    results.append((base_path, base_cost, base_served))
    print(f"基础路径: 服务{base_served}请求, 代价{base_cost:.2f}, 长度{len(base_path)}")

    # 步骤2: 对每个请求尝试插入
    unserved = [req for req in requests if not is_request_served(base_path, req)]
    print(f"未服务请求: {[req.name for req in unserved]}")

    for request in unserved:
        inserted = balanced_insert(graph, base_path, request, capacity)
        if inserted:
            new_path, new_cost = inserted
            new_served = count_served_requests(new_path, requests)
            results.append((new_path, new_cost, new_served))
            print(f"插入 {request.name}: 服务{new_served}请求, 代价{new_cost:.2f}, 长度{len(new_path)}")

    # 步骤3: 对已插入的路径继续插入剩余请求
    improved_results = list(results)
    for path, cost, served in results:
        if served < len(requests):  # 还有未服务的请求
            remaining_requests = [req for req in requests if not is_request_served(path, req)]
            for req in remaining_requests:
                inserted = balanced_insert(graph, path, req, capacity)
                if inserted:
                    new_path, new_cost = inserted
                    new_served = count_served_requests(new_path, requests)
                    if new_served > served:  # 只有真正改进才加入
                        improved_results.append((new_path, new_cost, new_served))
                        print(f"二次插入 {req.name}: 服务{new_served}请求, 代价{new_cost:.2f}, 长度{len(new_path)}")

    # 步骤4: 过滤、排序和去重
    valid_results = []
    seen_paths = set()

    for path, cost, served in improved_results:
        if is_reasonable_path(path):
            path_key = tuple(path)
            if path_key not in seen_paths:
                valid_results.append((path, cost, served))
                seen_paths.add(path_key)

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
    """检查请求是否被路径服务（pickup在dropoff之前）"""
    pickup_idx = -1
    dropoff_idx = -1

    for i, node in enumerate(path):
        if node == request.pickup and pickup_idx == -1:
            pickup_idx = i
        if node == request.dropoff:
            dropoff_idx = i

    return pickup_idx != -1 and dropoff_idx != -1 and pickup_idx < dropoff_idx


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
        print(f"Top{i}: 服务{served}请求, 代价{cost:.2f}, 路径长度{len(path)}")
        print(f"      路径: {' -> '.join(path)}")

        # 显示服务的具体请求
        if served > 0:
            served_reqs = []
            for req in requests:
                if is_request_served(path, req):
                    served_reqs.append(f"{req.name}({req.pickup}→{req.dropoff})")
            print(f"      服务请求: {', '.join(served_reqs)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())