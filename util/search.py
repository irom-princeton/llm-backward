import numpy as np
import heapq


def dijkstra(graph, source, target):
    """Run Dijkstra's algorithm to find the shortest path from source to target"""
    distances = {node: np.inf for node in graph}
    distances[source] = 0
    prev = {node: None for node in graph}
    pq = [(0, source)]  # Priority queue (distance, node)
    num_computations = 0

    while pq:
        dist, current = heapq.heappop(pq)
        if current == target:
            break

        for neighbor in graph.neighbors(current):
            num_computations += 1
            new_distance = (
                dist + 1
            )  # assumes fixed edge length, graph[current][neighbor]
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                prev[neighbor] = current
                heapq.heappush(pq, (new_distance, neighbor))

    path = []
    current = target
    while prev[current] is not None:
        path.append(current)
        current = prev[current]
    path.append(source)
    path.reverse()

    return distances[target], path, num_computations


def bfs(graph, source, target):
    """Run breadth-first search to find the shortest path from source to target"""
    distances = {node: np.inf for node in graph}
    distances[source] = 0
    explored = [source]
    prev = {node: None for node in graph}
    open_set = [(0, source)]  # Open nodes (distance, node)
    num_computations = 0

    while open_set:
        dist, current = open_set.pop(0)
        if current == target:
            break

        for neighbor in graph.neighbors(current):
            num_computations += 1
            new_distance = (
                dist + 1
            )  # assumes fixed edge length, graph[current][neighbor]
            if neighbor not in explored:
                distances[neighbor] = new_distance
                prev[neighbor] = current
                open_set.append((new_distance, neighbor))
                explored.append(neighbor)

    path = []
    current = target
    while prev[current] is not None:
        path.append(current)
        current = prev[current]
    path.append(source)
    path.reverse()

    return distances[target], path, num_computations
