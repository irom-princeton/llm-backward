reorder_incident_header = """
You will be asked to re-order a directed graph.

** Example **
Nodes 8 points to node 0
Nodes 4, 10 points to node 1
Nodes 5 points to node 2
Nodes 1, 9, 11 points to node 3
Nodes 5, 6, 11 points to node 4
Nodes  points to node 5
Nodes 2, 9 points to node 6
Nodes 1, 10 points to node 7
Nodes 2, 4, 10 points to node 8
Nodes 10 points to node 9
Nodes 2, 3, 7 points to node 10
Nodes 1, 2 points to node 11

Full procedure:
1. List all directed edges
8 -> 0
4 -> 1
10 -> 1
5 -> 2
1 -> 3
9 -> 3
11 -> 3
5 -> 4
6 -> 4
11 -> 4
2 -> 6
9 -> 6
1 -> 7
10 -> 7
2 -> 8
4 -> 8
10 -> 8
10 -> 9
2 -> 10
3 -> 10
7 -> 10
1 -> 11
2 -> 11
2. Group the edges for each node
0 ->
1 -> 3, 7, 11
2 -> 6, 8, 10, 11
3 -> 10
4 -> 1, 8
5 -> 2, 4
6 -> 4
7 -> 10
9 -> 3, 6
10 -> 1, 7, 8, 9
11 -> 3, 4
3. Convert the edges into the text format
Node 1 points to nodes 3, 7, 11
Node 2 points to nodes 6, 8, 10, 11
Node 3 points to node 10
Node 4 points to nodes 1, 8
Node 5 points to nodes 2, 4
Node 6 points to node 4
Node 7 points to node 10
Node 8 points to node 0
Node 9 points to nodes 3, 6
Node 10 points to nodes 1, 7, 8, 9
Node 11 points to nodes 3, 4

** Current Graph **
"""
