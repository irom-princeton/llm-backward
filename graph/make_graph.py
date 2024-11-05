import networkx as nx
import random
import numpy as np

from util.search import bfs, dijkstra


def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, labels={node: node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos=pos)


def get_graph_property(
    G,
    node_init,
    node_goal,
    verbose=False,
    only_init=False,
    use_bfs=False,
    computations=False,
    directed=False,
):
    """
    This function computes the number of steps when searching in the forward and backward directions
    """
    paths = nx.all_shortest_paths(G, node_init, node_goal)
    for path in paths:
        shortest_path = path
        break
    radius = len(shortest_path) - 1
    G_initial = nx.generators.ego_graph(
        G,
        node_init,
        radius=radius,
        undirected=not directed,
    )
    if directed:  # no need to reverse the graph if undirected
        G_rev = G.reverse()
        G_goal = nx.generators.ego_graph(
            G_rev, node_goal, radius=radius, undirected=not directed
        )
    else:
        G_goal = nx.generators.ego_graph(
            G, node_goal, radius=radius, undirected=not directed
        )
    if not computations:
        num_nodes_init = len(G_initial.nodes()) - 1
        num_nodes_goal = len(G_goal.nodes()) - 1
        verbose and print(
            "nodes within radius:",
            radius,
            ", init:",
            num_nodes_init,
            ", goal:",
            num_nodes_goal,
        )
        if num_nodes_goal == 0 and not only_init:
            return np.inf
        elif not only_init:
            return num_nodes_init / num_nodes_goal
        else:
            return num_nodes_init
    elif use_bfs:
        _, _, num_computations_fwd = bfs(G, node_init, node_goal)
        if directed:
            _, _, num_computations_back = bfs(G_rev, node_goal, node_init)
        else:
            _, _, num_computations_back = bfs(G, node_goal, node_init)
        return num_computations_fwd, num_computations_back
    else:
        _, _, num_computations_fwd = dijkstra(G, node_init, node_goal)
        if directed:
            _, _, num_computations_back = dijkstra(G_rev, node_goal, node_init)
        else:
            _, _, num_computations_back = dijkstra(G, node_goal, node_init)
        return num_computations_fwd, num_computations_back


def build_graph_prompt_fixed_length(
    G=None,
    node_init=None,
    node_goal=None,
    len_shortest_path=4,
    num_nodes=10,
    max_tries=100,
    edge_rate=0.3,
    directed=True,
    incident=False,
    use_preamble=True,
    add_path_prompt=True,
):
    tries = 0
    feasible = False

    if G is None:
        while not feasible:
            G = nx.gnp_random_graph(
                num_nodes,
                edge_rate,
                seed=None,
                directed=directed,
            )
            node_init, node_goal = random.sample(list(G.nodes()), 2)
            try:
                paths = nx.all_shortest_paths(G, node_init, node_goal)
                for path in paths:
                    shortest_path = path
                if not len(shortest_path) == len_shortest_path and tries < max_tries:
                    tries += 1
                    continue
                feasible = True
            except:
                G = nx.gnp_random_graph(
                    num_nodes,
                    edge_rate,
                    seed=None,
                    directed=True,
                )
                node_init, node_goal = random.sample(list(G.nodes()), 2)
    else:
        paths = nx.all_shortest_paths(G, node_init, node_goal)
        for path in paths:
            shortest_path = path
        assert len(shortest_path) == len_shortest_path

    preamble = ""
    if use_preamble:
        if directed:
            preamble = "Print the first shortest path from initial to goal node for the following directed graph.\n\n"
        else:
            preamble = "Print the first shortest path from initial to goal node for the following **undirected** graph.\n\n"
    if not incident:
        preamble += "Edges:\n"
    prompt = preamble
    flip_prompt = preamble
    if incident:
        # also flip
        for n in G.nodes():
            if len([str(x) for x in G.neighbors(n)]) > 0:
                if directed:
                    ver = "points to"
                else:
                    ver = "is connected to"
                prompt += f"Node {n} {ver} nodes {', '.join([str(x) for x in G.neighbors(n)])}\n"
                if directed:
                    flip_prompt += f"Nodes {', '.join([str(x) for x in G.neighbors(n)])} {ver} node {n}\n"
                else:
                    flip_prompt += f"Node {n} {ver} nodes {', '.join([str(x) for x in G.neighbors(n)])}\n"
    else:
        # flip edge for flip
        for e in G.edges():
            prompt += "(" + str(e[0]) + ", " + str(e[1]) + ")\n"
            flip_prompt += "(" + str(e[1]) + ", " + str(e[0]) + ")\n"
    prompt += "\nInitial: " + str(node_init) + "\nGoal: " + str(node_goal)
    flip_prompt += "\nInitial: " + str(node_goal) + "\nGoal: " + str(node_init)

    if add_path_prompt:
        fwd_prompt = prompt + "\nShortest Path:"
        back_prompt = prompt + "\nReverse Shortest Path:"
        flip_prompt += "\nShortest Path:"
    else:
        fwd_prompt = prompt
        back_prompt = prompt

    fwd_soln = str(shortest_path).replace("[", "(").replace("]", ")").strip()
    shortest_path.reverse()
    back_soln = str(shortest_path).replace("[", "(").replace("]", ")").strip()

    all_fwd_solns, all_back_solns = [], []
    paths = nx.all_shortest_paths(G, node_init, node_goal)
    for path in paths:
        shortest_path = path
        fwd_soln = str(shortest_path).replace("[", "(").replace("]", ")").strip()
        all_fwd_solns.append(fwd_soln)
        path.reverse()
        back_soln = str(path).replace("[", "(").replace("]", ")").strip()
        all_back_solns.append(back_soln)

    return (
        fwd_prompt,
        back_prompt,
        flip_prompt,
        fwd_soln,
        back_soln,
        G,
        node_init,
        node_goal,
        all_fwd_solns,
        all_back_solns,
    )


def build_graph_prompt_verify(
    len_shortest_path=4,
    num_nodes=10,
    max_tries=100,
    edge_rate=0.3,
    bad_rate=0.5,
    directed=True,
    incident=False,
    use_preamble=True,
):
    """Sample random path to goal, so not necessarily shortest path length.
    Also sample wrong paths
    """
    while 1:  # sometimes no feasible path
        try:
            G = nx.gnp_random_graph(num_nodes, edge_rate, seed=None, directed=directed)
            node_init, node_goal = random.sample(list(G.nodes()), 2)
            paths = list(nx.all_shortest_paths(G, node_init, node_goal))
            simple_paths = list(nx.all_simple_paths(G, node_init, node_goal))
            random.shuffle(paths)
            random.shuffle(simple_paths)
            shortest_path = paths[0]

            # remove paths longer than 1.5x the shortest path and remove shortest path from simple paths
            new_paths = []
            for path in simple_paths:
                if len(path) < 1.5 * len_shortest_path and path != shortest_path:
                    new_paths.append(path)
            simple_paths = new_paths
            if len(simple_paths) < 3:
                raise Exception("Not enough simple paths")
            break
        except:
            continue
    num_total = random.randint(2, 4)
    all_paths = [shortest_path] + simple_paths[: (num_total - 1)]
    ords = list(range(len(all_paths)))
    random.shuffle(ords)
    answer_ind = ords.index(0)
    all_paths = [all_paths[i] for i in ords]
    for path in all_paths:
        if path == shortest_path or len(path) <= 2:
            continue
        # randomly replace node with another node
        if random.random() < bad_rate:
            ind = random.randint(1, len(path) - 2)  # exclude start and goal
            while 1:
                new_node = random.choice(list(G.nodes()))
                if new_node != path[0] and new_node != path[-1]:
                    break
            path[ind] = new_node

    # put prompt together
    preamble = ""
    if use_preamble:
        if directed:
            preamble += "Print the first shortest path from initial to goal node for the following directed graph.\n\n"
        else:
            preamble += "Print the first shortest path from initial to goal node for the following **undirected** graph.\n\n"
        if not incident:
            preamble += "Edges:\n"
    prompt = preamble
    if incident:
        for n in G.nodes():
            if directed:
                ver = "points to"
            else:
                ver = "is connected to"
            prompt += (
                f"Node {n} {ver} nodes {', '.join([str(x) for x in G.neighbors(n)])}\n"
            )
    else:
        for e in G.edges():
            prompt += "(" + str(e[0]) + ", " + str(e[1]) + ")\n"
    prompt += "\nInitial: " + str(node_init) + "\nGoal: " + str(node_goal)

    fwd_soln = str(shortest_path).replace("[", "(").replace("]", ")").strip()

    # add plan to prompt
    labels = ["A", "B", "C", "D"]
    prompt += "\n\nWhich one is the correct shortest path?\n"
    for ind, path in enumerate(all_paths):
        prompt += (
            labels[ind]
            + ". "
            + str(path).replace("[", "(").replace("]", ")").strip()
            + "\n"
        )
    prompt += "Checking each options step by step:\n"
    valids = []
    for ind, path in enumerate(all_paths):
        prompt += f"{labels[ind]}:"
        valid = True
        for n_ind in range(len(path) - 1):
            cur_node = path[n_ind]
            next_node = path[n_ind + 1]
            neighbors = list(G.neighbors(cur_node))
            if G.has_edge(cur_node, next_node):
                if incident:
                    if directed:
                        prompt += f" check {cur_node} to {next_node}, {cur_node} points to {neighbors}, {next_node} in {neighbors}? True;"
                    else:
                        prompt += f" {cur_node} connected to {next_node}? True;"
                else:
                    prompt += f" edge ({cur_node}, {next_node}), {cur_node} neighbors {neighbors}, {next_node} in neighbors? True;"
            else:
                if incident:
                    if directed:
                        prompt += f" check {cur_node} to {next_node}, {cur_node} points to {neighbors}, {next_node} in {neighbors}? False;"
                    else:
                        prompt += f" {cur_node} connected to {next_node}? False;"
                else:
                    prompt += f" edge ({cur_node}, {next_node}), {cur_node} neighbors {neighbors}, {next_node} in neighbors? False;"
                valid = False
                break
        if valid:
            prompt = prompt[:-1] + f" - valid path of length {len(path)}"
            valids.append((ind, "length " + str(len(path))))
        else:
            prompt = prompt[:-1] + " - invalid path"
        prompt += "\n"
    prompt += f"Valid options:"
    for valid in valids:
        prompt += f" {labels[valid[0]]} with length {valid[1]},"
    prompt = prompt[:-1] + f". Thus the correct shortest option is {labels[answer_ind]}"
    return prompt
