import numpy as np
import networkx as nx
from util.search import bfs, dijkstra


def draw_pcfg_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, labels={node: node for node in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos=pos)


def get_graph_property(
    # G,
    funcs,
    node_init,
    node_goal,
    # verbose=False,
    # only_init=False,
    # computations=False,
):
    init_str = str(node_init).replace("'", "")
    goal_str = str(node_goal).replace("'", "")

    def generate_graph(node, graph, depth, max_depth=4):
        if depth > max_depth:
            return
        for func in funcs:
            name = func.__name__
            vars = node.replace("[", "").replace("]", "").split(", ")
            try:
                new_vars = func(vars)
                new_node = str(new_vars).replace("'", "")
                generate_graph(new_node, depth + 1, max_depth)
                # check if edge already exists
                if not graph.has_edge(node, new_node, key=name):
                    graph.add_edge(node, new_node, key=name)
            except:
                continue

    # build forward graph
    G = nx.MultiDiGraph()  # directed graph with possible multiple edges between nodes
    generate_graph(init_str, G, 0, max_depth=6)

    # build backward graph
    G_rev = nx.MultiDiGraph()
    generate_graph(goal_str, G_rev, 0, max_depth=6)

    #
    # paths = nx.all_shortest_paths(G, node_init, node_goal)
    # for path in paths:
    #     shortest_path = path
    #     break
    # radius = len(shortest_path) - 1
    # G_rev = G.reverse()
    # G_initial = nx.generators.ego_graph(
    #     G,
    #     node_init,
    #     radius=radius,
    #     undirected=False,
    # )  # default: directed
    # G_goal = nx.generators.ego_graph(
    #     G_rev,
    #     node_goal,
    #     radius=radius,
    #     undirected=False,
    # )
    # num_nodes_init = len(G_initial.nodes()) - 1
    # num_nodes_goal = len(G_goal.nodes()) - 1
    # _, _, num_computations_fwd = dijkstra(G, node_init, node_goal)
    # _, _, num_computations_back = dijkstra(G_rev, node_goal, node_init)
    _, _, num_computations_fwd = bfs(G, node_init, node_goal)
    _, _, num_computations_back = bfs(G_rev, node_goal, node_init)
    return (
        G,
        G_rev,
        # num_nodes_init,
        # num_nodes_goal,
        num_computations_fwd,
        num_computations_back,
    )


def generate_pcfg_graph(
    num_fns=0,
    required_path_length=None,
    max_repeat=1,
    # plot_graph=False,
    vars_init=None,
    max_try=20,
    build_funcs=None,
    run_funcs=None,
    max_graph_depth=6,
    verbose=False,
):
    """Avoid both cut and repeat in the same graph."""
    if build_funcs is None:
        build_funcs = pcfg_fns
    if run_funcs is None:
        run_funcs = pcfg_fns

    shortest_path_length = -1
    cnt_try = 0
    while shortest_path_length != required_path_length:

        # print("generating a graph")
        if vars_init is None:
            vars_init_c = np.random.choice(vocab, size=x_size, replace=True).tolist()
        else:
            vars_init_c = vars_init
        while 1:
            fns = np.random.choice(build_funcs, size=num_fns, replace=True).tolist()
            if np.sum([is_expand(fn) for fn in fns]) <= max_repeat and not (
                pcfg_cut in fns and pcfg_repeat in fns
            ):
                break

        # cut makes it difficult to find the path - flip it first
        if pcfg_cut in fns:
            fns_flip = [back_pcfg_fns[pcfg_fns.index(fn)] for fn in fns]
            vars = vars_init_c
            for fn in fns_flip:
                vars = fn(vars)
            vars_final = vars

            # now flip back!
            vars_init_copy = vars_init_c
            vars_init_c = vars_final
            vars_final = vars_init_copy
            if len(vars_final) > 10:
                breakpoint()
        else:
            vars = vars_init_c
            for fn in fns:
                vars = fn(vars)
            vars_final = vars
        # print("vars_final", vars_final)

        problem = ""
        problem += "Initial: " + str(vars_init_c).replace("'", "")
        problem += "\nFinal: " + str(vars_final).replace("'", "")
        # problem += '\nFunctions:'
        problem_flip = ""
        problem_flip += "Initial: " + str(vars_final).replace("'", "")
        problem_flip += "\nFinal: " + str(vars_init_c).replace("'", "")

        verbose and print("*" * 20, "PROBLEM", "*" * 20)
        verbose and print(problem)
        verbose and print("*" * 20, "TRUE ANSWER", "*" * 20)
        verbose and print(fns)

        init_str = str(vars_init_c).replace("'", "")
        goal_str = str(vars_final).replace("'", "")

        def generate_graph(node, graph, depth, max_depth=4):
            if depth > max_depth:
                return
            for func in run_funcs:
                name = func.__name__
                vars = node.replace("[", "").replace("]", "").split(", ")
                try:
                    new_vars = func(vars)
                    new_node = str(new_vars).replace("'", "")
                    generate_graph(new_node, graph, depth + 1, max_depth)
                    # check if edge already exists
                    if not graph.has_edge(node, new_node, key=name):
                        graph.add_edge(node, new_node, key=name)
                except:
                    continue

        # build forward graph
        G = (
            nx.MultiDiGraph()
        )  # directed graph with possible multiple edges between nodes
        generate_graph(init_str, G, 0, max_depth=max_graph_depth)
        # plot_graph and draw_pcfg_graph(G)

        shortest_path_length = nx.shortest_path_length(
            G,
            init_str,
            goal_str,
        )
        if required_path_length is None:
            required_path_length = shortest_path_length
        cnt_try += 1
        if cnt_try > max_try:
            return [None for _ in range(14)]

    # build backward graph
    G_rev = nx.MultiDiGraph()
    generate_graph(goal_str, G_rev, 0, max_depth=max_graph_depth)
    _, _, num_computations_fwd = bfs(G, init_str, goal_str)
    _, _, num_computations_back = bfs(G_rev, goal_str, init_str)

    # make example prompt
    fwd_example = (
        "Initial: "
        + str(vars_init_c).replace("'", "")
        + "\nFinal: "
        + str(vars_final).replace("'", "")
        + "\nInitial to Final Steps:"
    )
    vars = vars_init_c
    for fn in fns:
        vars = fn(vars)
        fwd_example += "\n  " + fns_to_str([fn])[0] + ": " + str(vars).replace("'", "")
    fwd_example += "\nFunctions: " + str(fns_to_str(fns)).replace("'", "")
    flip_example = (
        "Initial: "
        + str(vars_final).replace("'", "")
        + "\nFinal: "
        + str(vars_init_c).replace("'", "")
        + "\nInitial to Final Steps:"
    )
    vars = vars_final
    reversed_fns = []
    for fn in reversed(fns):
        reverse_fn = back_pcfg_fns[pcfg_fns.index(fn)]
        reversed_fns.append(reverse_fn)
        vars = reverse_fn(vars)
        flip_example += (
            "\n  " + fns_to_str([reverse_fn])[0] + ": " + str(vars).replace("'", "")
        )
    flip_example += "\nFunctions: " + str(fns_to_str(reversed_fns)).replace("'", "")
    back_example = (
        "Initial: "
        + str(vars_init_c).replace("'", "")
        + "\nFinal: "
        + str(vars_final).replace("'", "")
        + "\nFinal to Initial Steps:"
    )
    vars = vars_final
    reversed_fns = []
    for fn in reversed(fns):
        reverse_fn = back_pcfg_fns[pcfg_fns.index(fn)]
        reversed_fns.append(reverse_fn)
        vars = reverse_fn(vars)
        back_example += "\n  " + fns_to_str([fn])[0] + ": " + str(vars).replace("'", "")
    back_example += "\nFunctions: " + str(fns_to_str(fns[::-1])).replace("'", "")

    node_init = str(vars_init_c).replace("'", "")
    node_goal = str(vars_final).replace("'", "")
    return (
        G,
        G_rev,
        num_computations_fwd,
        num_computations_back,
        node_init,
        node_goal,
        problem,
        problem_flip,
        fns,
        vars_init_c,
        vars_final,
        fwd_example,
        back_example,
        flip_example,
    )


def back_pcfg_repeat(x):
    if len(x) % 2 != 0:
        return "failed"
    half_len = int(len(x) / 2)
    if x[half_len:] == x[:half_len]:
        return x[half_len:]
    else:
        return "failed"


def back_pcfg_cut(x):
    return x + x


def back_pcfg_reverse(x):
    return x[::-1]


def back_pcfg_shift_left(x):
    return x[-1:] + x[:-1]


def back_pcfg_shift_right(x):
    return x[1:] + x[:1]


def back_pcfg_swap(x):
    return x if len(x) < 2 else x[-1:] + x[1:-1] + x[:1]


def pcfg_repeat(x):
    return x + x


def pcfg_cut(x):
    if len(x) % 2 != 0:
        return "failed"
    half_len = int(len(x) / 2)
    if x[half_len:] == x[:half_len]:
        return x[half_len:]
    else:
        return "failed"


def pcfg_reverse(x):
    return x[::-1]


def pcfg_shift_left(x):
    return x[1:] + x[:1]


def pcfg_shift_right(x):
    return x[-1:] + x[:-1]


def pcfg_copy(x):
    return x


def pcfg_swap(x):
    return x if len(x) < 2 else x[-1:] + x[1:-1] + x[:1]


def pcfg_echo(x):
    return x + x[-1:]


def pcfg_append(x, y):
    return x + y


def pcfg_prepend(x, y):
    return y + x


def pcfg_remove_first(x, y):
    return y


def pcfg_remove_second(x, y):
    return x


pcfg_fns = [
    pcfg_reverse,
    pcfg_shift_left,
    pcfg_shift_right,
    pcfg_swap,
    pcfg_repeat,
    pcfg_cut,
]
back_pcfg_fns = [
    back_pcfg_reverse,
    back_pcfg_shift_left,
    back_pcfg_shift_right,
    back_pcfg_swap,
    back_pcfg_repeat,
    back_pcfg_cut,
]


def is_expand(pcfg_fn):
    return pcfg_fn in [
        pcfg_append,
        pcfg_prepend,
        pcfg_repeat,
        pcfg_cut,
    ]


def fns_to_str(fns):
    fn_strs = []
    for fn in fns:
        if fn == pcfg_shift_left or fn == back_pcfg_shift_right:
            fn_strs.append("shift_left")
        elif fn == pcfg_shift_right or fn == back_pcfg_shift_left:
            fn_strs.append("shift_right")
        elif fn == pcfg_swap or fn == back_pcfg_swap:
            fn_strs.append("swap")
        elif fn == pcfg_reverse or fn == back_pcfg_reverse:
            fn_strs.append("reverse")
        elif fn == pcfg_repeat or fn == back_pcfg_cut:
            fn_strs.append("repeat")
        elif fn == pcfg_cut or fn == back_pcfg_repeat:
            fn_strs.append("cut")
        else:
            print("function not in list")
    return fn_strs


def str_to_fns(response, preprocess=True, flip=False):
    if preprocess:
        if "functions: " not in response.lower():
            print("response not finished")
            return []
        answer = response.lower().split("functions: ")[-1].split("\n")[0].strip("[]")
        str_fns = answer.split(", ")
    else:
        str_fns = response

    fns = []
    for fn in str_fns:
        if (fn == "shift_left" and not flip) or (fn == "shift_right" and flip):
            fns.append(pcfg_shift_left)
        elif (fn == "shift_right" and not flip) or (fn == "shift_left" and flip):
            fns.append(pcfg_shift_right)
        elif fn == "swap":
            fns.append(pcfg_swap)
        elif (fn == "repeat" and not flip) or (fn == "cut" and flip):
            fns.append(pcfg_repeat)
        elif (fn == "cut" and not flip) or (fn == "repeat" and flip):
            fns.append(pcfg_cut)
        elif fn == "reverse":
            fns.append(pcfg_reverse)
        else:
            print("function not in list. skip")
    return fns
