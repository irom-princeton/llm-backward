"""
Run graph experiments

"""

import os
import datetime
import pickle
from tqdm import tqdm
import random
import numpy as np
import logging
from collections import deque

from graph.make_graph import (
    build_graph_prompt_fixed_length,
    build_graph_prompt_verify,
    get_graph_property,
)
from graph.prompts import reorder_incident_header
from util.llm import call_lm
from util.strings import flip_numbers_in_string


def main(args):
    # Config
    num_runs = args.num_runs
    model = args.model_name
    setting = args.setting
    directed = args.directed
    num_nodes = args.num_nodes
    edge_rate = args.edge_rate
    len_shortest_path = args.len_shortest_path
    verbose = args.verbose

    # Settings
    forward_only, backward_only, flip_only, no_backward = False, False, False, False
    if setting == "fwd_verify":
        forward_only = True
    elif setting == "back_verify":
        backward_only = True
    elif setting == "flip_verify":
        flip_only = True
    elif setting == "fwd_flip_verify":
        no_backward = True
    elif setting == "fwd_back_verify":
        pass  # no_backward = False
    reorder_incident = directed and (flip_only or no_backward)

    # Defaults
    incident = True
    use_bfs_for_computations = True
    num_shot = 3
    num_verify_shot = 3
    bad_rate = 0.9
    target_candidates = 4
    prev_try_maxlen = 4
    max_try_ratio = 2  # large value -> expensive!
    fwd_T = 0.5  # temperature for first guess
    back_T = 0.5
    flip_T = 0.5

    # holders
    init_goal_ratios = []
    num_init_nodes = []
    all_num_computations_fwd = []
    all_num_computations_back = []
    verify_successes = []
    candidates_all = []
    raw_verify_sols = []
    verify_sols = []
    all_fwd_graph_solns_all = []
    graph_all = []

    # Run
    header = f"You will be given {'a directed' if directed else 'an undirected'} graph search problem with a few examples."
    fwd_ender = f"\n\nPlan the shortest path from initial to goal node for the this **{'directed' if directed else 'undirected'}** graph. Follow the format 'Shortest Path: (...)' and do not output anything else."
    back_ender = f"\n\nPlan the reversed shortest path from goal to initial node for the this **{'directed' if directed else 'undirected'}** graph. Follow the format 'Reversed Shortest Path: (...)' and do not output anything else."
    for _ in tqdm(range(num_runs)):
        fwd_prompt = header
        back_prompt = header
        flip_prompt = header
        fwd_T_trial = 0  # always start with deterministic
        back_T_trial = 0
        flip_T_trial = 0

        # few-shot examples
        for example_ind in range(num_shot):
            (
                fwd_graph_prompt,
                back_graph_prompt,
                flip_graph_prompt,
                fwd_graph_soln,
                back_graph_soln,
                G,
                _,
                _,
                _,
                _,
            ) = build_graph_prompt_fixed_length(
                len_shortest_path=len_shortest_path,
                num_nodes=num_nodes,
                edge_rate=edge_rate,
                directed=directed,
                incident=incident,
                use_preamble=False,
            )
            fwd_prompt += (
                f"\n\n** Example {example_ind+1} **\n"
                + fwd_graph_prompt
                + " "
                + fwd_graph_soln
            )
            back_prompt += (
                f"\n\n** Example {example_ind+1} **\n"
                + back_graph_prompt
                + " "
                + back_graph_soln
            )
            if reorder_incident:
                flip_prompt += (
                    f"\n\n** Example {example_ind+1} **\n"
                    + fwd_graph_prompt
                    + " "
                    + fwd_graph_soln
                )
            else:
                flip_prompt += (
                    f"\n\n** Example {example_ind+1} **\n"
                    + flip_graph_prompt
                    + " "
                    + back_graph_soln
                )
            (
                fwd_graph_prompt,
                back_graph_prompt,
                flip_graph_prompt,
                fwd_graph_soln,
                back_graph_soln,
                G,
                node_init,
                node_goal,
                all_fwd_graph_solns,
                all_back_graph_solns,
            ) = build_graph_prompt_fixed_length(
                len_shortest_path=len_shortest_path,
                num_nodes=num_nodes,
                edge_rate=edge_rate,
                directed=directed,
                incident=incident,
                use_preamble=False,
                add_path_prompt=False,
            )
        fwd_prompt_with_problem = (
            fwd_prompt + "\n\n** Current problem **\n" + fwd_graph_prompt + fwd_ender
        ).strip()
        back_prompt_with_problem = (
            back_prompt + "\n\n** Current problem **\n" + back_graph_prompt + back_ender
        ).strip()
        flip_prompt_with_problem = (
            flip_prompt + "\n\n** Current problem **\n" + flip_graph_prompt + fwd_ender
        ).strip()
        verbose and logging.info(f"\n\n==== FWD PROMPT====\n{fwd_prompt_with_problem}")
        verbose and logging.info(
            f"\n\n==== BACK PROMPT====\n{back_prompt_with_problem}"
        )
        verbose and logging.info(
            f"\n\n==== FLIP PROMPT====\n{flip_prompt_with_problem}"
        )

        # get properties
        num_computations_fwd, num_computations_back = get_graph_property(
            G,
            node_init,
            node_goal,
            computations=True,
            use_bfs=use_bfs_for_computations,  # or dijkstra
            directed=directed,
        )
        all_num_computations_fwd.append(num_computations_fwd)
        all_num_computations_back.append(num_computations_back)
        init_goal_ratios.append(get_graph_property(G, node_init, node_goal))
        num_init_nodes.append(
            get_graph_property(
                G,
                node_init,
                node_goal,
                only_init=True,
            )
        )

        # reorder incident for flip - assume fewshot examples do not need to be re-ordered, essentially using forward examples
        if reorder_incident:
            print("reordering...")
            cur_graph_str = flip_graph_prompt.split("Initial")[0].strip()
            reorder_query = (
                reorder_incident_header
                + cur_graph_str
                + "\n\nRemember the edges are directed. Please re-order this directed graph with the exact same full procedure as the example. Follow the same format and do not output anything else."
            )
            verbose and logging.info(f"\n\n==== REORDER PROMPT====\n{reorder_query}")
            reorder_raw_response = call_lm(
                reorder_query,
                model=model,
                max_tokens=2048,
                temperature=0,
            )[0]
            try:
                reorder_graph_str = "Node" + reorder_raw_response.split("Node", 1)[1]
            except:
                logging.info(
                    f"Cannot parse reorder response: {reorder_raw_response}. Use full!"
                )
                reorder_graph_str = reorder_raw_response
            flip_prompt_with_problem = flip_prompt_with_problem.replace(
                cur_graph_str, reorder_graph_str
            )

        # get forward and backward solutions from LM
        max_try = max_try_ratio * target_candidates
        candidates_dir = []
        prev_try = deque(maxlen=prev_try_maxlen)
        for _ in range(max_try):
            try:
                use_fwd = random.random() < 0.5
                if forward_only or (not backward_only and not flip_only and use_fwd):
                    logging.info("Generating forward...")
                    direction = "fwd"
                    candidate = (
                        call_lm(
                            fwd_prompt_with_problem,
                            model=model,
                            max_tokens=32,
                            temperature=fwd_T_trial,
                        )[0]
                        .rsplit("Shortest Path: ", 1)[1]
                        .strip()
                    )
                    fwd_T_trial = fwd_T  # update
                    prev_try.append(candidate)
                    if candidate not in [c[0] for c in candidates_dir]:
                        candidates_dir.append((candidate, direction))
                    verbose and logging.info(f"\n\n==== FWD GEN====\n{candidate}")
                elif backward_only or (
                    not forward_only
                    and not flip_only
                    and not no_backward
                    and not use_fwd
                ):
                    logging.info("Generating backward...")
                    direction = "back"
                    rev_candidate = (
                        call_lm(
                            back_prompt_with_problem,
                            model=model,
                            max_tokens=32,
                            temperature=back_T_trial,
                        )[0]
                        .rsplit("Reversed Shortest Path: ", 1)[1]
                        .strip()
                    )
                    back_T_trial = back_T
                    rev_candidate = flip_numbers_in_string(rev_candidate)
                    prev_try.append(rev_candidate)
                    if rev_candidate not in [c[0] for c in candidates_dir]:
                        candidates_dir.append((rev_candidate, direction))
                    verbose and logging.info(f"\n\n==== BACK GEN====\n{rev_candidate}")
                elif flip_only or (
                    not forward_only
                    and not backward_only
                    and no_backward
                    and not use_fwd
                ):
                    logging.info("Generating flipped...")
                    direction = "flip"
                    rev_candidate = (
                        call_lm(
                            flip_prompt_with_problem,
                            model=model,
                            max_tokens=32,
                            temperature=flip_T_trial,
                        )[0]
                        .rsplit("Shortest Path: ", 1)[1]
                        .strip()
                    )
                    flip_T_trial = flip_T
                    rev_candidate = flip_numbers_in_string(rev_candidate)
                    prev_try.append(rev_candidate)
                    if rev_candidate not in [c[0] for c in candidates_dir]:
                        candidates_dir.append((rev_candidate, direction))
                    verbose and logging.info(f"\n\n==== FLIP GEN====\n{rev_candidate}")
                else:
                    raise NotImplementedError
                if len(candidates_dir) == target_candidates:
                    break
                # break if all in prev_try are the same
                if len(prev_try) == prev_try_maxlen and len(set(prev_try)) == 1:
                    break
            except:
                logging.info(f"Error in parsing the output.")
                continue
        random.shuffle(candidates_dir)
        candidates = [c[0] for c in candidates_dir]

        #######################################
        ######     self-verification     ######
        #######################################
        labels = ["A", "B", "C", "D"]
        verify_examples_prompt = header
        for example_ind in range(num_verify_shot):
            example = build_graph_prompt_verify(
                len_shortest_path=len_shortest_path,
                num_nodes=num_nodes,
                edge_rate=edge_rate,
                bad_rate=bad_rate,
                directed=directed,
                incident=incident,
                use_preamble=False,
            )
            verify_examples_prompt += f"\n\n** Example {example_ind+1} **\n" + example
        verify_prompt = (
            verify_examples_prompt
            + "\n\n** Current problem **\n"
            + fwd_graph_prompt.rsplit("Shortest Path:", 1)[0]
            + "\nWhich one is the correct shortest path?\n"
        )
        for i in range(len(candidates)):
            verify_prompt += labels[i] + ". " + candidates[i] + "\n"
        verify_prompt += f"Remember the graph is {'directed' if directed else 'undirected'}. Follow the exact same format as the examples and check each options step by step. Begin with 'Checking each options step by step:'"
        raw_verify_sol = call_lm(
            verify_prompt,
            model=model,
            max_tokens=1024,
            temperature=0,
            stop=["Print", "Since"],  # tend to ramble from Since...
        )[0].strip()
        try:
            verify_sol_label = raw_verify_sol.split("Thus")[1].strip()
        except:
            verify_sol_label = "None"
            logging.info(f"Error: {verify_sol_label}")
        verbose and logging.info(f"\n\n==== VERIFY PROMPT====\n{verify_prompt}")
        logging.info(f"\n\n==== VERIFY RESPONSE ====\n{raw_verify_sol}")
        possible_labels = []
        for label in labels[: len(candidates)]:
            if label in verify_sol_label:  # assume there is no other capital letter
                possible_labels.append(label)
        if len(possible_labels) == 0:
            logging.info(
                f"Unknown verify answer (choose random one): {verify_sol_label}"
            )
            verify_sol_label = random.choice(labels[: len(candidates)])
        else:
            verify_sol_label = random.choice(possible_labels)
        verify_sol = candidates[labels.index(verify_sol_label)]
        verify_successes.append(verify_sol in all_fwd_graph_solns)
        logging.info(
            f"==== FINAL ====\ncandidates {candidates}, chosen {verify_sol}, sols {all_fwd_graph_solns}, success: {verify_successes[-1]}"
        )

        # Save
        candidates_all.append(candidates_dir)  # not shuffled
        verify_sols.append(verify_sol)
        raw_verify_sols.append(raw_verify_sol)
        all_fwd_graph_solns_all.append(all_fwd_graph_solns)
        graph_all.append(
            [
                fwd_graph_prompt,
                back_graph_prompt,
                flip_graph_prompt,
                fwd_graph_soln,
                back_graph_soln,
                G,
                node_init,
                node_goal,
                all_fwd_graph_solns,
                all_back_graph_solns,
                verify_examples_prompt,
                fwd_prompt,
                back_prompt,
            ]
        )

        # save results
        with open(args.result_path, "wb") as f:
            pickle.dump(
                {
                    "all_num_computations_fwd": all_num_computations_fwd,
                    "all_num_computations_back": all_num_computations_back,
                    "init_goal_ratios": init_goal_ratios,
                    "verify_successes": verify_successes,
                    "num_init_nodes": num_init_nodes,
                    "num_goal_nodes": np.asarray(num_init_nodes)
                    / np.asarray(init_goal_ratios),
                    "candidates": candidates_all,
                    "verify_sols": verify_sols,
                    "raw_verify_sols": raw_verify_sols,
                    "all_fwd_graph_solns": all_fwd_graph_solns_all,
                    "graphs": graph_all,
                },
                f,
            )
        logging.info(f"saved files to: {args.result_path}")


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument(
        "--setting",
        type=str,
        default="fwd_flip_verify",
        help="one of {fwd_verify, back_verify, flip_verify, fwd_flip_verify, fwd_back_verify}",
    )
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=12)
    parser.add_argument("--edge_rate", type=float, default=0.2)
    parser.add_argument("--len_shortest_path", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    save_name = f"{args.setting}_{args.directed}_{args.num_nodes}_{int(args.edge_rate * 100)}_{args.len_shortest_path}_{args.num_runs}_{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}.pkl"
    log_dir = os.environ.get("BACKWARD_LOG_DIR")
    result_path = os.path.join(log_dir, "graph", save_name)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    log_path = result_path.replace(".pkl", ".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    args.result_path = result_path

    # Run
    main(args)
