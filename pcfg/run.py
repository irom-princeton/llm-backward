"""
Run Array Transformation (PCFG) experiments


"""

import os
import datetime
import logging
import pickle
import numpy as np
import random
from collections import deque
from tqdm import tqdm

from pcfg.make_graph import (
    generate_pcfg_graph,
    str_to_fns,
    pcfg_reverse,
    pcfg_shift_left,
    pcfg_shift_right,
    pcfg_swap,
    pcfg_repeat,
    pcfg_cut,
)
from pcfg.prompts import get_prompts
from util.llm import call_lm


def main(args):
    # Config
    num_runs = args.num_runs
    model = args.model_name
    verify_model = args.model_verify_name
    setting = args.setting
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
    if args.functions == "shift_repeat_cut":
        build_funcs = [pcfg_shift_left, pcfg_shift_right, pcfg_repeat, pcfg_cut]
        run_funcs = [pcfg_shift_left, pcfg_shift_right, pcfg_repeat, pcfg_cut]
    elif args.functions == "shift_reverse_swap":
        build_funcs = [pcfg_reverse, pcfg_swap, pcfg_shift_left, pcfg_shift_right]
        run_funcs = [pcfg_reverse, pcfg_swap, pcfg_shift_left, pcfg_shift_right]
    elif args.functions == "repeat_cut_reverse_swap":
        build_funcs = [pcfg_reverse, pcfg_swap, pcfg_repeat, pcfg_cut]
        run_funcs = [pcfg_reverse, pcfg_swap, pcfg_repeat, pcfg_cut]
    else:
        raise ValueError(f"Unknown functions: {args.functions}")
    fwd_prompt_header, back_prompt_header, flip_prompt_header, verify_prompt_fixed = (
        get_prompts(args.functions)
    )

    # Defaults
    vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    num_fns = 3
    x_size = 4
    num_shot = 3
    len_shortest_path = num_fns
    max_repeat = 1  # only one repeat to make prompt not too long
    prev_try_maxlen = 5
    max_try = 6
    max_token_candidate = 150
    max_token_verify = 1024
    fwd_T = 0.5  # temperature for first guess
    back_T = 0.5
    flip_T = 0.5

    # Holders
    verify_successes = []
    all_num_computations_fwd = []
    all_num_computations_back = []
    graph_all = []
    candidates_all = []

    # Run
    for _ in tqdm(range(num_runs)):
        fwd_T_trial = 0  # try deterministic first
        back_T_trial = 0
        flip_T_trial = 0

        # Sample fewshots
        fwd_examples = []
        back_examples = []
        flip_examples = []
        for _ in range(num_shot):
            while 1:
                vars_init = np.random.choice(
                    vocab,
                    size=x_size,
                    replace=True,
                ).tolist()
                (
                    G,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    fwd_example,
                    back_example,
                    flip_example,
                ) = generate_pcfg_graph(
                    num_fns=num_fns,
                    required_path_length=len_shortest_path,
                    vars_init=vars_init,
                    max_repeat=max_repeat,
                    build_funcs=build_funcs,
                    run_funcs=run_funcs,
                )
                if G is not None:
                    break
            fwd_examples.append(fwd_example)
            back_examples.append(back_example)
            flip_examples.append(flip_example)

        # Sample current problem
        while 1:
            vars_init = np.random.choice(
                vocab,
                size=x_size,
                replace=True,
            ).tolist()
            (
                G,
                G_rev,
                num_computations_fwd,
                num_computations_back,
                node_init,
                node_goal,
                problem,
                problem_flip,
                fns,
                vars_init,
                vars_final,
                _,
                _,
                _,
            ) = generate_pcfg_graph(
                num_fns=num_fns,
                required_path_length=len_shortest_path,
                vars_init=vars_init,
                max_repeat=max_repeat,
                build_funcs=build_funcs,
                run_funcs=run_funcs,
            )
            if G is not None:
                break

        all_num_computations_fwd.append(num_computations_fwd)
        all_num_computations_back.append(num_computations_back)
        graph_all.append(
            {
                "G": G,
                "G_rev": G_rev,
                "node_init": node_init,
                "node_goal": node_goal,
                "problem": problem,
                "problem_flip": problem_flip,
                "fns": fns,
                "vars_init": vars_init,
                "vars_final": vars_final,
                "fwd_examples": fwd_examples,
                "back_examples": back_examples,
                "flip_examples": flip_examples,
                "run_funcs": run_funcs,
                "build_funcs": build_funcs,
            }
        )

        # Call LLM forward/backward, verify
        prev_tries = deque(maxlen=prev_try_maxlen)
        assert not (forward_only and backward_only)
        candidates_dir = []
        for ind in range(max_try):
            use_fwd = random.random() < 0.5
            if forward_only or (not backward_only and not flip_only and use_fwd):
                logging.info("Generating forward...")
                direction = "fwd"
                query = (
                    fwd_prompt_header
                    + "\n\n".join(fwd_examples)
                    + "\n\n***** Current problem:\n"
                    + problem
                    + "\nPlease solve with the exact same format. Do not repeat the problem."
                )
                response, _ = call_lm(
                    query,
                    model=model,
                    max_tokens=max_token_candidate,
                    temperature=fwd_T_trial,
                )
                fwd_T_trial = fwd_T
            elif backward_only or (
                not no_backward and not forward_only and not flip_only and not use_fwd
            ):
                logging.info("Generating backward...")
                direction = "back"
                query = (
                    back_prompt_header
                    + "\n\n".join(back_examples)
                    + "\n\n***** Current problem:\n"
                    + problem
                    + "\nPlease solve with the exact same format. Do not repeat the problem."
                )
                response, _ = call_lm(
                    query,
                    model=model,
                    max_tokens=max_token_candidate,
                    temperature=back_T_trial,
                )
                back_T_trial = back_T
            elif flip_only or (
                not forward_only and not backward_only and no_backward and not use_fwd
            ):
                logging.info("Generating flipped...")
                direction = "flip"
                query = (
                    flip_prompt_header
                    + "\n\n".join(flip_examples)
                    + "\n\n***** Current problem:\n"
                    + problem_flip
                    + "\nPlease solve with the exact same format. Do not repeat the problem."
                )
                response, _ = call_lm(
                    query,
                    model=model,
                    max_tokens=max_token_candidate,
                    temperature=flip_T_trial,
                )
                flip_T_trial = flip_T
            else:
                raise NotImplementedError
            verbose and logging.info(f"{'*' * 20} GEN {'*' * 20}")
            verbose and logging.info(query)
            verbose and logging.info(response)

            # Extract functions
            if "functions: " not in response.lower():
                logging.info("response not finished")
                continue
            answer = (
                response.lower().split("functions: ")[-1].split("\n")[0].strip("[]")
            )
            str_fns = tuple(answer.split(", "))
            if direction in ["back", "flip"]:
                str_fns = str_fns[::-1]
                if direction == "flip":
                    str_fns = tuple(
                        s.replace("shift_left", "shift_left_left") for s in str_fns
                    )
                    str_fns = tuple(
                        s.replace("shift_right", "shift_right_right") for s in str_fns
                    )
                    str_fns = tuple(
                        s.replace("shift_left_left", "shift_right") for s in str_fns
                    )
                    str_fns = tuple(
                        s.replace("shift_right_right", "shift_left") for s in str_fns
                    )
            prev_tries.append(str_fns)
            candidates_dir.append([str_fns, direction, "incorrect"])

            # Self check - skip if already checked before
            if str_fns not in list(prev_tries)[:-1]:
                query = (
                    verify_prompt_fixed
                    + "\n***** Current problem:\n"
                    + problem
                    + f"\nFunctions: {'[' + ', '.join(str_fns) + ']'}"
                    + "\nPlease verify initial to final steps with the exactly same format. Do not repeat the problem."
                )
                response_verify, _ = call_lm(
                    query,
                    model=verify_model,
                    max_tokens=max_token_verify,
                    temperature=0,
                )
                verbose and logging.info(f"{'*' * 20} VERIFY {'*' * 20}")
                verbose and logging.info(query)
                verbose and logging.info(response_verify)
                if "Correct" in response_verify:
                    candidates_dir[-1][-1] = "correct"
                    break

            # break if all in prev_try are the same
            if len(prev_tries) == prev_try_maxlen and len(set(prev_tries)) == 1:
                break

        logging.info(f"Attempts: {prev_tries}")

        # Check with ground truth - use last response
        response_fns = str_to_fns(response, flip=direction == "flip")
        if direction in ["back", "flip"]:
            response_fns = response_fns[::-1]
        vars = vars_init
        for fn in response_fns:
            vars = fn(vars)
        vars_response = vars
        verbose and logging.info(f"{'*' * 20} TRUE ANSWER {'*' * 20}")
        verbose and logging.info(fns)
        verbose and logging.info(f"{'*' * 20} RESPONSE {'*' * 20}")
        verbose and logging.info(response)
        verbose and logging.info(f"{'*' * 20} EXTRACTED FNS {'*' * 20}")
        verbose and logging.info(str_fns)
        logging.info(f"RESPONSE VARS: {vars_response}")
        logging.info(f"TRUE VARS: {vars_final}")
        logging.info(f"SUCCESS: {vars_response == vars_final}")

        # Save
        verify_successes.append(vars_response == vars_final)
        candidates_all.append(candidates_dir)

        # save results
        with open(args.result_path, "wb") as f:
            pickle.dump(
                {
                    "all_num_computations_fwd": all_num_computations_fwd,
                    "all_num_computations_back": all_num_computations_back,
                    "verify_successes": verify_successes,
                    "candidates": candidates_all,
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
        "--model_verify_name", type=str, default="gpt-3.5-turbo-instruct-0914"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="fwd_flip_verify",
        help="one of {fwd_verify, back_verify, flip_verify, fwd_flip_verify, fwd_back_verify}",
    )
    parser.add_argument(
        "--functions",
        type=str,
        default="shift_repeat_cut",
        help="one of {shift_repeat_cut, shift_reverse_swap, repeat_cut_reverse_swap}",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    save_name = f"{args.setting}_{args.functions}_{args.num_runs}_{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}.pkl"
    log_dir = os.environ.get("BACKWARD_LOG_DIR")
    result_path = os.path.join(log_dir, "pcfg", save_name)
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
