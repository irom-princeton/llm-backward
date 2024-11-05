import pickle
import numpy as np


def main(args):
    with open(args.load_path, "rb") as f:
        data = pickle.load(f)
        # all_num_computations_fwd = data["all_num_computations_fwd"]
        # all_num_computations_back = data["all_num_computations_back"]
        # init_goal_ratios = data["init_goal_ratios"]
        verify_successes = data["verify_successes"]
        # rev_fwd_successes = data['rev_fwd_successes']
        # rev_back_successes = data['rev_back_successes']
        # num_init_nodes = data["num_init_nodes"]
        # num_goal_nodes = data["num_goal_nodes"]
        # all_fwd_graph_solns = data["all_fwd_graph_solns"]
        candidates_all = data["candidates"]
        graphs_all = data["graphs"]
        # verify_sols = data["verify_sols"]
        # raw_verify_sols = data["raw_verify_sols"]

    node_init_all = [g["node_init"] for g in graphs_all]
    node_goal_all = [g["node_goal"] for g in graphs_all]
    print("avg node init len:", np.mean([len(n) for n in node_init_all]))
    print("avg node goal len:", np.mean([len(n) for n in node_goal_all]))

    fwd_cnt = 0
    fwd_func_cnt = {
        "shift_left": 0,
        "shift_right": 0,
        "reverse": 0,
        "swap": 0,
        "repeat": 0,
        "cut": 0,
    }
    back_func_cnt = {
        "shift_left": 0,
        "shift_right": 0,
        "reverse": 0,
        "swap": 0,
        "repeat": 0,
        "cut": 0,
    }
    back_cnt = 0
    for candidates, success in zip(candidates_all, verify_successes):
        if len(candidates) == 0:
            print("No candidates!")
            continue
        chosen = candidates[-1]
        if success:
            if chosen[1] == "fwd":
                fwd_cnt += 1
                for func in chosen[0]:
                    fwd_func_cnt[func] += 1
            else:
                back_cnt += 1
                for func in chosen[0]:
                    back_func_cnt[func] += 1

    # normalize
    fwd_total_cnt = sum(fwd_func_cnt.values())
    back_total_cnt = sum(back_func_cnt.values())
    if fwd_total_cnt > 0:
        for key in fwd_func_cnt:
            fwd_func_cnt[key] /= fwd_total_cnt
    if back_total_cnt > 0:
        for key in back_func_cnt:
            back_func_cnt[key] /= back_total_cnt

    # fwd_success_rate = np.mean(fwd_successes)
    # back_success_rate = np.mean(back_successes)
    verify_success_rate = np.mean(verify_successes)
    # print("fwd_success_rate:", fwd_success_rate)
    # print("back_success_rate:", back_success_rate)
    print("verify_success_rate:", verify_success_rate)
    print("fwd_cnt:", fwd_cnt)
    print("back_cnt:", back_cnt)
    print("fwd_func_cnt:", fwd_func_cnt)
    print("back_func_cnt:", back_func_cnt)


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    args = parser.parse_args()
    main(args)
