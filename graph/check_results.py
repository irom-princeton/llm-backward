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
        all_fwd_graph_solns = data["all_fwd_graph_solns"]
        candidates_all = data["candidates"]
        verify_sols = data["verify_sols"]
        raw_verify_sols = data["raw_verify_sols"]
        # graphs = data["graphs"]
    verify_success_rate = np.mean(verify_successes)
    num_data = len(verify_successes)
    fwd_sol_in_solutions_all = []
    back_sol_in_solutions_all = []
    flip_sol_in_solutions_all = []
    no_sol_all = []
    verify_wrong_all = []
    verify_wrong_fwd_all = []
    verify_wrong_back_all = []
    verify_wrong_flip_all = []
    # lens = []
    # verify_sol_dir_wrong_back = []
    oracle = []
    # wrong_len = []
    for i in range(num_data):
        solutions = all_fwd_graph_solns[i]
        candidates = [s[0] for s in candidates_all[i]]
        candidates_dir = [s[1] for s in candidates_all[i]]
        verify_sol = verify_sols[i]
        verify_sol_dir = candidates_dir[candidates.index(verify_sol)]

        # labels = ["A", "B", "C", "D"]
        # verify_sol = sols[labels.index(verify_sol)]
        # verify_sol_dir = sols_dir[labels.index(verify_sol)]

        # check if any sol is in solutions
        fwd_sol_in_solutions = False
        back_sol_in_solutions = False
        flip_sol_in_solutions = False
        for sol, sol_dir in zip(candidates, candidates_dir):
            if sol in solutions and sol_dir == "fwd":
                fwd_sol_in_solutions = True
            if sol in solutions and sol_dir == "back":
                back_sol_in_solutions = True
            if sol in solutions and sol_dir == "flip":
                flip_sol_in_solutions = True
            if sol_dir not in ["fwd", "back", "flip"]:
                print("sol_dir not in [fwd, back, flip]:", sol_dir)
        fwd_sol_in_solutions_all.append(fwd_sol_in_solutions)
        back_sol_in_solutions_all.append(back_sol_in_solutions)
        flip_sol_in_solutions_all.append(flip_sol_in_solutions)
        oracle.append(
            fwd_sol_in_solutions or back_sol_in_solutions or flip_sol_in_solutions
        )

        # failure modes: no solution in sols, verify_sol not in solutions (bad verification)
        no_sol = (
            not fwd_sol_in_solutions
            and not back_sol_in_solutions
            and not flip_sol_in_solutions
        )
        no_sol_all.append(no_sol)
        verify_wrong = (
            fwd_sol_in_solutions or back_sol_in_solutions or flip_sol_in_solutions
        ) and verify_sol not in solutions
        verify_wrong_all.append(verify_wrong)

        verify_wrong_fwd = verify_wrong and fwd_sol_in_solutions
        verify_wrong_fwd_all.append(verify_wrong_fwd)
        verify_wrong_back = verify_wrong and back_sol_in_solutions
        verify_wrong_back_all.append(verify_wrong_back)
        verify_wrong_flip = verify_wrong and flip_sol_in_solutions
        verify_wrong_flip_all.append(verify_wrong_flip)

        # if verify_wrong_fwd:
        #     print(solutions, verify_sol, verify_sol_dir, candidates)
        #     print(raw_verify_sols[i])
        #     print()
        #     wrong_len += [len(s) for s in solutions]
        #     verify_sol_dir_wrong_back.append(verify_sol_dir == "back")
    # print(np.mean(lens))
    # print(np.mean(verify_sol_dir_wrong_back))
    print("verify_success_rate:", verify_success_rate)
    print("fwd_sol_in_solutions_all:", np.mean(fwd_sol_in_solutions_all))
    print("back_sol_in_solutions_all:", np.mean(back_sol_in_solutions_all))
    print("flip_sol_in_solutions_all:", np.mean(flip_sol_in_solutions_all))
    print("No solution in candidates:", np.mean(no_sol_all))
    print("Wrong verification:", np.mean(verify_wrong_all))
    print("Wrong verification fwd:", np.mean(verify_wrong_fwd_all))
    print("Wrong verification back:", np.mean(verify_wrong_back_all))
    print("Wrong verification flip:", np.mean(verify_wrong_flip_all))
    print("Oracle:", np.mean(oracle))


if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    args = parser.parse_args()
    main(args)
