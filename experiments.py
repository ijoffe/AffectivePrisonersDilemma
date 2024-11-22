# Isaac Joffe (2024)


# basic built-in libraries required
from tqdm import tqdm
import matplotlib.pyplot as plt
# other code developed for project
from tournament import *


# experiments to test arousal parameter
def arousal_experiment():
    a_vals = [
        0.5,
        0.75,
        1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25, 
        3.5,
        3.75,
        4,
        4.25,
        4.5, 
        4.75,
        5,
    ]
    agents = [PicardAgent(f"Arousal={a:.1f}", {"a": a, "b": 1, "e": 0}) for a in a_vals]
    return a_vals, agents


# experiments to test temperament parameter
def temperament_experiment():
    b_vals = [
        0.5,
        0.75,
        1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25,
        3.5,
        3.75,
        4,
        4.25,
        4.5,
        4.75,
        5,
    ]
    agents = [PicardAgent(f"Temperament={b:.1f}", {"a": 1, "b": b, "e": 0}) for b in b_vals]
    return b_vals, agents


# experiments to test disposition parameter
def disposition_experiment():
    e_vals = [
        -1,
        -0.9,
        -0.8, 
        -0.7,
        -0.6,
        -0.5,
        -0.45,
        -0.4,
        -0.35, 
        -0.3,
        -0.25,
        -0.2,
        -0.15,
        -0.1,
        -0.05,
        0, 
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ]
    agents = [PicardAgent(f"Disposition={e:.1f}", {"a": 1, "b": 1, "e": e}) for e in e_vals]
    return e_vals, agents


# run experiment that isolates performance by parameter
def run_parameter_experiment():
    # set general experiment parameters
    n_iterations = 200
    reward_matrix = [0, 1, 3, 5]
    n_trials = 100
    n_err = 10

    # set experiment type to be carried out
    mode = "arousal"
    # mode = "temperament"
    # mode = "disposition"

    # set up agents
    if mode == "arousal":
        vals, agents = arousal_experiment()
    elif mode == "temperament":
        vals, agents = temperament_experiment()
    elif mode == "disposition":
        vals, agents = disposition_experiment()

    # run experiment
    totals = np.zeros((n_err, len(agents)))
    averages = np.zeros(len(agents))
    maxs = np.zeros(len(agents))
    mins = np.zeros(len(agents)) + reward_matrix[-1] * n_iterations
    for k in tqdm(range(n_err)):
        for i in tqdm(range(n_trials)):
            tournament = Tournament(
                n_iterations,
                reward_matrix,
                agents,
            )
            totals[k] = totals[k] + np.array(tournament.play()) / (len(agents) + 1) / n_trials
        averages = averages + totals[k] / n_err
        for j in range(len(agents)):
            if totals[k][j] > maxs[j]:
                maxs[j] = totals[k][j]
            if totals[k][j] < mins[j]:
                mins[j] = totals[k][j]
    # display and plot results
    print(totals)
    [print(f"${vals[i]}$ & ${averages[i]:.2f}$ \\\\") for i in range(len(vals))]
    combined = [(vals[i], averages[i]) for i in range(len(agents))]
    combined.sort(key=(lambda x: -x[1]))
    n_rows = int(len(combined) / 3) + (1 if (len(combined) % 3) else 0)
    [print(f"$({combined[i%n_agents][0]}$) & ${combined[i%n_agents][1]:.2f}$ & $({combined[(i+n_rows)%n_agents][0]})$ & ${combined[(i+n_rows)%n_agents][1]:.2f}$ & $({combined[(i+2*n_rows)%n_agents][0]})$ & ${combined[(i+2*n_rows)%n_agents][1]:.2f}$ \\\\") for i in range(n_rows)]
    errs = maxs - mins
    plt.errorbar(vals, averages, yerr=errs, fmt="b--o", ecolor="black", capsize=3)
    plt.ylabel("Average Agent Performance")
    plt.xlim([np.min(vals), np.max(vals)])

    # format results properly
    if mode == "arousal":
        plt.title("Average Agent Performance with Varying Arousal")
        plt.xlabel("Agent Arousal, $a$")
    elif mode == "temperament":
        plt.title("Average Agent Performance with Varying Temperament")
        plt.xlabel("Agent Temperament, $b$")
    elif mode == "disposition":
        plt.title("Average Agent Performance with Varying Disposition")
        plt.xlabel("Agent Disposition, $e$")
    plt.savefig("temp.png", dpi=1000)
    plt.show()
    return


# run experiment to test all parameters
def run_global_experiment():
    # set general experiment parameters
    n_iterations = 200
    reward_matrix = [0, 1, 3, 5]
    n_trials = 100
    n_err = 10

    # construct agents for large-scale tournaments
    a_vals = [0.75, 1, 1.25, 1.5, 2, 3, 4.5]
    b_vals = [0.75, 1, 1.25, 1.5, 2, 3, 4.5]
    e_vals = [-0.8, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.8]
    n_agents = 200
    agents = []
    params = np.zeros((3, n_agents))
    for i in range(n_agents):
        a = np.round(np.random.choice(a_vals) + np.random.normal(0, 0.1), 2)
        b = np.round(np.random.choice(b_vals) + np.random.normal(0, 0.1), 2)
        e = np.round((np.random.choice(e_vals) + np.random.normal(0, 0.1)) * a, 2)
        assert (a > 0.5 and a < 5) and (b > 0.5 and b < 5) and (e > -a and e < a)
        params[:,i] = [a, b, e]
        agents.append(PicardAgent(f"a={a:.2f},b={b:.2f},e={e:.2f}", {"a": a, "b": b, "e": e}))

    # run experiment
    totals = np.zeros((n_err, len(agents)))
    averages = np.zeros(len(agents))
    maxs = np.zeros(len(agents))
    mins = np.zeros(len(agents)) + reward_matrix[-1] * n_iterations
    for k in tqdm(range(n_err)):
        for i in tqdm(range(n_trials)):
            tournament = Tournament(
                n_iterations,
                reward_matrix,
                agents,
            )
            totals[k] = totals[k] + np.array(tournament.play()) / (len(agents) + 1) / n_trials
        averages = averages + totals[k] / n_err
        for j in range(len(agents)):
            if totals[k][j] > maxs[j]:
                maxs[j] = totals[k][j]
            if totals[k][j] < mins[j]:
                mins[j] = totals[k][j]

    # display and plot results
    combined = [(agents[i], averages[i]) for i in range(len(agents))]
    combined.sort(key=(lambda x: -x[1]))
    n_rows = int(len(combined) / 3) + (1 if (len(combined) % 3) else 0)
    [print(f"$({combined[i%n_agents][0]}$) & ${combined[i%n_agents][1]:.2f}$ & $({combined[(i+n_rows)%n_agents][0]})$ & ${combined[(i+n_rows)%n_agents][1]:.2f}$ & $({combined[(i+2*n_rows)%n_agents][0]})$ & ${combined[(i+2*n_rows)%n_agents][1]:.2f}$ \\\\") for i in range(n_rows)]
    print(averages)
    errs = maxs - mins

    # analyze results based on each parameter
    plt.errorbar(params[0], averages, yerr=errs, fmt="bo", ecolor="black", capsize=3)
    plt.title("Average Agent Performance with Varying Arousal")
    plt.xlabel("Agent Arousal, $a$")
    plt.ylabel("Average Agent Performance")
    plt.xlim([0.5, 5])
    plt.savefig("temp1.png", dpi=1000)
    plt.show()
    plt.errorbar(params[1], averages, yerr=errs, fmt="bo", ecolor="black", capsize=3)
    plt.title("Average Agent Performance with Varying Temperament")
    plt.xlabel("Agent Temperament, $b$")
    plt.ylabel("Average Agent Performance")
    plt.xlim([0.5, 5])
    plt.savefig("temp2.png", dpi=1000)
    plt.show()
    plt.errorbar(params[2] / params[0], averages, yerr=errs, fmt="bo", ecolor="black", capsize=3)
    plt.title("Average Agent Performance with Varying Disposition")
    plt.xlabel("Agent Disposition, $e$")
    plt.ylabel("Average Agent Performance")
    plt.xlim([-1, 1])
    plt.savefig("temp3.png", dpi=1000)
    plt.show()
    return


# run the desired experiment
def main():
    # run_parameter_experiment()
    run_global_experiment()
    return


# run main function if called as program
if __name__ == "__main__":
    main()
