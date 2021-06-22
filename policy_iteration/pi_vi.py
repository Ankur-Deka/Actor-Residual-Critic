# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
import os, sys
# import hw3q1.lake_envs as lake_env
import lake_envs as lake_env

def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.
    policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
      q = np.zeros(env.nA)
      for a in range(env.nA):
        _, s_next, r, is_terminal = env.P[s][a][0]
        q[a] = r + gamma*value_func[s_next]*(1-is_terminal)
      policy[s] = np.argmax(q)
    return policy


def evaluate_policy_sync(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    for i in range(max_iterations):
      value_next = np.zeros_like(value_func)
      for s in range(env.nS):
        a = policy[s]
        _, s_next, r, is_terminal = env.P[s][a][0]
        value_next[s] = r + gamma*value_func[s_next]*(1-is_terminal)
      delta = np.max(np.abs(value_next-value_func))
      value_func = value_next
      if(delta<tol):
        break
    return value_func, (i+1)


def evaluate_policy_sync_q(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    for i in range(max_iterations):
      value_next = np.zeros_like(value_func)
      for s in range(env.nS):
        for a in range(env.nA):
          _, s_next, r, is_terminal = env.P[s][a][0]
          a_next = policy[s_next]
          value_next[s,a] = r + gamma*value_func[s_next,a_next]*(1-is_terminal)
      delta = np.max(np.abs(value_next-value_func))
      value_func = value_next
      if(delta<tol):
        break
    return value_func, (i+1)


def evaluate_policy_sync_c(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    for i in range(max_iterations):
      value_next = np.zeros_like(value_func)
      for s in range(env.nS):
        for a in range(env.nA):
          _, s_next, r, is_terminal = env.P[s][a][0]
          
          a_next = policy[s_next]
          _, _, r_next, _ = env.P[s_next][a_next][0]
          value_next[s,a] = gamma*(1-is_terminal)*(r_next + value_func[s_next,a_next])


      delta = np.max(np.abs(value_next-value_func))
      value_func = value_next
      if(delta<tol):
        break
    return value_func, (i+1)

def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    for i in range(max_iterations):
      delta = 0
      for s in range(env.nS):
        a = policy[s]
        _, s_next, r, is_terminal = env.P[s][a][0]
        val = r + gamma*value_func[s_next]*(1-is_terminal)
        delta = np.max([delta, np.abs(val-value_func[s])])
        value_func[s] = val
      if(delta<tol):
        break
    return value_func, (i+1)

def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    for i in range(max_iterations):
      delta = 0
      seq = np.arange(env.nS)
      np.random.shuffle(seq)
      for s in seq:
        a = policy[s]
        _, s_next, r, is_terminal = env.P[s][a][0]
        val = r + gamma*value_func[s_next]*(1-is_terminal)
        delta = np.max([delta, np.abs(val-value_func[s])])
        value_func[s] = val
      if(delta<tol):
        break
    return value_func, (i+1)

def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    old_policy = policy
    policy = np.zeros_like(policy)
    for s in range(env.nS):
      q = np.zeros(env.nA)
      for a in range(env.nA):
        _, s_next, r, is_terminal = env.P[s][a][0]
        q[a] = r + gamma*value_func[s_next]*(1-is_terminal)
      policy[s] = np.argmax(q)

    return np.any(policy!=old_policy), policy


def improve_policy_q(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    new_policy = np.argmax(value_func, axis=1)

    return np.any(new_policy!=policy), new_policy

def improve_policy_c(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    rew = np.zeros_like(value_func)
    for s in range(env.nS):
      for a in range(env.nA):
        _, s_next, r, is_terminal = env.P[s][a][0]
        rew[s,a] = r
    new_policy = np.argmax(rew+value_func, axis=1)

    return np.any(new_policy!=policy), new_policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    improv_steps, eval_steps = 0, 0

    for i in range(max_iterations):
      value_func, c = evaluate_policy_sync(env, value_func, gamma, policy, max_iterations=max_iterations, tol=tol)
      eval_steps += c

      changed, policy = improve_policy(env, gamma, value_func, policy)
      improv_steps +=1 
      if not changed:
        break
    return policy, value_func, improv_steps, eval_steps

def policy_iteration_sync_q(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros((env.nS, env.nA))
    improv_steps, eval_steps = 0, 0

    for i in range(max_iterations):
      value_func, c = evaluate_policy_sync_q(env, value_func, gamma, policy, max_iterations=max_iterations, tol=tol)
      eval_steps += c

      changed, policy = improve_policy_q(env, gamma, value_func, policy)
      improv_steps +=1 
      if not changed:
        break
    return policy, value_func, improv_steps, eval_steps


def policy_iteration_sync_c(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros((env.nS, env.nA))
    improv_steps, eval_steps = 0, 0

    for i in range(max_iterations):
      value_func, c = evaluate_policy_sync_c(env, value_func, gamma, policy, max_iterations=max_iterations, tol=tol)
      eval_steps += c

      changed, policy = improve_policy_c(env, gamma, value_func, policy)
      improv_steps +=1 
      if not changed:
        break
    return policy, value_func, improv_steps, eval_steps



def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    improv_steps, eval_steps = 0, 0

    for i in range(max_iterations):
      value_func, c = evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations=max_iterations, tol=tol)
      eval_steps += c

      changed, policy = improve_policy(env, gamma, value_func, policy)
      improv_steps +=1 
      if not changed:
        break
    return policy, value_func, improv_steps, eval_steps


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    improv_steps, eval_steps = 0, 0

    for i in range(max_iterations):
      value_func, c = evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations=max_iterations, tol=tol)
      eval_steps += c

      changed, policy = improve_policy(env, gamma, value_func, policy)
      improv_steps +=1 
      if not changed:
        break
    return policy, value_func, improv_steps, eval_steps

def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    for i in range(max_iterations):
      value_next = np.zeros_like(value_func)
      for s in range(env.nS):
        q = np.zeros(env.nA)
        for a in range(env.nA):
          _, s_next, r, is_terminal = env.P[s][a][0]
          q[a] = r + gamma*value_func[s_next]*(1-is_terminal)
        value_next[s] = np.max(q)
      delta = np.max(np.abs(value_next-value_func))
      value_func = value_next
      if(delta<tol):
        break
    return value_func, i+1


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    for i in range(max_iterations):
      delta = 0
      for s in range(env.nS):
        q = np.zeros(env.nA)
        for a in range(env.nA):
          _, s_next, r, is_terminal = env.P[s][a][0]
          q[a] = r + gamma*value_func[s_next]*(1-is_terminal)
        val = np.max(q)
        delta = np.max([delta,np.abs(val-value_func[s])])
        value_func[s] = val
      if(delta<tol):
        break
    return value_func, i+1


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    for i in range(max_iterations):
      delta = 0
      seq = np.arange(env.nS)
      np.random.shuffle(seq)
      for s in seq:
        q = np.zeros(env.nA)
        for a in range(env.nA):
          _, s_next, r, is_terminal = env.P[s][a][0]
          q[a] = r + gamma*value_func[s_next]*(1-is_terminal)
        val = np.max(q)
        delta = np.max([delta,np.abs(val-value_func[s])])
        value_func[s] = val
      if(delta<tol):
        break
    return value_func, i+1


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    goal_map = env.desc==b'G'
    goal_loc = np.argwhere(goal_map)[0]
    X, Y = np.meshgrid(np.arange(env.nrow), np.arange(env.ncol))
    dists = np.abs(Y-goal_loc[0]) + np.abs(X-goal_loc[1])
    seq = np.argsort(dists.flatten())
    for i in range(max_iterations):
      delta = 0
      for s in seq:
        q = np.zeros(env.nA)
        for a in range(env.nA):
          _, s_next, r, is_terminal = env.P[s][a][0]
          q[a] = r + gamma*value_func[s_next]*(1-is_terminal)
        val = np.max(q)
        delta = np.max([delta,np.abs(val-value_func[s])])
        value_func[s] = val
      if(delta<tol):
        break
    return value_func, i+1


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy, name=None):
    """Displays a policy as letters, as required by problem 2.2 & 2.6

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)

    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))

    if not name is None:
        mystr = '{}\n{}\n'.format('-'*30,name)
        for row in range(env.nrow):
          mystr+='&\\texttt{'
          for col in range(env.ncol):
            mystr += policy_letters[row, col]
          mystr += '}\\\\ \n'
        mystr += '\n'
        if not os.path.exists('Result'):
          os.makedirs('Result')
        with open('Result/summary.txt', 'a') as file:
          file.write(mystr)

def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))

    for state in range(env.nS):
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func, name=None):
    """Visualize a policy as a heatmap, as required by problem 2.3 & 2.5

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                # yticklabels = np.arange(1, env.nrow+1)[::-1],
                # xticklabels = np.arange(1, env.nrow+1))
                yticklabels = np.arange(env.nrow),
                xticklabels = np.arange(env.nrow))
    plt.title('Value function')
    plt.tight_layout()
    if name is None:
      plt.show()
    else:
      if not os.path.exists('Result'):
        os.makedirs('Result')
      plt.savefig('Result/{}'.format(name))
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None

def save_stats_policy(improv_steps, eval_steps, name):
    mystr = '{}\nImprov steps: {}, Eval steps: {}\n\n'.format(name, improv_steps, eval_steps)
    print(mystr)
    with open('Result/summary.txt', 'a') as file:
      file.write(mystr)

def save_stats_value(steps, name):
    mystr = '{}\nValue iteration steps: {}\n\n'.format(name, steps)
    print(mystr)
    with open('Result/summary.txt', 'a') as file:
      file.write(mystr)    

if __name__ == "__main__":
    envs = ['Deterministic-4x4-FrozenLake-v0']#, 'Deterministic-8x8-FrozenLake-v0']
    # Define num_trials, gamma and whatever variables you need below.
    
    # Problem 1.2
    print('-'*20)
    print('Problem 1.2')
    for env_name in envs:
      env = gym.make(env_name)
      policy, value_func, improv_steps, eval_steps = policy_iteration_sync_q(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)

      print(env_name)
      print('Environment')
      env.render()

      print('Optimal policy')
      display_policy_letters(env, policy, name='q1_2_{}_policy.txt'.format(env_name))

      # print('Value function')
      # rew = np.zeros_like(value_func)
      # for s in range(env.nS):
      #   for a in range(env.nA):
      #     _, s_next, r, is_terminal = env.P[s][a][0]
      #     rew[s,a] = r
      # policy = np.argmax(rew+value_func, axis=1)
      # print(policy.reshape((4,4)))
      
      # c = value_func[np.arange(16), policy]
      # print(c.reshape((4,4)))
      print(improv_steps, eval_steps)
      
      # value_func_heatmap(env, valuealue_func, name='q1_2_{}_value.png'.format(env_name))

      # save_stats_policy(improv_steps, eval_steps, 'q1_2_{}_iterations'.format(env_name))

    # # Problem 1.3
    # print('-'*20)
    # print('Problem 1.3')
    # for env_name in envs:
    #   env = gym.make(env_name)
    #   value_func, steps = value_iteration_sync(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
    #   policy = value_function_to_policy(env, 0.9, value_func)

    #   print(env_name)
    #   print('Environment')
    #   env.render()

    #   print('Optimal policy')
    #   display_policy_letters(env, policy, name='q1_3_{}_policy.txt'.format(env_name))

    #   print('Value function')
    #   value_func_heatmap(env, value_func, name='q1_3_{}_value.png'.format(env_name))

    #   save_stats_value(steps, name='q1_3_{}_iterations'.format(env_name))

    # # Problem 1.4(a)- async policy iteration ordered
    # print('-'*20)
    # print('Problem 1.4(a)')
    # for env_name in envs:
    #   env = gym.make(env_name)
    #   policy, value_func, improv_steps, eval_steps = policy_iteration_async_ordered(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)

    #   print(env_name)
    #   print('Environment')
    #   env.render()

    #   print('Optimal policy')
    #   display_policy_letters(env, policy, name='q1_4_{}_policy.txt'.format(env_name))

    #   print('Value function')
    #   value_func_heatmap(env, value_func, name='q1_4_{}_value.png'.format(env_name))

    #   save_stats_policy(improv_steps, eval_steps, 'q1_4_{}_iterations'.format(env_name))


    # # Problem 1.4(b) - async policy iteration random perm
    # print('-'*20)
    # print('Problem 1.4(b)')

    # for env_name in envs:
    #   env = gym.make(env_name)
    #   avg_improv_steps, avg_eval_steps = 0, 0
    #   for i in range(10):
    #     policy, value_func, improv_steps, eval_steps = policy_iteration_async_randperm(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
    #     avg_improv_steps += improv_steps
    #     avg_eval_steps += eval_steps

    #   avg_improv_steps /= 10
    #   avg_eval_steps /= 10

    #   print(env_name)
    #   print('Environment')
    #   env.render()

    #   print('Optimal policy')
    #   display_policy_letters(env, policy, name='q1_4_rand_{}_policy.txt'.format(env_name))

    #   print('Value function')
    #   value_func_heatmap(env, value_func, name='q1_4_rand_{}_value.png'.format(env_name))

    #   save_stats_policy(avg_improv_steps, avg_eval_steps, 'q1_4_rand_{}_iterations'.format(env_name))

    # # Problem 1.5.1 (a)
    # print('-'*20)
    # print('Problem 1.5.1(a)')
    # for env_name in envs:
    #   env = gym.make(env_name)
    #   value_func, steps = value_iteration_async_ordered(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
    #   policy = value_function_to_policy(env, 0.9, value_func)

    #   print(env_name)
    #   print('Environment')
    #   env.render()

    #   print('Optimal policy')
    #   display_policy_letters(env, policy, name='q1_5_1_{}_policy.txt'.format(env_name))

    #   print('Value function')
    #   value_func_heatmap(env, value_func, name='q1_5_1_{}_value.png'.format(env_name))

    #   save_stats_value(steps, name='q1_5_1_{}_iterations'.format(env_name))


    # # Problem 1.5.1 (b)
    # print('-'*20)
    # print('Problem 1.5.1(b)')
    # for env_name in envs:
    #   env = gym.make(env_name)
    #   avg_steps = 0
    #   for i in range(10):
    #     value_func, steps = value_iteration_async_randperm(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
    #     policy = value_function_to_policy(env, 0.9, value_func)
    #     avg_steps += steps
    #   avg_steps /= 10

    #   print(env_name)
    #   print('Environment')
    #   env.render()

    #   print('Optimal policy')
    #   display_policy_letters(env, policy, name='q1_5_1_{}_policy.txt'.format(env_name))

    #   print('Value function')
    #   value_func_heatmap(env, value_func, name='q1_5_1_{}_value.png'.format(env_name))

    #   save_stats_value(avg_steps, name='q1_5_1_{}_iterations'.format(env_name))

    # # # Problem 1.5.2
    # print('-'*20)
    # print('Problem 1.5.2')
    # for env_name in envs:
    #   env = gym.make(env_name)
    #   value_func, steps = value_iteration_async_custom(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
    #   policy = value_function_to_policy(env, 0.9, value_func)

    #   print(env_name)
    #   print('Environment')
    #   env.render()

    #   print('Optimal policy')
    #   display_policy_letters(env, policy, name='q1_5_2_{}_policy.txt'.format(env_name))

    #   print('Value function')
    #   value_func_heatmap(env, value_func, name='q1_5_2_{}_value.png'.format(env_name))

    #   save_stats_value(steps, name='q1_5_2_{}_iterations'.format(env_name))