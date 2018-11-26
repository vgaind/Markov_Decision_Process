"""
Solving FrozenLake8x8 environment using Value-Itertion.


Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

def run_episode_ql(env,t_max=10000, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy[obs]
        obs, reward, done, _ = env.step(action)
        #reward=reward*100-1
        total_reward += reward
        #total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def run_episode(env, policy, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.

    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.

    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += reward#(gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy,   n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy,  render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                #q_sa[a] += (p * (r + gamma * v[s_]))
                q_sa[a] += ((r + p*gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.env.nS)
    eps = 1e-10
    value_diff=[]

    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            #v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
            v[s] = sum([ (r + p*gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        value_diff_curr=np.sum((np.fabs(prev_v - v)))
        value_diff.append(value_diff_curr)
        if (value_diff_curr <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # initialize a random policy
    max_iterations = 200000

    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy, i+1




def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    value_diff=np.zeros(max_iterations)
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            #q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            q_sa = [sum([(r+ gamma*p*prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)
        value_diff[i]=np.sum(np.fabs(prev_v - v))
        if (value_diff[i] <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v,value_diff[0:i]



def q_learning(env,env_name='Taxi-v2',random_q=True,gamma=1.0):
    np.random.seed(0)
    initial_lr = 1.0  # Learning rate
    min_lr = 0.003
    iter_max = 100000
    n_states=env.env.nS
    initial_eps=0.95
    t_max = 100000
    min_eps = 0.1
    print ('----- using Q Learning -----')
    q_table = np.zeros((n_states, env.action_space.n))
    if random_q:
        if env_name=='FrozenLake8x8-v0':
            q_table=np.random.rand(q_table.shape[0],q_table.shape[1])
        else:
            q_table = np.random.randn(q_table.shape[0], q_table.shape[1])

    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eps= max(min_eps,initial_eps * (0.85 ** (i // 10000)))
        eta = max(min_lr, initial_lr * (0.85 ** (i // 10000)))
        for j in range(t_max):

            if np.random.uniform(0, 1) < eps:
                action = env.action_space.sample()  #np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[obs])
            obs_next, reward, done, _ = env.step(action)
            #reward=reward*100-1
            total_reward += reward
            # update q table

            q_table[obs][action] = q_table[obs][action] + eta * (
                        reward + gamma * np.max(q_table[obs_next]) - q_table[obs][action])
            obs=obs_next
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))
    solution_policy = np.argmax(q_table, axis=1)
    solution_policy_scores = [run_episode_ql(env=env,t_max=t_max, policy=solution_policy, render=False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    return np.mean(solution_policy_scores)

if __name__ == '__main__':
    env_name  = 'Taxi-v2'#'FrozenLake8x8-v0'#'Taxi-v2'#'FrozenLake8x8-v0'
    value_policy_q=2 #0 value, 1 policy, 2 q
    env = gym.make(env_name)
    gamma_counter=0
    gamma_array=np.arange(0.1,1.,0.2)
    gamma_array=np.round(gamma_array*100)/100.
    policy_score_array=np.zeros(gamma_array.shape)
    iter_step_list=[]
    fig0,ax0=plt.subplots()
    fig1,ax1=plt.subplots()
    for gamma in gamma_array:
        print(gamma)
        if value_policy_q==0:
            optimal_v,value_diff = value_iteration(env, gamma)
            policy = extract_policy(optimal_v, gamma)
            ax0.semilogy(value_diff, label=str(gamma))
        elif value_policy_q==1:
            policy, iter_step = policy_iteration(env, gamma)
            iter_step_list.append(iter_step)
        elif value_policy_q==2:
            policy_score=q_learning(env,env_name=env_name,random_q=True,gamma=gamma)

        if value_policy_q<=1:
            policy_score = evaluate_policy(env, policy,  n=1000)
        policy_score_array[gamma_counter]=policy_score
        print('Policy average score = ', policy_score)
        gamma_counter+=1


    ax1.plot(gamma_array,policy_score_array)
    ax1.set_xlabel('gamma')
    ax1.set_ylabel('Average Reward')

    if value_policy_q==0:
        ax0.set_xlabel('Number of iterations')
        ax0.set_ylabel('Value difference')
        ax0.legend()
        fig0.savefig(env_name + 'value_iteration_error.png')
        fig1.savefig(env_name + 'value_iteration_reward.png')
    elif value_policy_q==1:
        ax0.plot(gamma_array, np.array(iter_step_list))
        ax0.set_xlabel('gamma')
        ax0.set_ylabel('number policy iterations')

        fig0.savefig(env_name + 'policy_iteration_convergence.png')
        fig1.savefig(env_name + 'policy_iteration_reward.png')
    elif value_policy_q==2:
        fig1.savefig(env_name + 'q_learning_iteration_reward.png')
    print('Done')