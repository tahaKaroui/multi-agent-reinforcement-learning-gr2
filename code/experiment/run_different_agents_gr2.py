import numpy as np
import argparse

from maci.learners import MAVBAC, MASQL
from maci.misc.sampler import MASampler
from maci.environments import PBeautyGame, MatrixGame
from maci.environments import make_particle_env
from maci.misc import logger
import gtimer as gt
import datetime
from copy import deepcopy
from maci.get_agents import ddpg_agent, masql_agent, pr2ac_agent

import maci.misc.tf_utils as U
import os

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)


def add_agents(names, desired_number):
    """
    Adds as many agents to each cluster as the desired_number (another word for number of agents)
    Parameters:
        names: string coming from the arg list designing the names of the agents of each cluster separated by '-'
            between each adjacent clusters and '_' between agents in that cluster.
        desired_number: total number of agents desired

    """
    def add_final(clusters, cluster, agent):
        cluster.append(agent)
        clusters.append(cluster)
        cluster = []
        agent = ""
        return agent, cluster, clusters

    if names:
        _agent = ""
        _clusters = []
        _cluster = []
        for j, i in enumerate(names):
            if i == '_':
                _cluster.append(_agent)
                _agent = ""
            elif i == '-':
                _agent, _cluster, _clusters = add_final(_clusters, _cluster, _agent)
            else:
                _agent += i
                if j == len(names) - 1:
                    _agent, _cluster, _clusters = add_final(_clusters, _cluster, _agent)
        agents_per_cluster = desired_number // len(_clusters)
        if len(_clusters) == 1:
            type_of_agent_num = agents_per_cluster
        else:
            _new_clusters = []
            for _cluster in _clusters:
                type_of_agent_num = agents_per_cluster // len(_cluster)
                _new_cluster = []
                for _agent in _cluster:
                    _new_cluster = _new_cluster + [_agent for _ in range(type_of_agent_num)]
                if len(_new_cluster) < agents_per_cluster:
                    while len(_new_cluster) < agents_per_cluster:
                        _new_cluster.append(_new_cluster[-1])
                _new_cluster = '_'.join(_new_cluster)
                _new_clusters.append(_new_cluster)
        return '-'.join(_new_clusters)


def get_particle_game(particle_game_name, arglist):
    env = make_particle_env(game_name=particle_game_name)
    agent_num = env.n
    adv_agent_num = 0
    if particle_game_name == 'simple_push' or particle_game_name == 'simple_adversary':
        adv_agent_num = 1
    elif particle_game_name == 'simple_tag':
        adv_agent_num = 3
    model_names_setting = arglist.model_names_setting.split('_')
    model_name = '_'.join(model_names_setting)
    model_names = [model_names_setting[1]] * adv_agent_num + [model_names_setting[0]] * (agent_num - adv_agent_num)
    return env, agent_num, model_name, model_names


def parse_args():
    """Meant for arguments coming from a CLI """

    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
    # Environment
    # ['particle-simple_spread', 'particle-simple_adversary', 'particle-simple_tag', 'particle-simple_push']
    # matrix-prison , matrix-prison
    # pbeauty
    parser.add_argument('-g', "--game_name", type=str, default="pbeauty", help="name of the game")
    parser.add_argument('-p', "--p", type=float, default=2 / 3, help="p")
    parser.add_argument('-mu', "--mu", type=float, default=1.5, help="mu")
    parser.add_argument('-r', "--reward_type", type=str, default="abs", help="reward type")
    parser.add_argument('-mp', "--max_path_length", type=int, default=1, help="path length")
    parser.add_argument('-ms', "--max_steps", type=int, default=20000, help="number of epochs")
    parser.add_argument('-me', "--memory", type=int, default=0, help="memory")
    parser.add_argument('-n', "--n", type=int, default=3, help="number of agents per cluster")
    parser.add_argument('-bs', "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument('-hm', "--hidden_size", type=int, default=100, help="hidden size")
    parser.add_argument('-re', "--repeat", type=bool, default=False, help="repeat or not")
    parser.add_argument('-a', "--aux", type=bool, default=True, help="")
    parser.add_argument('-m', "--model_names_setting", type=str,
                        default='PR2AC4_PR2AC3_PR2AC2-MADDPG_MADDPG_MADDPG-DDPG_DDPG_DDPG-MASQL',
                        help="models setting agent vs adv")
    parser.add_argument('-c', "--number_of_clusters", type=int, default='3', help="number of clusters")
    return parser.parse_args()


def main(arglist):
    game_name = arglist.game_name
    # 'abs', 'one'
    reward_type = arglist.reward_type
    p = arglist.p
    clusters = len(arglist.model_names_setting.split('-')) if len(arglist.model_names_setting.split('-')) > 1 else 2
    agent_num = clusters * arglist.n
    agents_per_cluster = agent_num // clusters
    clusters_schema = dict()
    i = -1
    for _cluster in range(clusters):
        clusters_schema[_cluster] = []
        for _agent in range(agents_per_cluster):
            i += 1
            clusters_schema[_cluster].append(i)

    agent_names_per_cluster = add_agents(arglist.model_names_setting, agent_num)
    # agent_names_per_cluster = arglist.model_names_setting
    u_range = 1.
    k = 0

    model_names_setting_clusters = agent_names_per_cluster.split('-')
    model_names = [i.split('_') for i in model_names_setting_clusters]
    model_name = ""
    for j, i in enumerate(model_names):
        model_name = model_name + '_'.join(i)
        if j < len(model_names) - 1:
            model_name += '_'
    path_prefix = game_name
    if game_name == 'pbeauty':
        env = PBeautyGame(agent_num=agent_num, reward_type=reward_type, p=p)
        path_prefix = game_name + '-' + reward_type + '-' + str(p)
    elif 'matrix' in game_name:
        matrix_game_name = game_name.split('-')[-1]
        repeated = arglist.repeat
        max_step = arglist.max_path_length
        memory = arglist.memory
        env = MatrixGame(game=matrix_game_name, agent_num=agent_num,
                         action_num=2, repeated=repeated,
                         max_step=max_step, memory=memory,
                         discrete_action=False, tuple_obs=False)
        path_prefix = '{}-{}-{}-{}'.format(game_name, repeated, max_step, memory)

    elif 'particle' in game_name:
        particle_game_name = game_name.split('-')[-1]
        env, agent_num, model_name, model_names = get_particle_game(particle_game_name, arglist)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    if 'CG' in model_name:
        model_name = model_name + '-{}'.format(arglist.mu)
    if not arglist.aux:
        model_name = model_name + '-{}'.format(arglist.aux)

    suffix = '{}/{}/{}/{}'.format(path_prefix, agent_num, model_name, timestamp)


    logger.add_tabular_output('./log/{}.csv'.format(suffix[:10]))
    snapshot_dir = './snapshot/{}'.format(suffix[:10])
    policy_dir = './policy/{}'.format(suffix[:10])
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)
    logger.set_snapshot_dir(snapshot_dir)

    agents = dict()
    M = arglist.hidden_size
    # the batch size refers to the number of samples or experiences used in each iteration of training the agent.
    batch_size = arglist.batch_size
    sampler = MASampler(agent_num=agent_num, joint=True, max_path_length=30, min_pool_size=100, batch_size=batch_size)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': 1,
        'n_epochs': arglist.max_steps,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10
    }

    with U.single_threaded_session():
        i = -1
        for model in model_names:
            i += 1
            for model_name in model:
                if 'PR2AC' in model_name:
                    cluster = "level-k"
                    if cluster not in agents.keys():
                        agents[cluster] = []
                    k = int(model_name[-1])
                    g = False
                    mu = arglist.mu
                    if 'G' in model_name:
                        g = True
                    agent = pr2ac_agent(model_name, i, env, M, u_range, base_kwargs, k=k, g=g, mu=mu,
                                        game_name=game_name, aux=arglist.aux)
                elif model_name == 'MASQL':
                    cluster = "masql"
                    if cluster not in agents.keys():
                        agents[cluster] = []
                    agent = masql_agent(model_name, i, env, M, u_range, base_kwargs, game_name=game_name)
                else:
                    if model_name == 'DDPG':
                        cluster = "ddpg"
                        if cluster not in agents.keys():
                            agents[cluster] = []
                        joint = False
                        opponent_modelling = False
                    elif model_name == 'MADDPG':
                        cluster = "maddpg"
                        if cluster not in agents.keys():
                            agents[cluster] = []
                        joint = True
                        opponent_modelling = False
                    elif model_name == 'DDPG-OM' or model_name == 'DDPG-ToM':
                        cluster = "ddpg-om_ddpg-tom"
                        if cluster not in agents.keys():
                            agents[cluster] = []
                        joint = True
                        opponent_modelling = True
                    agent = ddpg_agent(joint, opponent_modelling, model_names, i, env, M, u_range, base_kwargs,
                                       game_name=game_name)

                agents[cluster].append(agent)

        sampler.initialize(env, agents)

        for cluster_agents in agents.values():
            for agent in cluster_agents:
                agent._init_training()
        gt.rename_root('MARLAlgorithm')
        gt.reset()
        gt.set_def_unique(False)
        initial_exploration_done = False
        noise = 1.
        alpha = .5

        for cluster_agents in agents.values():
            for agent in cluster_agents:
                try:
                    agent.policy.set_noise_level(noise)
                except:
                    pass

        batch_n = []
        recent_batch_n = []
        indices = None
        recent_indices = None

        for epoch in gt.timed_for(range(base_kwargs['n_epochs'] + 1)):
            logger.push_prefix('Epoch #%d | ' % epoch)
            if epoch % 1 == 0:
            for t in range(base_kwargs['epoch_length']):
                if not initial_exploration_done:
                    if epoch >= 1000:
                        initial_exploration_done = True
                sampler.sample(clusters_schema)
                if not initial_exploration_done:
                    continue
                gt.stamp('sample')
                if epoch == base_kwargs['n_epochs']:
                    noise = 0.1

                    for cluster_agents in agents.values():
                        for agent in cluster_agents:
                            try:
                                agent.policy.set_noise_level(noise)
                            except:
                                pass
                if epoch > base_kwargs['n_epochs'] / 10:
                    noise = 0.1
                    for cluster_agents in agents.values():
                        for agent in cluster_agents:
                            try:
                                agent.policy.set_noise_level(noise)
                            except:
                                pass
                if epoch > base_kwargs['n_epochs'] / 5:
                    noise = 0.05
                    for cluster_agents in agents.values():
                        for agent in cluster_agents:
                            try:
                                agent.policy.set_noise_level(noise)
                            except:
                                pass
                if epoch > base_kwargs['n_epochs'] / 6:
                    noise = 0.01
                    for cluster_agents in agents.values():
                        for agent in cluster_agents:
                            try:
                                agent.policy.set_noise_level(noise)
                            except:
                                pass

                for j in range(base_kwargs['n_train_repeat']):
                    if len(batch_n) == 0:
                        batch_n = []
                        recent_batch_n = []
                        indices = None
                        recent_indices = None
                        i = -1
                        for cluster_agents in agents.values():
                            for agent in cluster_agents:
                                i += 1
                                if i == 0:
                                    batch = agent.pool.random_batch(batch_size)
                                    indices = agent.pool.indices
                                    recent_indices = list(range(agent.pool._top - batch_size, agent.pool._top))

                                batch_n.append(agent.pool.random_batch_by_indices(indices))
                                recent_batch_n.append(agent.pool.random_batch_by_indices(recent_indices))

                    target_next_actions_n = []
                    try:
                        cluster_agents = [agent for cluster in agents.values() for agent in cluster]
                        for agent, batch in zip(cluster_agents, batch_n):
                            target_next_actions_n.append(agent._target_policy.get_actions(batch['next_observations']))
                    except:
                        pass

                    opponent_actions_n = np.array([batch['actions'] for batch in batch_n])
                    recent_opponent_actions_n = np.array([batch['actions'] for batch in recent_batch_n])

                    # _____figure out
                    recent_opponent_observations_n = []
                    for batch in recent_batch_n:
                        recent_opponent_observations_n.append(batch['observations'])
                    cluster_agents = [agent for cluster in agents.values() for agent in cluster]
                    current_actions = [cluster_agents[i]._policy.get_actions(batch_n[i]['next_observations'])[0][0] for i
                                       in range(agent_num)]
                    all_actions_k = []
                    i = -1
                    for cluster_agents in agents.values():
                        for agent in cluster_agents:
                            i += 1
                            if isinstance(agent, MAVBAC):
                                if agent._k > 0:
                                    batch_actions_k = agent._policy.get_all_actions(batch_n[i]['next_observations'])
                                    actions_k = [a[0][0] for a in batch_actions_k]
                                    all_actions_k.append(';'.join(list(map(str, actions_k))))
                    if len(all_actions_k) > 0:
                        with open('{}/all_actions.csv'.format(policy_dir), 'a') as f:
                            f.write(','.join(list(map(str, all_actions_k))) + '\n')
                    with open('{}/policy.csv'.format(policy_dir), 'a') as f:
                        f.write(','.join(list(map(str, current_actions))) + '\n')
                    i = -1
                    for cluster_agents in agents.values():
                        for agent in cluster_agents:
                            i += 1
                            try:
                                batch_n[i]['next_actions'] = deepcopy(target_next_actions_n[i])
                            except:
                                pass
                            batch_n[i]['opponent_actions'] = np.reshape(np.delete(deepcopy(opponent_actions_n), i, 0),
                                                                        (-1, agent._opponent_action_dim))
                            if agent.joint:
                                if agent.opponent_modelling:
                                    batch_n[i]['recent_opponent_observations'] = recent_opponent_observations_n[i]
                                    batch_n[i]['recent_opponent_actions'] = np.reshape(
                                        np.delete(deepcopy(recent_opponent_actions_n), i, 0),
                                        (-1, agent._opponent_action_dim))
                                    batch_n[i]['opponent_next_actions'] = agent.opponent_policy.get_actions(
                                        batch_n[i]['next_observations'])
                                else:
                                    batch_n[i]['opponent_next_actions'] = np.reshape(
                                        np.delete(deepcopy(target_next_actions_n), i, 0),
                                        (-1, agent._opponent_action_dim))
                            if isinstance(agent, MAVBAC) or isinstance(agent, MASQL):
                                agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i],
                                                   annealing=alpha)
                            else:
                                agent._do_training(iteration=t + epoch * agent._epoch_length, batch=batch_n[i])
                gt.stamp('train')

            logger.pop_prefix()
            sampler.terminate()


if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)
