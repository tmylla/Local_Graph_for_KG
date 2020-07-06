import torch
import os
import logging
from graph import Knowledge_graph
from environment import Environment
from baseline import ReactiveBaseline
import numpy as np
import utils
import globalvar as gl


class Trainer():
    def __init__(self, args, agent, data_loader):
        self.args = args
        self.agent = agent
        self.data_loader = data_loader
        # self.plotter = plotter
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.args.learning_rate)
        self.positive_reward = 1
        self.negative_reward = 0
        self.baseline = ReactiveBaseline(args, self.args.Lambda)
        self.decaying_beta_init = self.args.beta

        if self.args.use_cuda:
            self.agent.cuda()

    def save_model(self):
        dir_path = os.path.join(self.args.exps_dir, self.args.dataset)
        path = os.path.join(dir_path, "model.pkt")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.agent.state_dict(), path)

    def load_model(self):
        dir_path = os.path.join(self.args.exps_dir, self.args.dataset)
        path = os.path.join(dir_path, "model.pkt")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.agent.load_state_dict(torch.load(path))

    def get_reward(self, current_entities, answers, all_correct):
        reward = (current_entities == answers)
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)
        return reward

    def calc_cum_discounted_reward(self, rewards):
        '''
        R =
        :param rewards: Undiscounted rewards
        :return: Discounted rewards
        '''
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.args.max_step_length])  # (B*train_times, Max_S)

        if self.args.use_cuda:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.args.max_step_length - 1] = rewards
        for t in reversed(range(self.args.max_step_length)):
            running_add = self.args.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)  # (B * train_times, Max_out, Max_S)
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # scalar
        return entropy_loss

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward, decaying_beta):

        loss = torch.stack(all_loss, dim=1)  # (B * train_times, Max_S)
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value   # (B*train_times, Max_S)

        reward_mean = torch.mean(final_reward)  # scalar
        reward_std = torch.std(final_reward) + 1e-6  # scalar
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)  # (B * train_times, Max_S)
        entropy_loss = decaying_beta * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss  # scalar

        return total_loss

    def visual(self, env_name):
        gl._init()

        plotter = utils.VisdomLinePlotter(env_name=env_name)
        gl.set_value(env_name, plotter)

        return gl.get_value(env_name)


    def train(self):
        plotter = self.visual(env_name='Tutorial plots')

        logging.info("Begin train")
        train_graph = Knowledge_graph(self.args, self.data_loader, self.data_loader.get_train_graph_data())
        train_data = self.data_loader.get_train_data()
        environment = Environment(self.args, train_graph, train_data, "train")
        self.agent.set_graph(train_graph)

        batch_counter = 0
        current_decay = self.decaying_beta_init
        current_decay_count = 0

        for start_entities, queries, answers, all_correct in environment.get_next_batch():
            if batch_counter > self.args.train_batch:
                break
            else:
                batch_counter += 1

            current_decay_count += 1
            if current_decay_count == self.args.decay_batch:
                current_decay *= self.args.decay_rate
                current_decay_count = 0

            batch_size = start_entities.shape[0]  # (B * train_times)
            prev_state = [torch.zeros(start_entities.shape[0], self.args.state_embed_size),
                          torch.zeros(start_entities.shape[0], self.args.state_embed_size)]
			# list contains tenspr[(B * train_times, SE),(B * train_times, SE)]
            prev_relation = self.agent.get_dummy_start_relation(batch_size)  # (B * train_times)
            current_entities = start_entities  # (B * train_times)
            if self.args.use_cuda:
                prev_relation = prev_relation.cuda()
                prev_state[0] = prev_state[0].cuda()
                prev_state[1] = prev_state[1].cuda()

            all_loss = []
            all_logits = []
            all_action_id = []

            for step_length in range(self.args.max_step_length):
                loss, new_state, logits, action_id, next_entities, chosen_relation= \
                    self.agent.train_step(prev_state, prev_relation, current_entities,
                                    start_entities, queries, answers, all_correct, step_length)
                # （B * train_times）  (B * train_times, SE)  (B * train_times, Max_out)  (B * train_times)  (B * train_times)
                all_loss.append(loss)  #  len(list[(B * train_times),...])=Max_S
                all_logits.append(logits)
                all_action_id.append(action_id)
                prev_relation = chosen_relation
                current_entities = next_entities  # (B * train_times)
                prev_state = new_state

            rewards_np = self.agent.get_reward(current_entities, answers, all_correct,
                                            self.positive_reward, self.negative_reward)  # (B * train_times)
            rewards = torch.from_numpy(rewards_np)
            if self.args.use_cuda:
                rewards = rewards.cuda()
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            reinforce_loss = self.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward, current_decay)

            logging.info("reward: %s" % (np.mean(rewards_np)))
            # self.plotter.plot('reward', 'train', 'Rewards', batch_counter, np.mean(rewards_np))
            plotter.plot('reward', 'train', 'Rewards', batch_counter, np.mean(rewards_np))

            self.baseline.update(torch.mean(cum_discounted_reward))
            self.agent.zero_grad()
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.args.grad_clip_norm, norm_type=2)
            self.optimizer.step()

    def test(self):
        logging.info("\n***************************Begin test******************************")
        with torch.no_grad():
            test_graph = Knowledge_graph(self.args, self.data_loader, self.data_loader.get_test_graph_data())
            test_data = self.data_loader.get_test_data()
            environment = Environment(self.args, test_graph, test_data, "test")
            self.agent.set_graph(test_graph)

            total_examples = len(test_data)
            all_final_reward_1 = 0
            all_final_reward_3 = 0
            all_final_reward_5 = 0
            all_final_reward_10 = 0
            all_final_reward_20 = 0
            all_r_rank = 0

            for _start_entities, _relations, _answers, start_entities, relations, answers, all_correct\
                    in environment.get_next_batch():
                batch_size = len(start_entities)
                prev_state = [torch.zeros(start_entities.shape[0], self.args.state_embed_size),
                              torch.zeros(start_entities.shape[0], self.args.state_embed_size)]
                prev_relation = self.agent.get_dummy_start_relation(start_entities.shape[0])
                current_entities = start_entities
                log_current_prob = torch.zeros(start_entities.shape[0])
                if self.args.use_cuda:
                    prev_relation = prev_relation.cuda()
                    prev_state[0] = prev_state[0].cuda()
                    prev_state[1] = prev_state[1].cuda()
                    log_current_prob = log_current_prob.cuda()

                for step_length in range(self.args.max_step_length):
                    if step_length == 0:
                        chosen_state, chosen_relation, chosen_entities, log_current_prob = \
                            self.agent.test_step(prev_state, prev_relation, current_entities, log_current_prob,
                                                 start_entities, relations, answers, all_correct, batch_size, step_length)

                    else:
                        chosen_state, chosen_relation, chosen_entities, log_current_prob = \
                            self.agent.test_step(prev_state, prev_relation, current_entities, log_current_prob,
                                                 _start_entities, _relations, _answers, all_correct, batch_size, step_length)

                    prev_relation = chosen_relation
                    current_entities = chosen_entities
                    prev_state = chosen_state

                # 计算指标
                final_reward_1 = 0
                final_reward_3 = 0
                final_reward_5 = 0
                final_reward_10 = 0
                final_reward_20 = 0
                r_rank = 0

                rewards = self.agent.get_reward(current_entities, _answers, all_correct,
                                                self.positive_reward, self.negative_reward)
                rewards = np.array(rewards)
                rewards = rewards.reshape(-1, self.args.test_times)

                if self.args.use_cuda:
                    current_entities = current_entities.cpu()
                current_entities_np = current_entities.numpy()
                current_entities_np = current_entities_np.reshape(-1, self.args.test_times)

                for line_id in range(rewards.shape[0]):
                    seen = set()
                    pos = 0
                    find_ans = False
                    for loc_id in range(rewards.shape[1]):
                        if rewards[line_id][loc_id] == self.positive_reward:
                            find_ans = True
                            break
                        if current_entities_np[line_id][loc_id] not in seen:
                            seen.add(current_entities_np[line_id][loc_id])
                            pos += 1

                    if find_ans:
                        if pos < 20:
                            final_reward_20 += 1
                            if pos < 10:
                                final_reward_10 += 1
                                if pos < 5:
                                    final_reward_5 += 1
                                    if pos < 3:
                                        final_reward_3 += 1
                                        if pos < 1:
                                            final_reward_1 += 1
                    else:
                        r_rank += 1.0 / (pos + 1)

                    #log.info(("pos", (pos, find_ans, relations[line_id])))

                all_final_reward_1 += final_reward_1
                all_final_reward_3 += final_reward_3
                all_final_reward_5 += final_reward_5
                all_final_reward_10 += final_reward_10
                all_final_reward_20 += final_reward_20
                all_r_rank += r_rank

            all_final_reward_1 /= total_examples
            all_final_reward_3 /= total_examples
            all_final_reward_5 /= total_examples
            all_final_reward_10 /= total_examples
            all_final_reward_20 /= total_examples
            all_r_rank /= total_examples

            logging.info(("all_final_reward_1", all_final_reward_1))
            logging.info(("all_final_reward_3", all_final_reward_3))
            logging.info(("all_final_reward_5", all_final_reward_5))
            logging.info(("all_final_reward_10", all_final_reward_10))
            logging.info(("all_final_reward_20", all_final_reward_20))
            logging.info(("all_r_rank", all_r_rank))



    def save_model(self):
        dir_path = os.path.join(self.args.model_dir, self.args.dataset)
        path = os.path.join(dir_path, "model.pkt")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.agent.state_dict(), path)

    def load_model(self):
        dir_path = os.path.join(self.args.model_dir, self.args.dataset)
        path = os.path.join(dir_path, "model.pkt")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.agent.load_state_dict(torch.load(path))