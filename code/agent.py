import torch.nn as nn
import numpy as np
import torch
import logging
from collections import defaultdict
from torchsummary import summary


class Policy_step(nn.Module):
    '''
    s(t)=LSTMCell(a(t-1), s(t-1))
    inputs : action(t-1), state(t-1)
    outputs : s(t), Cell_state(t)

    s(t)_size,Cell_state(t)_size:(batch, hidden_size)
    '''
    def __init__(self, args):
        super(Policy_step, self).__init__()
        self.args = args
        self.lstm_cell = torch.nn.LSTMCell(input_size=self.args.action_embed_size,
                          hidden_size=self.args.state_embed_size)

    def forward(self, prev_action, prev_state):
        output, new_state = self.lstm_cell(prev_action, prev_state)
        return output, (output, new_state)


class Policy_mlp(nn.Module):
    '''
    inputs : state_query(current_state + ori_relation)
    outputs : next_action
    '''
    def __init__(self, args):
        super(Policy_mlp, self).__init__()
        self.args = args
        self.hidden_size = args.mlp_hidden_size
        self.mlp_l1= nn.Linear(self.args.state_embed_size + self.args.relation_embed_size,
                               self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(self.hidden_size, self.args.action_embed_size, bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden)).unsqueeze(1)
        return output


class Agent(nn.Module):
    def __init__(self, args, data_loader, graph=None):
        super(Agent, self).__init__()
        self.args = args
        self.data_loader = data_loader
        self.graph = graph
        self.relation_embedding = nn.Embedding(self.args.num_relation, self.args.relation_embed_size)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.policy_step = Policy_step(self.args)
        self.policy_mlp = Policy_mlp(self.args)

        if self.args.use_entity_embed:
            self.entity_embedding = nn.Embedding(self.args.num_entity, self.args.entity_embed_size)

    def train_step(self, prev_state, prev_relation, current_entities, start_entities, queries, answers, all_correct, step_length):
        prev_action_embedding = self.relation_embedding(prev_relation)  # (B * train_times, AE)  Action_Embed_size
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # (B * train_times, SE)  Batch, State_Embed_size

        actions_id = self.graph.get_out(current_entities, start_entities, queries, answers, all_correct, step_length)  # (B * train_times, Max_out, 2)
        out_relations_id = actions_id[:, :, 0]  # (B * train_times, Max_out)
        out_entities_id = actions_id[:, :, 1]  # (B * train_times, Max_out)
        out_relations = self.relation_embedding(out_relations_id)  # (B * train_times, Max_out, RE)
        action = out_relations  # (B * train_times, Max_out, RE)

        current_state = output.squeeze()
        queries_embedding = self.relation_embedding(queries)
        state_query = torch.cat([current_state, queries_embedding], -1)  # (B * train_times, SE + RE)
        output = self.policy_mlp(state_query)  # (B * train_times, 1, AE)

        prelim_scores = torch.sum(torch.mul(output, action), dim=-1)  # (B * train_times, Max_out, AE)  -sum->  (B * train_times, Max_out)
        dummy_relations_id = torch.ones_like(out_relations_id, dtype=torch.int64) * self.data_loader.relation2num["Pad"]
        mask = torch.eq(out_relations_id, dummy_relations_id)  # (B * train_times, Max_out)
        dummy_scores = torch.ones_like(prelim_scores) * (-99999)
        scores = torch.where(mask, dummy_scores, prelim_scores)  # (B * train_times, Max_out)

        action_prob = torch.softmax(scores, dim=1)  # (B * train_times, Max_out)
        action_id = torch.multinomial(action_prob, 1)  #  （B * train_times,1）  概率取样
        chosen_relation = torch.gather(out_relations_id, dim=1, index=action_id).squeeze()  # (B * train_times)

        logits = torch.nn.functional.log_softmax(scores, dim=1)  # (B * train_times, Max_out)
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)  # (B * train_times, Max_out)
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)  # (B * train_times)

        action_id = action_id.squeeze()
        next_entities = self.graph.get_next(current_entities, action_id)

        # sss = self.data_loader.num2relation[(int)(queries[0])] + "\t" + self.data_loader.num2relation[(int)(chosen_relation[0])]
        # #log.info(sss)

        return loss, new_state, logits, action_id, next_entities, chosen_relation

    def test_step(self, prev_state, prev_relation, current_entities, log_current_prob,
                  start_entities, queries, answers, all_correct, batch_size, step_length):
        prev_action_embedding = self.relation_embedding(prev_relation)
        output, new_state = self.policy_step(prev_action_embedding, prev_state)

        actions_id = self.graph.get_out(current_entities, start_entities, queries, answers, all_correct, step_length)
        out_relations_id = actions_id[:, :, 0]
        out_entities_id = actions_id[:, :, 1]
        out_relations = self.relation_embedding(out_relations_id)
        action = out_relations

        current_state = output.squeeze()
        queries_embedding = self.relation_embedding(queries)
        state_query = torch.cat([current_state, queries_embedding], -1)
        output = self.policy_mlp(state_query)

        prelim_scores = torch.sum(torch.mul(output, action), dim=-1)
        dummy_relations_id = torch.ones_like(out_relations_id, dtype=torch.int64) * self.data_loader.relation2num["Pad"]
        mask = torch.eq(out_relations_id, dummy_relations_id)
        dummy_scores = torch.ones_like(prelim_scores) * (-9999)
        scores = torch.where(mask, dummy_scores, prelim_scores)

        action_prob = torch.softmax(scores, dim=1)  # (B/N, Max_out)
        log_action_prob = torch.log(action_prob)

        chosen_state, chosen_relation, chosen_entities, log_current_prob = self.test_search\
            (new_state, log_current_prob, log_action_prob, out_relations_id, out_entities_id, batch_size)

        return chosen_state, chosen_relation, chosen_entities, log_current_prob

    def test_search(self, new_state, log_current_prob, log_action_prob, out_relations_id, out_entities_id, batch_size):
        log_current_prob = log_current_prob.repeat_interleave(self.args.max_out).view(batch_size, -1)  # Shape ??
        log_action_prob = log_action_prob.view(batch_size, -1)
        log_trail_prob = torch.add(log_action_prob, log_current_prob)
        top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, self.args.test_times)

        new_state_0 = new_state[0].repeat_interleave(self.args.max_out)\
            .view(batch_size, -1, self.args.state_embed_size)
        new_state_1 = new_state[1].repeat_interleave(self.args.max_out) \
            .view(batch_size, -1, self.args.state_embed_size)

        out_relations_id = out_relations_id.view(batch_size, -1)
        out_entities_id = out_entities_id.view(batch_size, -1)

        chosen_relation = torch.gather(out_relations_id, dim=1, index=top_k_action_id).view(-1)
        chosen_entities = torch.gather(out_entities_id, dim=1, index=top_k_action_id).view(-1)
        log_current_prob = torch.gather(log_trail_prob, dim=1, index=top_k_action_id).view(-1)

        top_k_action_id_state = top_k_action_id.unsqueeze(2).repeat(1, 1, self.args.state_embed_size)
        chosen_state = \
            (torch.gather(new_state_0, dim=1, index=top_k_action_id_state).view(-1, self.args.state_embed_size),
             torch.gather(new_state_1, dim=1, index=top_k_action_id_state).view(-1, self.args.state_embed_size))

        return chosen_state, chosen_relation, chosen_entities, log_current_prob

    def set_graph(self, graph):
        self.graph = graph

    def get_dummy_start_relation(self, batch_size):
        '''
        Set the relationship to num['Start'] at the initial step.
        '''
        dummy_start_item = self.data_loader.relation2num["Strat"]
        dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
        return dummy_start

    def get_reward(self, current_entities, answers, all_correct, positive_reward, negative_reward):
        reward = (current_entities == answers).cpu()

        reward = reward.numpy()  # (B * train_times)
        condlist = [reward == True, reward == False]
        choicelist = [positive_reward, negative_reward]
        reward = np.select(condlist, choicelist)
        return reward

    def print_parameter(self):
        for param in self.named_parameters():
            print(param[0], param[1])