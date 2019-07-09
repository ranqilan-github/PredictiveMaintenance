# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:38:03 2019

@author: makraus
"""

import pyro
import torch
from pyro.distributions import Normal, Gumbel, LogNormal, Gamma
from torch import zeros, ones

cuda = True
ftype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
ltype = torch.cuda.LongTensor if cuda else torch.LongTensor
p = 26
hidden_size_1 = 100
hidden_size_2 = 50

softplus = torch.nn.Softplus()


class RegressionModel(torch.nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(p, 1, bias=True)

    def forward(self, x):
        return self.linear(x)
    
    
class LSTMModel(torch.nn.Module):
    def __init__(self, p):
        super(LSTMModel, self).__init__()
        self.lstm_1 = torch.nn.LSTM(input_size=p, hidden_size=hidden_size_1)
        self.lstm_2 = torch.nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2)
        self.linear = torch.nn.Linear(hidden_size_2, 1)
        self.dropout = torch.nn.Dropout(p=0.5)
      
    def forward(self, x):
        h_t, c_t = self.lstm_1(x)
        h_t = self.dropout(h_t)
        h_t2, c_t2 = self.lstm_2(h_t)
        h_t2 = self.dropout(h_t2)
        x = self.linear(h_t2[-1])
        return x


regression_model = RegressionModel(p)
lstm_model = LSTMModel(p)
if cuda:
    softplus.cuda()
    regression_model.cuda()
    lstm_model.cuda()


def model(x_data, y_data):
    ##################################################################
    # Structured effect: Create unit normal priors over the parameters
    ##################################################################
    
    alpha_mean = 1. * ones(1,1)
    alpha_scale = 1. * ones(1,1)
    beta_mean = 1. * ones(1,1)
    beta_scale = 1. * ones(1,1)
    
    alpha = pyro.sample('alpha', Normal(alpha_mean, alpha_scale))
    beta = pyro.sample('beta', Normal(beta_mean, beta_scale))
    
    struct_sample = pyro.sample('struct_sample', LogNormal(alpha_mean, beta_mean))
    
    ##############################################################
    # Linear effect: Create unit normal priors over the parameters
    ##############################################################
    
    w_mean = zeros(1, p).type(ftype)
    w_scale = 2 * ones(1, p).type(ftype)
    w_bias_mean = zeros(1, 1).type(ftype)
    w_bias_scale = 10. * ones(1, 1).type(ftype)
    w_prior = Normal(w_mean, w_scale).reshape(extra_event_dims=1)
    b_prior = Normal(w_bias_mean, w_bias_scale).reshape(extra_event_dims=1)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    
    lifted_module = pyro.random_module("Linear", regression_model, priors)
    lifted_reg_model = lifted_module().cuda()
    
    ##############################################################
    # Random effect: Create unit normal priors over the parameters
    ##############################################################
    
    # First LSTM layer
    p_lstm_1_hidden_hidden_weights_mean = zeros(4 * hidden_size_1, hidden_size_1).type(ftype)
    p_lstm_1_hidden_hidden_weights_scale = 10. * ones(4 * hidden_size_1, hidden_size_1).type(ftype)
    p_lstm_1_input_hidden_weights_mean = zeros(4 * hidden_size_1,p).type(ftype)
    p_lstm_1_input_hidden_weights_scale = 10. * ones(4 * hidden_size_1,p).type(ftype)
    p_lstm_1_hidden_hidden_bias_mean = zeros(4 * hidden_size_1).type(ftype)
    p_lstm_1_hidden_hidden_bias_scale = 10. * ones(4 * hidden_size_1).type(ftype)
    p_lstm_1_input_hidden_bias_mean = zeros(4 * hidden_size_1).type(ftype)
    p_lstm_1_input_hidden_bias_scale = 10. * ones(4 * hidden_size_1).type(ftype)
    
    lstm_1_hh_w_prior = Normal(p_lstm_1_hidden_hidden_weights_mean, 
                           p_lstm_1_hidden_hidden_weights_scale).reshape(extra_event_dims=2)
    lstm_1_ih_w_prior = Normal(p_lstm_1_input_hidden_weights_mean, 
                           p_lstm_1_input_hidden_weights_scale).reshape(extra_event_dims=2)
    lstm_1_hh_b_prior = Normal(p_lstm_1_hidden_hidden_bias_mean, 
                           p_lstm_1_hidden_hidden_bias_scale).reshape(extra_event_dims=1)
    lstm_1_ih_b_prior = Normal(p_lstm_1_input_hidden_bias_mean, 
                           p_lstm_1_input_hidden_bias_scale).reshape(extra_event_dims=1)
    
    #Second LSTM layer
    p_lstm_2_hidden_hidden_weights_mean = zeros(4 * hidden_size_2, hidden_size_2).type(ftype)
    p_lstm_2_hidden_hidden_weights_scale = 10. * ones(4 * hidden_size_2, hidden_size_2).type(ftype)
    p_lstm_2_input_hidden_weights_mean = zeros(4 * hidden_size_2,hidden_size_1).type(ftype)
    p_lstm_2_input_hidden_weights_scale = 10. * ones(4 * hidden_size_2,hidden_size_1).type(ftype)
    p_lstm_2_hidden_hidden_bias_mean = zeros(4 * hidden_size_2).type(ftype)
    p_lstm_2_hidden_hidden_bias_scale = 10. * ones(4 * hidden_size_2).type(ftype)
    p_lstm_2_input_hidden_bias_mean = zeros(4 * hidden_size_2).type(ftype)
    p_lstm_2_input_hidden_bias_scale = 10. * ones(4 * hidden_size_2).type(ftype)
    
    lstm_2_hh_w_prior = Normal(p_lstm_2_hidden_hidden_weights_mean, 
                           p_lstm_2_hidden_hidden_weights_scale).reshape(extra_event_dims=2)
    lstm_2_ih_w_prior = Normal(p_lstm_2_input_hidden_weights_mean, 
                           p_lstm_2_input_hidden_weights_scale).reshape(extra_event_dims=2)
    lstm_2_hh_b_prior = Normal(p_lstm_2_hidden_hidden_bias_mean, 
                           p_lstm_2_hidden_hidden_bias_scale).reshape(extra_event_dims=1)
    lstm_2_ih_b_prior = Normal(p_lstm_2_input_hidden_bias_mean, 
                           p_lstm_2_input_hidden_bias_scale).reshape(extra_event_dims=1)

    #Linear output layer
    p_lstm_linear_w_mean = zeros(1, hidden_size_2).type(ftype)
    p_lstm_linear_w_scale = 1. * ones(1, hidden_size_2).type(ftype)
    p_lstm_linear_b_mean = zeros(1).type(ftype)
    p_lstm_linear_b_scale = 1. * ones(1).type(ftype)
    
    lstm_linear_w_prior = Normal(p_lstm_linear_w_mean, p_lstm_linear_w_scale).reshape(extra_event_dims=1)
    lstm_linear_b_prior = Normal(p_lstm_linear_b_mean, p_lstm_linear_b_scale).reshape(extra_event_dims=1)
    
    priors_lstm = {'lstm_1.weight_ih_l0': lstm_1_ih_w_prior,
                  'lstm_1.weight_hh_l0': lstm_1_hh_w_prior, 
                  'lstm_1.bias_hh_l0': lstm_1_hh_b_prior,
                  'lstm_1.bias_ih_l0': lstm_1_ih_b_prior,
                  'lstm_2.weight_ih_l0': lstm_2_ih_w_prior,
                  'lstm_2.weight_hh_l0': lstm_2_hh_w_prior, 
                  'lstm_2.bias_hh_l0': lstm_2_hh_b_prior,
                  'lstm_2.bias_ih_l0': lstm_2_ih_b_prior,
                  'linear.weight': lstm_linear_w_prior,
                  'linear.bias': lstm_linear_b_prior}
    
    lifted_lstm_module = pyro.random_module("LSTM", lstm_model, priors_lstm)
    lifted_lstm_model = lifted_lstm_module().cuda()
        
    #######################################
    # Compose struct, linear and LSTM parts
    #######################################
    
    with pyro.iarange('map', use_cuda=True):
        linear_sample = lifted_reg_model(x_data[:,-1,:]).squeeze(-1)
        lstm_sample = lifted_lstm_model(torch.transpose(x_data,0,1)).squeeze(-1)
        y_pred = struct_sample + linear_sample + lstm_sample
        pyro.sample('obs', Normal(y_pred, 1. * torch.ones(1,1).type(ftype)), obs=y_data)
        
def guide(x_data, y_data):
    ##################################################################
    # Structured effect: Create unit normal priors over the parameters
    ##################################################################
    q_struct_alpha_mean = torch.tensor(torch.ones(1,1) + torch.randn(1,1), requires_grad=True)
    q_struct_alpha_scale = torch.tensor(torch.ones(1,1) + torch.randn(1,1), requires_grad=True)
    q_struct_beta_mean = torch.tensor(torch.ones(1,1) + torch.randn(1,1), requires_grad=True)
    q_struct_beta_scale = torch.tensor(torch.ones(1,1) + torch.randn(1,1), requires_grad=True)
    
    ##############################################################
    # Linear effect: Create unit normal priors over the parameters
    ##############################################################
    q_linear_w_mean = torch.tensor(torch.zeros(1,p) + torch.randn(1,p)).type(ftype)
    q_linear_w_scale = torch.tensor(torch.ones(1,p) + torch.randn(1,p)).type(ftype)
    q_linear_b_mean = torch.tensor(torch.zeros(1,1) + torch.randn(1,1)).type(ftype)
    q_linear_b_scale = torch.tensor(torch.ones(1,1) + torch.randn(1,1)).type(ftype)
    
    linear_a_loc_param = pyro.param("guide_linear_alpha_mean", q_linear_w_mean)
    linear_a_scale_param = softplus(pyro.param("guide_linear_alpha_log_scale", q_linear_w_scale))
    linear_b_loc_param = pyro.param("guide_linear_bias_mean", q_linear_b_mean)
    linear_b_scale_param = softplus(pyro.param("guide_linear_bias_log_scale", q_linear_b_scale))

    w_dist = Normal(linear_a_loc_param, linear_a_scale_param).reshape(extra_event_dims=1)
    b_dist = Normal(linear_b_loc_param, linear_b_scale_param).reshape(extra_event_dims=1)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}

    lifted_module = pyro.random_module("Linear", regression_model, dists)
    
    ##############################################################
    # Random effect: Create unit normal priors over the parameters
    ##############################################################
    
    #First LSTM Layer
    q_lstm_1_hidden_hidden_weights_mean = torch.tensor(torch.zeros(4 * hidden_size_1,hidden_size_1) + 
                                                     torch.randn(4 * hidden_size_1,hidden_size_1)).type(ftype)
    q_lstm_1_hidden_hidden_weights_scale = torch.tensor(torch.ones(4 * hidden_size_1,hidden_size_1) + 
                                                      torch.randn(4 * hidden_size_1,hidden_size_1)).type(ftype)
    q_lstm_1_input_hidden_weights_mean = torch.tensor(torch.zeros(4 * hidden_size_1,p) + 
                                                    torch.randn(4 * hidden_size_1,p)).type(ftype)
    q_lstm_1_input_hidden_weights_scale = torch.tensor(torch.ones(4 * hidden_size_1,p) + 
                                                     torch.randn(4 * hidden_size_1,p)).type(ftype)
    q_lstm_1_hidden_hidden_bias_mean = torch.tensor(torch.zeros(4 * hidden_size_1) + 
                                                  torch.randn(4 * hidden_size_1)).type(ftype)
    q_lstm_1_hidden_hidden_bias_scale = torch.tensor(torch.ones(4 * hidden_size_1) + 
                                                   torch.randn(4 * hidden_size_1)).type(ftype)
    q_lstm_1_input_hidden_bias_mean = torch.tensor(torch.zeros(4 * hidden_size_1) + 
                                                 torch.randn(4 * hidden_size_1)).type(ftype)
    q_lstm_1_input_hidden_bias_scale = torch.tensor(torch.ones(4 * hidden_size_1) + 
                                                  torch.randn(4 * hidden_size_1)).type(ftype)
    
    lstm_1_hh_w_loc_param = pyro.param("lstm_1_hh_w_loc_param", q_lstm_1_hidden_hidden_weights_mean)
    lstm_1_hh_w_scale_param = softplus(pyro.param("lstm_1_hh_w_scale_param", q_lstm_1_hidden_hidden_weights_scale))
    lstm_1_ih_w_loc_param = pyro.param("lstm_1_ih_w_loc_param", q_lstm_1_input_hidden_weights_mean)
    lstm_1_ih_w_scale_param = softplus(pyro.param("lstm_1_ih_w_scale_param", q_lstm_1_input_hidden_weights_scale))
    lstm_1_hh_b_loc_param = pyro.param("lstm_1_hh_b_loc_param", q_lstm_1_hidden_hidden_bias_mean)
    lstm_1_hh_b_scale_param = softplus(pyro.param("lstm_1_hh_b_scale_param", q_lstm_1_hidden_hidden_bias_scale))
    lstm_1_ih_b_loc_param = pyro.param("lstm_1_ih_b_loc_param", q_lstm_1_input_hidden_bias_mean)
    lstm_1_ih_b_scale_param = softplus(pyro.param("lstm_1_ih_b_scale_param", q_lstm_1_input_hidden_bias_scale))
    
    lstm_1_hh_w = Normal(lstm_1_hh_w_loc_param, lstm_1_hh_w_scale_param).reshape(extra_event_dims=2)
    lstm_1_ih_w = Normal(lstm_1_ih_w_loc_param, lstm_1_ih_w_scale_param).reshape(extra_event_dims=2)
    lstm_1_hh_b = Normal(lstm_1_hh_b_loc_param, lstm_1_hh_b_scale_param).reshape(extra_event_dims=1)
    lstm_1_ih_b = Normal(lstm_1_ih_b_loc_param, lstm_1_ih_b_scale_param).reshape(extra_event_dims=1)
    
    #Second LSTM Layer
    q_lstm_2_hidden_hidden_weights_mean = torch.tensor(torch.zeros(4 * hidden_size_2,hidden_size_2) + 
                                                     torch.randn(4 * hidden_size_2,hidden_size_2)).type(ftype)
    q_lstm_2_hidden_hidden_weights_scale = torch.tensor(torch.ones(4 * hidden_size_2,hidden_size_2) + 
                                                      torch.randn(4 * hidden_size_2,hidden_size_2)).type(ftype)
    q_lstm_2_input_hidden_weights_mean = torch.tensor(torch.zeros(4 * hidden_size_2,hidden_size_1) + 
                                                    torch.randn(4 * hidden_size_2,hidden_size_1)).type(ftype)
    q_lstm_2_input_hidden_weights_scale = torch.tensor(torch.ones(4 * hidden_size_2,hidden_size_1) + 
                                                     torch.randn(4 * hidden_size_2,hidden_size_1)).type(ftype)
    q_lstm_2_hidden_hidden_bias_mean = torch.tensor(torch.zeros(4 * hidden_size_2) + 
                                                  torch.randn(4 * hidden_size_2)).type(ftype)
    q_lstm_2_hidden_hidden_bias_scale = torch.tensor(torch.ones(4 * hidden_size_2) + 
                                                   torch.randn(4 * hidden_size_2)).type(ftype)
    q_lstm_2_input_hidden_bias_mean = torch.tensor(torch.zeros(4 * hidden_size_2) + 
                                                 torch.randn(4 * hidden_size_2)).type(ftype)
    q_lstm_2_input_hidden_bias_scale = torch.tensor(torch.ones(4 * hidden_size_2) + 
                                                  torch.randn(4 * hidden_size_2)).type(ftype)
      
    lstm_2_hh_w_loc_param = pyro.param("lstm_2_hh_w_loc_param", q_lstm_2_hidden_hidden_weights_mean)
    lstm_2_hh_w_scale_param = softplus(pyro.param("lstm_2_hh_w_scale_param", q_lstm_2_hidden_hidden_weights_scale))
    lstm_2_ih_w_loc_param = pyro.param("lstm_2_ih_w_loc_param", q_lstm_2_input_hidden_weights_mean)
    lstm_2_ih_w_scale_param = softplus(pyro.param("lstm_2_ih_w_scale_param", q_lstm_2_input_hidden_weights_scale))
    lstm_2_hh_b_loc_param = pyro.param("lstm_2_hh_b_loc_param", q_lstm_2_hidden_hidden_bias_mean)
    lstm_2_hh_b_scale_param = softplus(pyro.param("lstm_2_hh_b_scale_param", q_lstm_2_hidden_hidden_bias_scale))
    lstm_2_ih_b_loc_param = pyro.param("lstm_2_ih_b_loc_param", q_lstm_2_input_hidden_bias_mean)
    lstm_2_ih_b_scale_param = softplus(pyro.param("lstm_2_ih_b_scale_param", q_lstm_2_input_hidden_bias_scale))
    
    lstm_2_hh_w = Normal(lstm_2_hh_w_loc_param, lstm_2_hh_w_scale_param).reshape(extra_event_dims=2)
    lstm_2_ih_w = Normal(lstm_2_ih_w_loc_param, lstm_2_ih_w_scale_param).reshape(extra_event_dims=2)
    lstm_2_hh_b = Normal(lstm_2_hh_b_loc_param, lstm_2_hh_b_scale_param).reshape(extra_event_dims=1)
    lstm_2_ih_b = Normal(lstm_2_ih_b_loc_param, lstm_2_ih_b_scale_param).reshape(extra_event_dims=1)
    
    #Linear output layer
    q_lstm_linear_w_mean = torch.tensor(torch.zeros(1,hidden_size_2) + torch.randn(1,hidden_size_2)).type(ftype)
    q_lstm_linear_w_scale = torch.tensor(torch.ones(1,hidden_size_2) + torch.randn(1,hidden_size_2)).type(ftype)
    q_lstm_linear_b_mean = torch.tensor(torch.zeros(1) + torch.randn(1)).type(ftype)
    q_lstm_linear_b_scale = torch.tensor(torch.ones(1) + torch.randn(1)).type(ftype)
    
    lstm_linear_a_loc_param = pyro.param("lstm_linear_a_loc_param", q_lstm_linear_w_mean)
    lstm_linear_a_scale_param = softplus(pyro.param("lstm_linear_a_scale_param", q_lstm_linear_w_scale))
    lstm_linear_b_loc_param = pyro.param("lstm_linear_b_loc_param", q_lstm_linear_b_mean)
    lstm_linear_b_scale_param = softplus(pyro.param("lstm_linear_b_scale_param", q_lstm_linear_b_scale))  
    
    lstm_linear_w = Normal(lstm_linear_a_loc_param, lstm_linear_a_scale_param).reshape(extra_event_dims=1)
    lstm_linear_b = Normal(lstm_linear_b_loc_param, lstm_linear_b_scale_param).reshape(extra_event_dims=1)
    
    lstm_dists = {'lstm_1.weight_hh_l0': lstm_1_hh_w, 
                  'lstm_1.bias_hh_l0': lstm_1_hh_b,
                  'lstm_1.weight_ih_l0': lstm_1_ih_w,
                  'lstm_1.bias_ih_l0': lstm_1_ih_b,
                  'lstm_2.weight_hh_l0': lstm_2_hh_w, 
                  'lstm_2.bias_hh_l0': lstm_2_hh_b,
                  'lstm_2.weight_ih_l0': lstm_2_ih_w,
                  'lstm_2.bias_ih_l0': lstm_2_ih_b,
                  'linear.weight': lstm_linear_w,
                  'linear.bias': lstm_linear_b}
    
    lifted_lstm_module = pyro.random_module("LSTM", lstm_model, lstm_dists)
    
    return lifted_module().cuda(), lifted_lstm_module().cuda()