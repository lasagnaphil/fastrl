//
// Created by lasagnaphil on 21. 1. 15..
//

#ifndef FASTRL_UTILS_H
#define FASTRL_UTILS_H

#include <fastrl/fastrl.h>
#include <toml.hpp>

namespace fastrl {

inline void copy_weights(torch::nn::Module& src, torch::nn::Module& dest) {
    torch::NoGradGuard guard {};
    auto dest_params = dest.named_parameters();
    auto dest_buffers = dest.named_buffers();
    for (auto& val : src.named_parameters()) {
        auto name = val.key();
        auto* t = dest_params.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        }
    }
    for (auto& val : src.named_buffers()) {
        auto name = val.key();
        auto* t = dest_buffers.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        }
    }
}

inline toml::value save_policy_options(const PolicyOptions& opt) {
    toml::value data;
    if (opt.action_dist_type == DistributionType::DiagGaussian) {
        data["action_dist_type"] = "DiagGaussian";
    }
    else if (opt.action_dist_type == DistributionType::Bernoulli) {
        data["action_dist_type"] = "Bernoulli";
    }
    if (opt.activation_type == NNActivationType::Tanh) {
        data["activation_type"] = "tanh";
    }
    else if (opt.activation_type == NNActivationType::ReLU) {
        data["activation_type"] = "relu";
    }
    data["actor_hidden_dim"] = opt.actor_hidden_dim;
    data["critic_hidden_dim"] = opt.critic_hidden_dim;
    data["log_std_init"] = opt.log_std_init;
    data["fix_log_std"] = opt.fix_log_std;
    data["ortho_init"] = opt.ortho_init;
    return data;
}

inline PolicyOptions load_policy_options(const toml::value& data) {
    PolicyOptions opt;

    auto action_dist_type = toml::find<std::string>(data, "action_dist_type");
    if (action_dist_type == "DiagGaussian") {
        opt.action_dist_type = DistributionType::DiagGaussian;
    }
    else if (action_dist_type == "Bernoulli") {
        opt.action_dist_type = DistributionType::Bernoulli;
    }
    else {
        std::cout << "Action distribution type " << action_dist_type << " invalid!" << std::endl;
    }

    auto activation_type = toml::find<std::string>(data, "activation_type");
    if (activation_type == "tanh") {
        opt.activation_type = NNActivationType::Tanh;
    }
    else if (activation_type == "relu") {
        opt.activation_type = NNActivationType::ReLU;
    }
    else {
        std::cout << "NN activation type " << action_dist_type << " invalid!" << std::endl;
    }

    opt.actor_hidden_dim = toml::find<std::vector<int>>(data, "actor_hidden_dim");
    opt.critic_hidden_dim = toml::find<std::vector<int>>(data, "critic_hidden_dim");
    opt.log_std_init = toml::find<float>(data, "log_std_init");
    opt.fix_log_std = toml::find<bool>(data, "fix_log_std");
    opt.ortho_init = toml::find<bool>(data, "ortho_init");

    return opt;
}

}

#endif //FASTRL_UTILS_H
