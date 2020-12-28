//
// Created by lasagnaphil on 20. 12. 26..
//

#include "fastrl/fastrl.h"

namespace fastrl {

NormalDistribution::NormalDistribution(torch::Tensor mu, torch::Tensor sigma) : mu(std::move(mu)), sigma(std::move(sigma)) {}

torch::Tensor NormalDistribution::entropy() {
    return 0.5f + 0.5f * std::log(2 * M_PI) + torch::log(sigma);
}

torch::Tensor NormalDistribution::log_prob(torch::Tensor value) const {
    return -(value - mu).pow(2) / (2 * sigma.pow(2)) - sigma.log() - std::log(std::sqrt(2*M_PI));
}

torch::Tensor NormalDistribution::sample(c10::ArrayRef<int64_t> sample_shape) const {
    auto no_grad_guard = torch::NoGradGuard();
    return at::normal(mu.expand(sample_shape), sigma.expand(sample_shape));
}

Policy::Policy(const PolicyOptions& options) : opt(options) {
    namespace nn = torch::nn;

    actor = register_module("actor", nn::Sequential());
    for (int i = 0; i < opt.actor_hidden_dim.size(); i++) {
        if (i < opt.actor_hidden_dim.size() - 1) {
            if (i == 0) {
                actor->push_back(nn::Linear(opt.state_size, opt.actor_hidden_dim[i]));
            }
            else {
                actor->push_back(nn::Linear(opt.actor_hidden_dim[i], opt.actor_hidden_dim[i+1]));
            }
            switch (opt.activation_type) {
                case NNActivationType::ReLU: actor->push_back(nn::ReLU()); break;
                case NNActivationType::Tanh: actor->push_back(nn::Tanh()); break;
            }
        }
        else {
            actor->push_back(nn::Linear(opt.actor_hidden_dim[i], opt.action_size));
        }
    }

    critic = register_module("critic", nn::Sequential());
    for (int i = 0; i < opt.critic_hidden_dim.size(); i++) {
        if (i < opt.critic_hidden_dim.size() - 1) {
            if (i == 0) {
                critic->push_back(nn::Linear(opt.state_size, opt.critic_hidden_dim[i]));
            }
            else {
                critic->push_back(nn::Linear(opt.critic_hidden_dim[i], opt.critic_hidden_dim[i+1]));
            }
            switch (opt.activation_type) {
                case NNActivationType::ReLU: critic->push_back(nn::ReLU()); break;
                case NNActivationType::Tanh: critic->push_back(nn::Tanh()); break;
            }
        }
        else {
            critic->push_back(nn::Linear(opt.critic_hidden_dim[i], 1));
        }
    }
}

std::pair<NormalDistribution, torch::Tensor> Policy::forward(torch::Tensor state) {
    auto action_dist = actor->forward(state).chunk(2, -1);
    torch::Tensor value = critic->forward(state);
    return {NormalDistribution{action_dist[0], action_dist[1]}, value};
}

RolloutBuffer::RolloutBuffer(RolloutBufferOptions options) : opt(options),
                                                                             observations_data(torch::zeros({opt.buffer_size, opt.num_envs, opt.state_size})),
                                                                             actions_data(torch::zeros({opt.buffer_size, opt.num_envs, opt.action_size})),
                                                                             rewards_data(torch::zeros({opt.buffer_size, opt.num_envs})),
                                                                             returns_data(torch::zeros({opt.buffer_size, opt.num_envs})),
                                                                             dones_data(torch::zeros({opt.buffer_size, opt.num_envs})),
                                                                             values_data(torch::zeros({opt.buffer_size, opt.num_envs})),
                                                                             log_probs_data(torch::zeros({opt.buffer_size, opt.num_envs})),
                                                                             advantages_data(torch::zeros({opt.buffer_size, opt.num_envs})),
                                                                             observations(observations_data.accessor<float, 3>()),
                                                                             actions(actions_data.accessor<float, 3>()),
                                                                             rewards(rewards_data.accessor<float, 2>()),
                                                                             returns(returns_data.accessor<float, 2>()),
                                                                             dones(dones_data.accessor<float, 2>()),
                                                                             values(values_data.accessor<float, 2>()),
                                                                             log_probs(log_probs_data.accessor<float, 2>()),
                                                                             advantages(advantages_data.accessor<float, 2>())
{
}

void RolloutBuffer::compute_returns_and_advantage(const float* last_values, const float* last_dones) {
    std::vector<float> next_non_terminal(opt.buffer_size);
    std::vector<float> next_values(opt.buffer_size);
    std::vector<float> delta(opt.buffer_size);
    std::vector<float> last_gae_lam(opt.buffer_size);

    for (int step = opt.buffer_size - 1; step >= 0; step--) {
        for (int k = 0; k < opt.num_envs; k++) {
            if (step == opt.buffer_size - 1) {
                next_non_terminal[k] = 1.0 - last_dones[k];
                next_values[k] = last_values[k];
            }
            else {
                next_non_terminal[k] = 1.0 - dones[step + 1][k];
                next_values[k] = values[step + 1][k];
            }
            delta[k] = rewards[step][k] + opt.gamma * next_values[k] * next_non_terminal[k] - values[step][k];
            last_gae_lam[k] = delta[k] + opt.gamma * opt.gae_lambda * next_non_terminal[k] * last_gae_lam[k];
        }
    }
    for (int step = 0; step < opt.buffer_size; step++) {
        for (int k = 0; k < opt.num_envs; k++) {
            returns[step][k] = advantages[step][k] + values[step][k];
        }
    }
}

void RolloutBuffer::add(int env_id, const float* obs, const float* action, float reward, bool done, float value,
                                float log_prob) {
    std::copy_n(obs, opt.state_size, observations[pos][env_id].data());
    std::copy_n(action, opt.action_size, actions[pos][env_id].data());
    rewards[pos][env_id] = reward;
    dones[pos][env_id] = done;
    values[pos][env_id] = value;
    log_probs[pos][env_id] = log_prob;
    pos++;
    if (pos == opt.buffer_size) {
        full = true;
    }
}

std::vector<RolloutBufferBatch> RolloutBuffer::get_samples(int batch_size) {
    if (!full) {
        std::cerr << "Error in get_samples(): RolloutBuffer is not full yet!" << std::endl;
        return {};
    }
    std::vector<int64_t> indices(opt.buffer_size * opt.num_envs);
    for (int64_t i = 0; i < indices.size(); i++) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

    std::vector<RolloutBufferBatch> batches;
    int start_idx = 0;
    while (start_idx < opt.buffer_size * opt.num_envs) {
        RolloutBufferBatch batch;
        batch.observations = torch::empty({batch_size, opt.state_size});
        batch.actions = torch::empty({batch_size, opt.action_size});
        batch.old_values = torch::empty({batch_size});
        batch.old_log_prob = torch::empty({batch_size});
        batch.advantages = torch::empty({batch_size});
        batch.returns = torch::empty({batch_size});

        auto batch_indices = c10::IntArrayRef(indices.data() + start_idx, batch_size);
        for (int64_t i : batch_indices) {
            std::copy_n(observations.data() + i*opt.state_size, opt.state_size, batch.observations.data_ptr<float>());
            std::copy_n(actions.data() + i*opt.action_size, opt.action_size, batch.observations.data_ptr<float>());
            batch.old_values.data()[i] = values.data()[i];
            batch.old_log_prob.data()[i] = log_probs.data()[i];
            batch.advantages.data()[i] = advantages.data()[i];
            batch.returns.data()[i] = returns.data()[i];
        }
        start_idx += batch_size;
        batches.push_back(batch);
    }
    return batches;
}

PPO::PPO(PPOOptions options, std::shared_ptr<Policy> policy, std::shared_ptr<RolloutBuffer> rollout_buffer)
        : opt(options), policy(std::move(policy)), rollout_buffer(std::move(rollout_buffer)) {

    optimizer = std::make_shared<torch::optim::SGD>(
            policy->parameters(), torch::optim::SGDOptions(opt.learning_rate));
}

void PPO::train() {
    // buffers for logging
    std::vector<float> entropy_losses, all_kl_divs, pg_losses, value_losses, clip_fractions;

    std::vector<RolloutBufferBatch> batches = rollout_buffer->get_samples(opt.batch_size);
    for (int epoch = 0; epoch < opt.num_epochs; epoch++) {
        std::vector<float> approx_kl_divs;

        for (auto& batch : batches) {
            auto observations = batch.observations.to(opt.device);
            auto actions = batch.actions.to(opt.device);
            auto old_values = batch.old_values.to(opt.device);
            auto old_log_prob = batch.old_log_prob.to(opt.device);
            auto advantages = batch.advantages.to(opt.device);
            auto returns = batch.returns.to(opt.device);

            auto [action_dist, values] = policy->forward(observations);
            auto log_prob = action_dist.log_prob(values);
            auto entropy = action_dist.entropy();

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8f);

            auto ratio = torch::exp(log_prob - old_log_prob);

            auto policy_loss_1 = advantages * ratio;
            auto policy_loss_2 = advantages * torch::clamp(ratio, 1.f - opt.clip_range, 1.f + opt.clip_range);
            auto policy_loss = -torch::min(policy_loss_1, policy_loss_2).mean();

            pg_losses.push_back(policy_loss.item().toFloat());
            auto clip_fraction = torch::mean(
                    (torch::abs(ratio - 1.0) > opt.clip_range).toType(c10::ScalarType::Float));
            clip_fractions.push_back(clip_fraction.item().toFloat());

            torch::Tensor values_pred;
            if (opt.clip_range_vf_enabled) {
                values_pred = old_values + torch::clamp(values - old_values, -opt.clip_range_vf, opt.clip_range_vf);
            }
            else {
                values_pred = values;
            }
            auto value_loss = torch::mse_loss(returns, values_pred);
            value_losses.push_back(value_loss.item().toFloat());

            auto entropy_loss = torch::mean(entropy);

            auto loss = policy_loss + opt.ent_coef * entropy_loss + opt.vf_coef * value_loss;

            optimizer->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(policy->parameters(), opt.max_grad_norm);
            optimizer->step();
            approx_kl_divs.push_back(torch::mean(old_log_prob - log_prob).item().toFloat());
        }
        float kl_div_mean =
                std::accumulate(approx_kl_divs.begin(), approx_kl_divs.end(), 0.f) / approx_kl_divs.size();
        all_kl_divs.push_back(kl_div_mean);

        if (opt.target_kl_enabled && kl_div_mean > 1.5f * opt.target_kl) {
            printf("Early stopping at step %d due to reaching max kl: %.2f\n", epoch, kl_div_mean);
            break;
        }
    }

    // TODO: logging
}

}
