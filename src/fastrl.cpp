//
// Created by lasagnaphil on 20. 12. 26..
//

#include "fastrl/fastrl.h"

#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/ProcessGroupMPI.hpp>

namespace fastrl {

torch::Tensor sum_independent_dims(torch::Tensor tensor) {
    if (tensor.sizes().size() > 1) {
        tensor = tensor.sum({1});
    }
    else {
        tensor = tensor.sum();
    }
    return tensor;
}

torch::Tensor orthogonal_(torch::Tensor tensor, double gain)
{
    torch::NoGradGuard guard;

    AT_ASSERT(tensor.ndimension() >= 2, "Only tensors with 2 or more dimensions are supported");

    const auto rows = tensor.size(0);
    const auto columns = tensor.numel() / rows;
    auto flattened = torch::randn({rows, columns});

    if (rows < columns)
    {
        flattened.t_();
    }

    // Compute the qr factorization
    torch::Tensor q, r;
    std::tie(q, r) = torch::qr(flattened);
    // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    auto d = torch::diag(r, 0);
    auto ph = d.sign();
    q *= ph;

    if (rows < columns)
    {
        q.t_();
    }

    tensor.view_as(q).copy_(q);
    tensor.mul_(gain);

    return tensor;
}

void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters,
                  double weight_gain,
                  double bias_gain)
{
    for (const auto &parameter : parameters)
    {
        if (parameter.value().size(0) != 0)
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                torch::nn::init::constant_(parameter.value(), bias_gain);
            }
            else if (parameter.key().find("weight") != std::string::npos)
            {
                orthogonal_(parameter.value(), weight_gain);
            }
        }
    }
}

DiagGaussianDistribution::DiagGaussianDistribution(torch::Tensor mean, torch::Tensor logstd) : mean(std::move(mean)), logstd(std::move(logstd)) {}

torch::Tensor DiagGaussianDistribution::entropy() {
    auto entropies = 0.5f + 0.5f * std::log(2 * M_PI) + logstd;
    return sum_independent_dims(entropies);
}

torch::Tensor DiagGaussianDistribution::log_prob(torch::Tensor value) const {
    auto log_probs = -(value - mean).pow(2) / (2 * logstd.exp().pow(2)) - logstd - std::log(std::sqrt(2 * M_PI));
    return sum_independent_dims(log_probs);
}

torch::Tensor DiagGaussianDistribution::sample(c10::ArrayRef<int64_t> sample_shape) const {
    auto no_grad_guard = torch::NoGradGuard();
    if (sample_shape.empty()) {
        return at::normal(mean, logstd.exp());
    }
    else {
        return at::normal(mean.expand(sample_shape), logstd.exp().expand(sample_shape));
    }
}

Policy::Policy(int state_size, int action_size, const PolicyOptions& options)
    : state_size(state_size), action_size(action_size), opt(options) {

    namespace nn = torch::nn;

    actor_seq_nn = nn::Sequential();
    for (int i = 0; i < opt.actor_hidden_dim.size() - 1; i++) {
        if (i == 0) {
            actor_seq_nn->push_back(nn::Linear(state_size, opt.actor_hidden_dim[i]));
        }
        else {
            actor_seq_nn->push_back(nn::Linear(opt.actor_hidden_dim[i], opt.actor_hidden_dim[i + 1]));
        }
        switch (opt.activation_type) {
            case NNActivationType::ReLU: actor_seq_nn->push_back(nn::ReLU()); break;
            case NNActivationType::Tanh: actor_seq_nn->push_back(nn::Tanh()); break;
        }
    }
    int last_hidden_dim = opt.actor_hidden_dim[opt.actor_hidden_dim.size()-1];
    actor_mu_nn = torch::nn::Linear(last_hidden_dim, action_size);
    actor_log_std = torch::ones({action_size}) * opt.log_std_init;

    critic_seq_nn = nn::Sequential();
    for (int i = 0; i < opt.critic_hidden_dim.size(); i++) {
        if (i < opt.critic_hidden_dim.size() - 1) {
            if (i == 0) {
                critic_seq_nn->push_back(nn::Linear(state_size, opt.critic_hidden_dim[i]));
            }
            else {
                critic_seq_nn->push_back(nn::Linear(opt.critic_hidden_dim[i], opt.critic_hidden_dim[i + 1]));
            }
            switch (opt.activation_type) {
                case NNActivationType::ReLU: critic_seq_nn->push_back(nn::ReLU()); break;
                case NNActivationType::Tanh: critic_seq_nn->push_back(nn::Tanh()); break;
            }
        }
        else {
            critic_seq_nn->push_back(nn::Linear(opt.critic_hidden_dim[i], 1));
        }
    }
    actor_seq_nn = register_module("actor_seq_nn", actor_seq_nn);
    actor_mu_nn = register_module("actor_mu_nn", actor_mu_nn);
    critic_seq_nn = register_module("critic_seq_nn", critic_seq_nn);

    if (!opt.fix_log_std) {
        actor_log_std = register_parameter("actor_log_std", actor_log_std);
    }

    this->to(opt.device);

    init_weights(actor_seq_nn->named_parameters(), std::sqrt(2.f), 0.f);
    init_weights(actor_mu_nn->named_parameters(), std::sqrt(2.f), 0.f);
    init_weights(critic_seq_nn->named_parameters(), std::sqrt(2.f), 0.f);
}

std::pair<DiagGaussianDistribution, torch::Tensor> Policy::forward(torch::Tensor state) {
    auto hidden = actor_seq_nn->forward(state);
    auto action_mu = actor_mu_nn->forward(hidden);
    torch::Tensor value = critic_seq_nn->forward(state);
    return {DiagGaussianDistribution{action_mu, actor_log_std}, value};
}

RolloutBuffer::RolloutBuffer(int state_size, int action_size, RolloutBufferOptions options)
    : state_size(state_size), action_size(action_size), opt(options), pos(opt.num_envs, 0),
      observations_data(torch::zeros(
              {opt.buffer_size, opt.num_envs, state_size})),
      actions_data(torch::zeros(
              {opt.buffer_size, opt.num_envs, action_size})),
      rewards_data(
              torch::zeros({opt.buffer_size, opt.num_envs})),
      returns_data(
              torch::zeros({opt.buffer_size, opt.num_envs})),
      dones_data(torch::zeros({opt.buffer_size, opt.num_envs}).toType(torch::kI8)),
      values_data(torch::zeros({opt.buffer_size, opt.num_envs})),
      log_probs_data(
              torch::zeros({opt.buffer_size, opt.num_envs})),
      advantages_data(
              torch::zeros({opt.buffer_size, opt.num_envs})),
      observations(observations_data.accessor<float, 3>()),
      actions(actions_data.accessor<float, 3>()),
      rewards(rewards_data.accessor<float, 2>()),
      returns(returns_data.accessor<float, 2>()),
      dones(dones_data.accessor<int8_t, 2>()),
      values(values_data.accessor<float, 2>()),
      log_probs(log_probs_data.accessor<float, 2>()),
      advantages(advantages_data.accessor<float, 2>()) {}

void RolloutBuffer::reset() {
    observations_data.zero_();
    actions_data.zero_();
    rewards_data.zero_();
    returns_data.zero_();
    dones_data.zero_();
    values_data.zero_();
    log_probs_data.zero_();
    advantages_data.zero_();
    for (int e = 0; e < opt.num_envs; e++) {
        pos[e] = 0;
    }
}

void RolloutBuffer::compute_returns_and_advantage(const float* last_values, const int8_t* last_dones) {
    std::vector<float> last_gae_lam(opt.num_envs, 0.0f);
    for (int step = opt.buffer_size - 1; step >= 0; step--) {
        for (int k = 0; k < opt.num_envs; k++) {
            float next_non_terminal, next_values;
            if (step == opt.buffer_size - 1) {
                next_non_terminal = last_dones[k] == 1? 0.0f : 1.0f;
                next_values = last_values[k];
            }
            else {
                next_non_terminal = dones[step + 1][k]? 0.0f : 1.0f;
                next_values = values[step + 1][k];
            }
            float delta = rewards[step][k] + opt.gamma * next_values * next_non_terminal - values[step][k];
            last_gae_lam[k] = delta + opt.gamma * opt.gae_lambda * next_non_terminal * last_gae_lam[k];
            advantages[step][k] = last_gae_lam[k];
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
    int p = pos[env_id];
    if (p == opt.buffer_size) {
        std::cerr << "Error in get_samples(): RolloutBuffer is full!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::copy_n(obs, state_size, observations[p][env_id].data());
    std::copy_n(action, action_size, actions[p][env_id].data());
    rewards[p][env_id] = reward;
    dones[p][env_id] = done? 1 : 0;
    values[p][env_id] = value;
    log_probs[p][env_id] = log_prob;
    pos[env_id]++;
}

std::vector<RolloutBufferBatch> RolloutBuffer::get_samples(int batch_size) {
    static std::default_random_engine random_engine;

    for (int e = 0; e < opt.num_envs; e++) {
        if (pos[e] != opt.buffer_size) {
            std::cerr << "Error in get_samples(): RolloutBuffer is not full yet!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::vector<int64_t> indices(opt.buffer_size * opt.num_envs);
    for (int64_t i = 0; i < indices.size(); i++) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), random_engine);

    std::vector<RolloutBufferBatch> batches;
    int start_idx = 0;
    while (start_idx < opt.buffer_size * opt.num_envs) {
        RolloutBufferBatch batch;
        batch.observations = torch::empty({batch_size, state_size});
        batch.actions = torch::empty({batch_size, action_size});
        batch.old_values = torch::empty({batch_size});
        batch.old_log_prob = torch::empty({batch_size});
        batch.advantages = torch::empty({batch_size});
        batch.returns = torch::empty({batch_size});

        auto batch_indices = c10::IntArrayRef(indices.data() + start_idx, batch_size);
        for (int k = 0; k < batch_size; k++) {
            int pos_idx = batch_indices[k] / opt.num_envs;
            int env_idx = batch_indices[k] % opt.num_envs;
            batch.observations[k] = observations_data[pos_idx][env_idx];
            batch.actions[k] = actions_data[pos_idx][env_idx];
            batch.old_values[k] = values_data[pos_idx][env_idx];
            batch.old_log_prob[k] = log_probs_data[pos_idx][env_idx];
            batch.advantages[k] = advantages_data[pos_idx][env_idx];
            batch.returns[k] = returns_data[pos_idx][env_idx];
        }
        start_idx += batch_size;
        batches.push_back(batch);
    }
    return batches;
}

float RolloutBuffer::get_average_episode_reward() {
    float average_episode_reward = 0.0f;
    int num_episodes = 0;
    for (int e = 0; e < opt.num_envs; e++) {
        float episode_reward = 0.0f;
        for (int step = 0; step < opt.buffer_size; step++) {
            episode_reward += rewards[step][e];
            if (dones[step][e] == 1 || step == opt.num_envs - 1) {
                average_episode_reward += episode_reward;
                episode_reward = 0.0f;
                num_episodes++;
            }
        }
    }
    average_episode_reward /= num_episodes;
    return average_episode_reward;
}

RolloutBuffer RolloutBuffer::merge(const RolloutBuffer* rbs, int num_rbs) {

    std::vector<torch::Tensor> observations(num_rbs), actions(num_rbs), rewards(num_rbs), advantages(num_rbs),
                               returns(num_rbs), dones(num_rbs), values(num_rbs), log_probs(num_rbs);

    int state_size = rbs[0].state_size;
    int action_size = rbs[0].action_size;
    RolloutBufferOptions opt = rbs->opt;
    opt.num_envs = 0;

    for (int i = 0; i < num_rbs; i++) {
        const RolloutBuffer& rb = rbs[i];
        assert(rb.state_size == state_size);
        assert(rb.action_size == action_size);
        assert(rb.opt.buffer_size == opt.buffer_size);
        assert(rb.opt.gae_lambda == opt.gae_lambda);
        assert(rb.opt.gamma == opt.gamma);
        opt.num_envs += rb.opt.num_envs;

        observations[i] = rb.observations_data;
        actions[i] = rb.actions_data;
        rewards[i] = rb.rewards_data;
        advantages[i] = rb.advantages_data;
        returns[i] = rb.returns_data;
        dones[i] = rb.dones_data;
        values[i] = rb.values_data;
        log_probs[i] = rb.log_probs_data;
    }

    RolloutBuffer res(state_size, action_size, opt);

    res.observations_data = torch::cat(observations, 1);
    res.actions_data = torch::cat(actions, 1);
    res.rewards_data = torch::cat(rewards, 1);
    res.advantages_data = torch::cat(advantages, 1);
    res.returns_data = torch::cat(returns, 1);
    res.dones_data = torch::cat(dones, 1);
    res.values_data = torch::cat(values, 1);
    res.log_probs_data = torch::cat(log_probs, 1);
    return res;
}

PPO::PPO(PPOOptions options, std::shared_ptr<Policy> policy, std::shared_ptr<TensorBoardLogger> logger)
        : opt(options), policy(policy), logger(logger) {

    optimizer = std::make_shared<torch::optim::Adam>(
            policy->parameters(), torch::optim::AdamOptions(opt.learning_rate));

    if (opt.use_distributed) {
        dist_rank = atoi(getenv("RANK"));
        dist_size = atoi(getenv("SIZE"));
        file_store = std::dynamic_pointer_cast<c10d::Store, c10d::FileStore>(
                std::make_shared<c10d::FileStore>("/tmp/fastrl", dist_size));
        switch (opt.dist_backend) {
            case DistributedBackend::Gloo:
                process_group = std::dynamic_pointer_cast<c10d::ProcessGroup, c10d::ProcessGroupGloo>(
                        std::make_shared<c10d::ProcessGroupGloo>(file_store, dist_rank, dist_size)); break;
            // TODO: Add support for other distributed backends later
            case DistributedBackend::MPI:
                process_group = std::dynamic_pointer_cast<c10d::ProcessGroup, c10d::ProcessGroupMPI>(
                        std::make_shared<c10d::ProcessGroupMPI>(file_store, dist_rank, dist_size)); break;
                fprintf(stderr, "Torch MPI not supported!\n");
                exit(EXIT_FAILURE);
                break;
            case DistributedBackend::NCCL:
                // process_group = std::make_shared<c10d::ProcessGroupNCCL>(file_store, rank, size); break;
                fprintf(stderr, "Torch NCCL not supported!\n");
                exit(EXIT_FAILURE);
        }
    }
}

float get_mean(const std::vector<float>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.f) / vec.size();
}

void PPO::train(const RolloutBufferBatch* batches, int num_batches) {
    // buffers for logging
    std::vector<float> entropy_losses, all_kl_divs, pg_losses, value_losses, clip_fractions;
    float loss_value;

    for (int sgd_iters = 0; sgd_iters < opt.num_epochs; sgd_iters++) {
        std::vector<float> approx_kl_divs;

        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            const RolloutBufferBatch& batch = batches[batch_idx];

            auto observations = batch.observations.to(opt.device);
            auto actions = batch.actions.to(opt.device);
            auto old_values = batch.old_values.to(opt.device);
            auto old_log_prob = batch.old_log_prob.to(opt.device);
            auto advantages = batch.advantages.to(opt.device);
            auto returns = batch.returns.to(opt.device);

            auto [action_dist, values] = policy->forward(observations);
            values = values.flatten();
            auto log_prob = action_dist.log_prob(observations);
            auto entropy = action_dist.entropy();

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8f);

            auto ratio = torch::exp(log_prob - old_log_prob);

            auto policy_loss_1 = advantages * ratio;
            auto policy_loss_2 = advantages * torch::clamp(ratio, 1.f - opt.clip_range, 1.f + opt.clip_range);
            auto policy_loss = -torch::min(policy_loss_1, policy_loss_2).mean();

            pg_losses.push_back(policy_loss.item<float>());
            auto clip_fraction = torch::mean(
                    (torch::abs(ratio - 1.f) > opt.clip_range).toType(torch::kFloat32));
            clip_fractions.push_back(clip_fraction.item<float>());

            torch::Tensor value_loss;
            if (opt.clip_range_vf_enabled) {
                auto values_clipped = old_values + torch::clamp(values - old_values, -opt.clip_range_vf, opt.clip_range_vf);
                auto value_loss_1 = torch::square(values - returns);
                auto value_loss_2 = torch::square(values_clipped - returns);
                value_loss = torch::mean(torch::maximum(value_loss_1, value_loss_2));
            }
            else {
                value_loss = torch::mean(torch::square(values - returns));
            }
            value_losses.push_back(value_loss.item<float>());
            // std::cout << "returns = " << returns << std::endl;
            // std::cout << "values_pred = " << values_pred << std::endl;

            torch::Tensor entropy_loss;
            if (opt.entropy_enabled) {
                entropy_loss = -torch::mean(entropy);
            }
            else {
                entropy_loss = -torch::mean(-log_prob);
            }
            entropy_losses.push_back(entropy_loss.item<float>());

            auto loss = policy_loss + opt.ent_coef * entropy_loss + opt.vf_coef * value_loss;
            loss_value = loss.item<float>();

            approx_kl_divs.push_back(torch::mean(old_log_prob - log_prob).item<float>());

            optimizer->zero_grad();
            loss.backward();

            if (opt.use_distributed) {
                auto params = policy->named_parameters();
                std::vector<torch::Tensor> grads(policy->named_parameters().size());
                std::transform(params.begin(), params.end(), grads.begin(), [](auto& item) {
                    return item.value().grad().alias();
                });

                auto grad_allreduce_work = process_group->allreduce(grads);
                grad_allreduce_work->wait();
                for (auto& grad : grads) {
                    grad.div_(dist_size);
                }
            }

            torch::nn::utils::clip_grad_norm_(policy->parameters(), opt.max_grad_norm);
            optimizer->step();
        }
        float kl_div_mean = get_mean(approx_kl_divs);
        all_kl_divs.push_back(kl_div_mean);

        if (opt.target_kl_enabled && kl_div_mean > 1.5f * opt.target_kl) {
            printf("Early stopping at step %d due to reaching max kl: %.2f\n", sgd_iters, kl_div_mean);
            break;
        }
    }

    logger->add_scalar("train/entropy_loss", iter, get_mean(entropy_losses));
    logger->add_scalar("train/policy_gradient_loss", iter, get_mean(pg_losses));
    logger->add_scalar("train/value_loss", iter, get_mean(value_losses));
    logger->add_scalar("train/approx_kl", iter, get_mean(all_kl_divs));
    logger->add_scalar("train/clip_fraction", iter, get_mean(clip_fractions));
    logger->add_scalar("train/loss", iter, loss_value);

    iter++;

    // std::vector<float> entropy_losses, all_kl_divs, pg_losses, value_losses, clip_fractions;
    std::printf("Iter results: \n");
    std::printf("entropy_loss=%.6f, kl_div_mean=%.6f, pg_loss=%.6f, value_loss=%.6f, clip_fraction=%.6f\n",
                get_mean(entropy_losses), get_mean(all_kl_divs), get_mean(pg_losses), get_mean(value_losses), get_mean(clip_fractions));
}

}
