//
// Created by lasagnaphil on 20. 12. 26..
//

#include "fastrl/fastrl.h"
#include "fastrl/utils.h"

#if defined(FASTRL_MPI)
#include <c10d/ProcessGroupMPI.hpp>
#endif
#if defined(FASTRL_GLOO)
#include <c10d/ProcessGroupGloo.hpp>
#endif
#if defined(FASTRL_NCCL)
#include <c10d/ProcessGroupNCCL.hpp>
#endif

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

torch::Tensor DiagGaussianDistribution::entropy() const {
    auto entropies = 0.5f + 0.5f * std::log(2 * M_PI) + logstd;
    return sum_independent_dims(entropies);
}

torch::Tensor DiagGaussianDistribution::log_prob(torch::Tensor value) const {
    auto log_probs = -(value - mean).pow(2) / (2 * logstd.exp().pow(2)) - logstd - std::log(std::sqrt(2 * M_PI));
    return sum_independent_dims(log_probs);
}

torch::Tensor DiagGaussianDistribution::sample() const {
    return at::normal(mean, logstd.exp());
}

BernoulliDistribution::BernoulliDistribution(torch::Tensor logits) : logits(logits) {
    probs = torch::sigmoid(logits);
}

torch::Tensor BernoulliDistribution::entropy() const {
    auto entropies = torch::binary_cross_entropy_with_logits(logits, probs, {}, {}, torch::Reduction::None);
    return sum_independent_dims(entropies);
}

torch::Tensor BernoulliDistribution::log_prob(torch::Tensor value) const {
    auto log_probs = -torch::binary_cross_entropy_with_logits(logits, value, {}, {}, torch::Reduction::None);
    return sum_independent_dims(log_probs);
}

torch::Tensor BernoulliDistribution::sample() const {
    torch::NoGradGuard guard {};
    return torch::bernoulli(probs);
}

torch::Tensor kl_divergence(const Distribution &dist1, const Distribution &dist2) {
    if (auto d1 = dynamic_cast<const DiagGaussianDistribution*>(&dist1)) {
        if (auto d2 = dynamic_cast<const DiagGaussianDistribution*>(&dist2)) {
            torch::Tensor std1 = d1->logstd.exp();
            torch::Tensor std2 = d2->logstd.exp();
            torch::Tensor kl1 = d2->logstd - d1->logstd;
            torch::Tensor kl2 = 0.5f * (std1.square() + (d2->mean - d1->mean).square()) / std2.square();
            return sum_independent_dims(kl1 + kl2 - 0.5f);
        }
    }
    if (auto d1 = dynamic_cast<const BernoulliDistribution*>(&dist1)) {
        if (auto d2 = dynamic_cast<const BernoulliDistribution*>(&dist2)) {
            auto invprobs1 = 1.0f - d1->probs;
            auto invprobs2 = 1.0f - d2->probs;
            auto kl = d1->probs * (d1->probs.log() - d2->probs.log()) + invprobs1 * (invprobs1.log() - invprobs2.log());
            return sum_independent_dims(kl);
        }
    }
    fprintf(stderr, "Unimplemented kl_divergence between two distributions...\n");
    exit(EXIT_FAILURE);
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

    if (opt.action_dist_type == DistributionType::DiagGaussian) {
        if (opt.fix_log_std) {
            actor_log_std = register_buffer("actor_log_std", actor_log_std);
        }
        else {
            actor_log_std = register_parameter("actor_log_std", actor_log_std);
        }
    }

    if (opt.ortho_init) {
        init_weights(actor_seq_nn->named_parameters(), std::sqrt(2.f), 0.f);
        init_weights(actor_mu_nn->named_parameters(), std::sqrt(2.f), 0.f);
        init_weights(critic_seq_nn->named_parameters(), std::sqrt(2.f), 0.f);
    }

    this->to(opt.device);
}

std::pair<std::shared_ptr<Distribution>, torch::Tensor> Policy::forward(torch::Tensor state) {
    auto hidden = actor_seq_nn->forward(state);
    auto action_mu = actor_mu_nn->forward(hidden);
    torch::Tensor value = critic_seq_nn->forward(state);
    switch (opt.action_dist_type) {
        case DistributionType::Bernoulli:
            return {std::make_shared<BernoulliDistribution>(action_mu), value};
        case DistributionType::DiagGaussian:
            return {std::make_shared<DiagGaussianDistribution>(action_mu, actor_log_std), value};
        default:
            return {nullptr, value};
    }
}

RolloutBuffer::RolloutBuffer(int state_size, int action_size, RolloutBufferOptions options)
    : state_size(state_size), action_size(action_size), opt(options), pos(opt.num_envs, 0),
      observations_data(torch::zeros({opt.buffer_size, opt.num_envs, state_size})),
      actions_data(torch::zeros({opt.buffer_size, opt.num_envs, action_size})),
      rewards_data(torch::zeros({opt.buffer_size, opt.num_envs})),
      returns_data(torch::zeros({opt.buffer_size, opt.num_envs})),
      dones_data(torch::zeros({opt.buffer_size, opt.num_envs}).toType(torch::kI8)),
      values_data(torch::zeros({opt.buffer_size, opt.num_envs})),
      log_probs_data(torch::zeros({opt.buffer_size, opt.num_envs})),
      advantages_data(torch::zeros({opt.buffer_size, opt.num_envs})),
      rewards(rewards_data.accessor<float, 2>()),
      returns(returns_data.accessor<float, 2>()),
      dones(dones_data.accessor<int8_t, 2>()),
      values(values_data.accessor<float, 2>()),
      log_probs(log_probs_data.accessor<float, 2>()),
      advantages(advantages_data.accessor<float, 2>()) {

    if (opt.action_dist_type == DistributionType::DiagGaussian) {
        actions_dist_data = std::make_shared<DiagGaussianDistribution>(
                torch::zeros({opt.buffer_size, opt.num_envs, action_size}),
                torch::zeros({opt.buffer_size, opt.num_envs, action_size}));
    }
    else if (opt.action_dist_type == DistributionType::Bernoulli) {
        actions_dist_data = std::make_shared<BernoulliDistribution>(
                torch::zeros({opt.buffer_size, opt.num_envs, action_size}));
    }
}

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
    if (opt.action_dist_type == DistributionType::DiagGaussian) {
        auto dist = dynamic_cast<const DiagGaussianDistribution*>(actions_dist_data.get());
        dist->mean.zero_();
        dist->logstd.zero_();
    }
    else if (opt.action_dist_type == DistributionType::Bernoulli) {
        auto dist = dynamic_cast<const BernoulliDistribution*>(actions_dist_data.get());
        dist->logits.zero_();
        dist->probs.zero_();
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

void RolloutBuffer::add(int env_id, torch::Tensor obs, torch::Tensor action, const Distribution& action_dist,
                        float reward, bool done, float value, float log_prob) {
    int p = pos[env_id];
    if (p == opt.buffer_size) {
        std::cerr << "Error in get_samples(): RolloutBuffer is full!" << std::endl;
        exit(EXIT_FAILURE);
    }
    observations_data[p][env_id] = std::move(obs);
    actions_data[p][env_id] = std::move(action);
    if (opt.action_dist_type == DistributionType::DiagGaussian) {
        auto dist = dynamic_cast<const DiagGaussianDistribution*>(&action_dist);
        auto data = dynamic_cast<const DiagGaussianDistribution*>(actions_dist_data.get());
        data->mean[p][env_id] = dist->mean;
        data->logstd[p][env_id] = dist->logstd;
    }
    else if (opt.action_dist_type == DistributionType::Bernoulli) {
        auto dist = dynamic_cast<const BernoulliDistribution*>(&action_dist);
        auto data = dynamic_cast<const BernoulliDistribution*>(actions_dist_data.get());
        data->logits[p][env_id] = dist->logits;
        data->probs[p][env_id] = dist->probs;
    }
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
        if (opt.action_dist_type == DistributionType::DiagGaussian) {
            batch.actions_dist = std::make_shared<DiagGaussianDistribution>(
                    torch::zeros({batch_size, action_size}),
                    torch::zeros({batch_size, action_size}));
        }
        else if (opt.action_dist_type == DistributionType::Bernoulli) {
            batch.actions_dist = std::make_shared<BernoulliDistribution>(
                    torch::zeros({batch_size, action_size}));
        }

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
        if (opt.action_dist_type == DistributionType::DiagGaussian) {
            auto dist = dynamic_cast<DiagGaussianDistribution *>(batch.actions_dist.get());
            auto data = dynamic_cast<DiagGaussianDistribution *>(actions_dist_data.get());
            for (int k = 0; k < batch_size; k++) {
                int pos_idx = batch_indices[k] / opt.num_envs;
                int env_idx = batch_indices[k] % opt.num_envs;
                dist->mean[k] = data->mean[pos_idx][env_idx];
                dist->logstd[k] = data->logstd[pos_idx][env_idx];
            }
        }
        else if (opt.action_dist_type == DistributionType::Bernoulli) {
            auto dist = dynamic_cast<BernoulliDistribution*>(batch.actions_dist.get());
            auto data = dynamic_cast<BernoulliDistribution*>(actions_dist_data.get());
            for (int k = 0; k < batch_size; k++) {
                int pos_idx = batch_indices[k] / opt.num_envs;
                int env_idx = batch_indices[k] % opt.num_envs;
                dist->logits[k] = data->logits[pos_idx][env_idx];
                dist->probs[k] = data->probs[pos_idx][env_idx];
            }
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
            if (dones[step][e] == 1 || step == opt.buffer_size - 1) {
                average_episode_reward += episode_reward;
                episode_reward = 0.0f;
                num_episodes++;
            }
        }
    }
    average_episode_reward /= num_episodes;
    return average_episode_reward;
}

float RolloutBuffer::get_average_episode_length() {
    float average_episode_length = 0.0f;
    int num_episodes = 0;
    for (int e = 0; e < opt.num_envs; e++) {
        int episode_length = 0;
        for (int step = 0; step < opt.buffer_size; step++) {
            episode_length++;
            if (dones[step][e] == 1 || step == opt.buffer_size - 1) {
                average_episode_length += (float)episode_length;
                episode_length = 0;
                num_episodes++;
            }
        }
    }
    average_episode_length /= num_episodes;
    return average_episode_length;
}

RolloutBuffer RolloutBuffer::merge(const RolloutBuffer* rbs, int num_rbs) {

    std::vector<torch::Tensor> observations(num_rbs), actions(num_rbs), rewards(num_rbs), advantages(num_rbs),
                               returns(num_rbs), dones(num_rbs), values(num_rbs), log_probs(num_rbs);
    std::vector<torch::Tensor> actions_mean(num_rbs), actions_logstd(num_rbs), actions_logits(num_rbs);

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
        if (opt.action_dist_type == DistributionType::DiagGaussian) {
            auto data = dynamic_cast<DiagGaussianDistribution*>(rb.actions_dist_data.get());
            actions_mean[i] = data->mean;
            actions_logstd[i] = data->logstd;
        }
        else if (opt.action_dist_type == DistributionType::Bernoulli) {
            auto data = dynamic_cast<BernoulliDistribution*>(rb.actions_dist_data.get());
            actions_logits[i] = data->logits;
        }
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
    if (opt.action_dist_type == DistributionType::DiagGaussian) {
        res.actions_dist_data = std::make_shared<DiagGaussianDistribution>(
                torch::cat(actions_mean, 1), torch::cat(actions_logstd, 1));
    }
    else if (opt.action_dist_type == DistributionType::Bernoulli) {
        res.actions_dist_data = std::make_shared<BernoulliDistribution>(
                torch::cat(actions_logits, 1));
    }
    return res;
}

void RolloutBuffer::normalize_observations(RunningMeanStd& obs_mstd) {
    auto obs_stack = observations_data.view({opt.buffer_size * opt.num_envs, -1});
    obs_mstd.update(obs_stack);
    observations_data = obs_mstd.apply(observations_data);
}

void RolloutBuffer::normalize_rewards(RunningMeanStd& rew_mstd) {
    rew_mstd.update(rewards_data);
    rewards_data = rew_mstd.apply(rewards_data);
    rewards = rewards_data.accessor<float, 2>();
}

PPO::PPO(PPOOptions options, std::shared_ptr<Policy> policy, std::shared_ptr<TensorBoardLogger> logger)
        : opt(options), policy(std::move(policy)), logger(std::move(logger)) {

    optimizer = std::make_shared<torch::optim::Adam>(
            this->policy->parameters(), torch::optim::AdamOptions(opt.learning_rate));

    dist_backend = DistributedBackend::None;
    cur_kl_coeff = opt.kl_coeff;
}

PPO::PPO(PPOOptions options, std::shared_ptr<Policy> policy, std::shared_ptr<TensorBoardLogger> logger,
         std::shared_ptr<c10d::ProcessGroup> process_group)
         : PPO(options, std::move(policy), std::move(logger)) {

#if defined(FASTRL_MPI)
    if (dynamic_cast<c10d::ProcessGroupMPI*>(process_group.get())) {
        dist_backend = DistributedBackend::MPI;
    }
#endif
#if defined(FASTRL_GLOO)
    if (dynamic_cast<c10d::ProcessGroupGloo*>(process_group.get())) {
        dist_backend = DistributedBackend::Gloo;
    }
#endif
#if defined(FASTRL_NCCL)
    if (dynamic_cast<c10d::ProcessGroupNCCL*>(process_group.get())) {
        dist_backend = DistributedBackend::NCCL;
    }
#endif

    this->process_group = process_group;

    dist_rank = process_group->getRank();
    dist_size = process_group->getSize();
}

float get_mean(const std::vector<float>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.f) / vec.size();
}

void PPO::train(const RolloutBufferBatch* batches, int num_batches) {
    // buffers for logging
    std::vector<float> entropy_losses, action_kl_losses, pg_losses, value_losses, clip_fractions;
    float loss_value;

    if (opt.learning_rate_schedule) {
        if (opt.max_timesteps == -1) {
            fprintf(stderr, "To use scheduling options, you need to set max_timesteps option!\n");
            exit(EXIT_FAILURE);
        }
        float remaining_progress = std::max(1.0f - (float)num_timesteps / (float)opt.max_timesteps, 0.0f);
        float cur_learning_rate = opt.learning_rate_schedule(remaining_progress);
        for (auto& param_group : optimizer->param_groups()) {
            dynamic_cast<torch::optim::AdamOptions&>(param_group.options()).lr(cur_learning_rate);
        }
        logger->add_scalar("train/learning_rate", iter, cur_learning_rate);
    }

    float cur_clip_range = opt.clip_range;
    if (opt.clip_range_schedule) {
        if (opt.max_timesteps == -1) {
            fprintf(stderr, "To use scheduling options, you need to set max_timesteps option!\n");
            exit(EXIT_FAILURE);
        }
        float remaining_progress = std::max(1.0f - (float)num_timesteps / (float)opt.max_timesteps, 0.0f);
        cur_clip_range = opt.clip_range_schedule(remaining_progress);
        logger->add_scalar("train/clip_range", iter, cur_clip_range);
    }

    for (int sgd_iters = 0; sgd_iters < opt.num_sgd_iters; sgd_iters++) {
        std::vector<float> action_kl_loss;

        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            const RolloutBufferBatch& batch = batches[batch_idx];

            auto observations = batch.observations.to(opt.device);
            auto actions = batch.actions.to(opt.device);
            auto old_values = batch.old_values.to(opt.device);
            auto old_log_prob = batch.old_log_prob.to(opt.device);
            auto advantages = batch.advantages.to(opt.device);
            auto returns = batch.returns.to(opt.device);
            auto old_action_dist = batch.actions_dist;
            old_action_dist->to_(opt.device);

            auto [action_dist, values] = policy->forward(observations);
            values = values.flatten();
            auto log_prob = action_dist->log_prob(actions);
            auto entropy = action_dist->entropy();

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8f);

            auto ratio = torch::exp(log_prob - old_log_prob);

            auto policy_loss_1 = advantages * ratio;
            auto policy_loss_2 = advantages * torch::clamp(ratio, 1.f - cur_clip_range, 1.f + cur_clip_range);
            auto policy_loss = -torch::min(policy_loss_1, policy_loss_2).mean();

            pg_losses.push_back(policy_loss.item<float>());
            auto clip_fraction = torch::mean(
                    (torch::abs(ratio - 1.f) > cur_clip_range).toType(torch::kFloat32));
            clip_fractions.push_back(clip_fraction.item<float>());

            auto action_kl = kl_divergence(*old_action_dist, *action_dist);
            auto action_loss = action_kl.mean();

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

            auto loss = policy_loss + cur_kl_coeff * action_loss + opt.vf_coeff * value_loss + opt.ent_coeff * entropy_loss;
            loss_value = loss.item<float>();

            action_kl_loss.push_back(action_loss.item<float>());

            optimizer->zero_grad();
            loss.backward();

            auto params = policy->parameters();
            if (dist_backend != DistributedBackend::None) {
                std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> works;
                for (auto& param : params) {
                    std::vector<torch::Tensor> tmp = {param.grad().data()};
                    works.push_back(std::move(process_group->allreduce(tmp)));
                }

                for (auto& work : works) {
                    try {
                        work->wait();
                    } catch (const std::exception& ex) {
                        std::cerr << "Exception received: " << ex.what() << std::endl;
#if defined(FASTRL_MPI)
                        if (dist_backend == DistributedBackend::MPI) {
                            dynamic_cast<c10d::ProcessGroupMPI*>(process_group.get())->abort();
                        }
#endif
#if defined(FASTRL_GLOO)
                        if (dist_backend == DistributedBackend::Gloo) {
                            exit(EXIT_FAILURE);
                        }
#endif
#if defined(FASTRL_NCCL)
                        if (dist_backend == DistributedBackend::NCCL) {
                            exit(EXIT_FAILURE);
                        }
#endif
                    }
                }
                for (auto& param : params) {
                    param.grad().div_(dist_size);
                }
            }

            if (opt.clip_grad_norm_enabled) {
                torch::nn::utils::clip_grad_norm_(params, opt.clip_grad_norm);
            }
            optimizer->step();
        }
        float action_kl_loss_this_iter = get_mean(action_kl_loss);
        action_kl_losses.push_back(action_kl_loss_this_iter);

        if (opt.target_kl_enabled) {
            if (action_kl_loss_this_iter > 2.0f * opt.target_kl) {
                cur_kl_coeff *= 1.5f;
            }
            else if (action_kl_loss_this_iter < 0.5f * opt.target_kl) {
                cur_kl_coeff *= 0.5f;
            }
        }
    }
    num_timesteps += num_batches * batches[0].returns.size(0);

    if (logger) {
        logger->add_scalar("train/entropy_loss", iter, get_mean(entropy_losses));
        logger->add_scalar("train/policy_gradient_loss", iter, get_mean(pg_losses));
        logger->add_scalar("train/action_kl_loss", iter, get_mean(action_kl_losses));
        logger->add_scalar("train/value_loss", iter, get_mean(value_losses));
        logger->add_scalar("train/loss", iter, loss_value);
        logger->add_scalar("train/cur_kl_coeff", iter, cur_kl_coeff);
        logger->add_scalar("train/clip_fraction", iter, get_mean(clip_fractions));
    }

    iter++;

    std::printf("Iter results: \n");
    std::printf("entropy_loss=%.6f, action_kl_loss=%.6f, kl_coeff=%.6f, "
                "pg_loss=%.6f, value_loss=%.6f, clip_fraction=%.6f\n",
                get_mean(entropy_losses), get_mean(action_kl_losses), cur_kl_coeff,
                get_mean(pg_losses), get_mean(value_losses), get_mean(clip_fractions));
}


}
