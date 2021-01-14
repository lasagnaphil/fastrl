//
// Created by lasagnaphil on 20. 12. 26..
//

#ifndef FASTRL_FASTRL_H
#define FASTRL_FASTRL_H

#include <vector>
#include <random>

#include <torch/torch.h>
#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include "tensorboard_logger.h"
#include <mpi.h>

namespace fastrl {

enum class NNActivationType {
    ReLU, Tanh
};

struct Distribution {
    virtual torch::Tensor entropy() const = 0;
    virtual torch::Tensor log_prob(torch::Tensor value) const = 0;
    virtual torch::Tensor sample() const = 0;
    virtual void to_(torch::Device device) = 0;

    virtual ~Distribution() = default;
};

struct DiagGaussianDistribution : public Distribution {
    torch::Tensor mean;
    torch::Tensor logstd;

    DiagGaussianDistribution(torch::Tensor mean, torch::Tensor logstd);

    torch::Tensor entropy() const override;

    torch::Tensor log_prob(torch::Tensor value) const override;

    torch::Tensor sample() const override;

    void to_(torch::Device device) override { mean = mean.to(device); logstd = logstd.to(device); }
};

struct BernoulliDistribution : public Distribution {
    torch::Tensor logits;
    torch::Tensor probs;

    explicit BernoulliDistribution(torch::Tensor logits);

    torch::Tensor entropy() const;

    torch::Tensor log_prob(torch::Tensor value) const;

    torch::Tensor sample() const;

    void to_(torch::Device device) override { logits = logits.to(device); probs = probs.to(device); }
};

torch::Tensor kl_divergence(const Distribution& dist1, const Distribution& dist2);

enum class DistributionType {
    Bernoulli, DiagGaussian
};

struct PolicyOptions {
    DistributionType action_dist_type = DistributionType::DiagGaussian;

    std::vector<int> actor_hidden_dim = {256, 256};
    std::vector<int> critic_hidden_dim = {256, 256};
    NNActivationType activation_type = NNActivationType::ReLU;
    float log_std_init = 0.0f;
    bool fix_log_std = false;
    bool ortho_init = true;

    torch::Device device = torch::kCPU;
};

class Policy : public torch::nn::Module {
public:
    Policy(int state_size, int action_size, const PolicyOptions& options);

    std::pair<std::shared_ptr<Distribution>, torch::Tensor> forward(torch::Tensor state);

    int state_size, action_size;
    PolicyOptions opt;
    torch::nn::Sequential actor_seq_nn = nullptr;
    torch::nn::Sequential critic_seq_nn = nullptr;
    torch::nn::Linear actor_mu_nn = nullptr;
    torch::Tensor actor_log_std;
};

struct RunningMeanStd {
    torch::Tensor mean;
    torch::Tensor var;
    int state_size;
    int64_t count = 0;
    float clip_value;
    float epsilon;

    RunningMeanStd(int state_size, float clip_value = 10.0f, float epsilon = 1e-8f)
            : state_size(state_size), clip_value(clip_value), epsilon(epsilon) {
        mean = torch::zeros({state_size});
        var = torch::ones({state_size});
    }

    void update(torch::Tensor arr) {
        auto batch_mean = torch::mean(arr, {0});
        auto batch_var = torch::var(arr, 0);
        int64_t batch_count = arr.size(0);
        update_from_moments(batch_mean, batch_var, batch_count);
    }

    void update_from_moments(torch::Tensor batch_mean, torch::Tensor batch_var, int64_t batch_count) {
        auto delta = batch_mean - mean;
        int64_t tot_count = count + batch_count;

        mean = mean + delta * batch_count / tot_count;
        var = (var * count + batch_var * batch_count + delta.square() * count * batch_count / tot_count) / tot_count;
        count = tot_count;
    }

    torch::Tensor apply(torch::Tensor value) const {
        return torch::clip((value - mean) / (var.sqrt() + epsilon), -clip_value, clip_value);
    }
};

struct RolloutBufferOptions {
    int buffer_size = 2048;
    float gae_lambda = 1.0f;
    float gamma = 0.99f;
    int num_envs = 1;
    DistributionType action_dist_type = DistributionType::DiagGaussian;
};

struct RolloutBufferBatch {
    torch::Tensor observations, actions;
    std::shared_ptr<Distribution> actions_dist;
    torch::Tensor old_values, old_log_prob, advantages, returns;
};

class RolloutBuffer {
public:
    RolloutBuffer(int state_size, int action_size, RolloutBufferOptions options);

    void reset();

    void compute_returns_and_advantage(const float* last_values, const int8_t* last_dones);

    void add(int env_id, torch::Tensor obs, torch::Tensor action, const Distribution& action_dist,
             float reward, bool done, float value, float log_prob);

    std::vector<RolloutBufferBatch> get_samples(int batch_size);

    float get_average_episode_reward();
    float get_average_episode_length();

    static RolloutBuffer merge(const RolloutBuffer* rollout_buffers, int rollout_buffer_count);
    void normalize_observations(RunningMeanStd& obs_mstd);
    void normalize_rewards(RunningMeanStd& rew_mstd);

    int state_size, action_size;
    RolloutBufferOptions opt;

    torch::Tensor observations_data, actions_data;
    std::shared_ptr<Distribution> actions_dist_data;
    torch::Tensor rewards_data, advantages_data, returns_data, dones_data, values_data, log_probs_data;

    torch::TensorAccessor<float, 2> rewards, advantages,
            returns, values, log_probs;
    torch::TensorAccessor<int8_t, 2> dones;

    std::vector<int> pos;
};

enum class DistributedBackend {
    None, Gloo, MPI, NCCL
};

struct PPOOptions {
    int64_t max_timesteps = -1;
    int num_sgd_iters = 10;
    float learning_rate = 3e-4f;
    std::function<float(float)> learning_rate_schedule = nullptr;
    float clip_range = 0.2f;
    std::function<float(float)> clip_range_schedule = nullptr;
    bool clip_range_vf_enabled = false;
    float clip_range_vf = 0.0f;
    bool entropy_enabled = true;
    float kl_coeff = 0.0f;
    float vf_coeff = 0.5f;
    float ent_coeff = 0.0f;
    bool clip_grad_norm_enabled = false;
    float clip_grad_norm = 0.5f;
    bool target_kl_enabled = true;
    float target_kl = 0.01f;

    torch::Device device = torch::kCPU;
};

class PPO {
public:
    PPO(PPOOptions options, std::shared_ptr<Policy> policy, std::shared_ptr<TensorBoardLogger> logger);
    PPO(PPOOptions options, std::shared_ptr<Policy> policy, std::shared_ptr<TensorBoardLogger> logger,
        std::shared_ptr<c10d::ProcessGroup> process_group);

    void train(const RolloutBufferBatch* batches, int num_batches);

    PPOOptions opt;

    std::shared_ptr<Policy> policy;
    std::shared_ptr<torch::optim::Optimizer> optimizer;
    std::shared_ptr<TensorBoardLogger> logger;

    int iter = 0;
    int64_t num_timesteps = 0;
    float cur_kl_coeff;

    std::shared_ptr<c10d::ProcessGroup> process_group;
    DistributedBackend dist_backend = DistributedBackend::None;
    int dist_rank, dist_size;

};

}

#endif //FASTRL_FASTRL_H
