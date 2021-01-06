//
// Created by lasagnaphil on 20. 12. 26..
//

#ifndef FASTRL_FASTRL_H
#define FASTRL_FASTRL_H

#include <vector>
#include <random>
#include <torch/torch.h>

namespace fastrl {

enum class NNActivationType {
    ReLU, Tanh
};

struct NormalDistribution {
    torch::Tensor mu;
    torch::Tensor sigma;

    NormalDistribution(torch::Tensor mu, torch::Tensor sigma);

    torch::Tensor entropy();

    torch::Tensor log_prob(torch::Tensor value) const;

    torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) const;

};

struct PolicyOptions {
    std::vector<int> actor_hidden_dim = {256, 256};
    std::vector<int> critic_hidden_dim = {256, 256};
    NNActivationType activation_type = NNActivationType::ReLU;
};

class Policy : public torch::nn::Module {
public:
    Policy(int state_size, int action_size, const PolicyOptions& options);

    std::pair<NormalDistribution, torch::Tensor> forward(torch::Tensor state);

    int state_size, action_size;
    PolicyOptions opt;
    torch::nn::Sequential actor = nullptr;
    torch::nn::Sequential critic = nullptr;
};

struct RolloutBufferOptions {
    int buffer_size = 2048;
    float gae_lambda = 1.0f;
    float gamma = 0.99f;
    int num_envs = 1;
};

struct RolloutBufferBatch {
    torch::Tensor observations, actions, old_values, old_log_prob, advantages, returns;
};

class RolloutBuffer {
public:
    RolloutBuffer(int state_size, int action_size, RolloutBufferOptions options);

    void reset();

    void compute_returns_and_advantage(const float* last_values, const bool* last_dones);

    void add(int env_id, const float* obs, const float* action, float reward,
             bool done, float value, float log_prob);

    std::vector<RolloutBufferBatch> get_samples(int batch_size);

    static RolloutBuffer merge(const RolloutBuffer* rollout_buffers, int rollout_buffer_count);

    int state_size, action_size;
    RolloutBufferOptions opt;

    torch::Tensor observations_data, actions_data, rewards_data, advantages_data,
            returns_data, dones_data, values_data, log_probs_data;

    torch::TensorAccessor<float, 2> rewards, advantages,
            returns, dones, values, log_probs;
    torch::TensorAccessor<float, 3> observations, actions;

    int pos = 0;
    bool full = false;
};

struct PPOOptions {
    int num_epochs = 10;
    float learning_rate = 3e-4f;
    float clip_range = 0.2f;
    bool clip_range_vf_enabled = false;
    float clip_range_vf = 0.0f;
    float ent_coef = 0.0f;
    float vf_coef = 0.5f;
    float max_grad_norm = 0.5f;
    bool target_kl_enabled = false;
    float target_kl = 0.0f;

    torch::Device device = torch::kCPU;
};

class PPO {
public:
    PPO(PPOOptions options, std::shared_ptr<Policy> policy);

    void train(const RolloutBufferBatch* batches, int num_batches);

    PPOOptions opt;

    std::shared_ptr<Policy> policy;
    std::shared_ptr<torch::optim::Optimizer> optimizer;
};

}

#endif //FASTRL_FASTRL_H
