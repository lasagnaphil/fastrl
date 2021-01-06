//
// Created by lasagnaphil on 21. 1. 3..
//

#include "pendulum_env.h"
#include "fastrl/fastrl.h"

int main(int argc, char** argv) {
    auto policyOpt = fastrl::PolicyOptions();
    auto rbOpt = fastrl::RolloutBufferOptions();
    auto ppoOpt = fastrl::PPOOptions();

    auto policy = std::make_shared<fastrl::Policy>(3, 1, policyOpt);
    auto rollout_buffer = fastrl::RolloutBuffer(3, 1, rbOpt);
    auto ppo = fastrl::PPO(ppoOpt, policy);

    PendulumEnv env;
    auto obs = env.reset();
    auto done = false;

    auto device = torch::kCPU;

    int max_steps = 1000;
    for (int step = 1; step <= max_steps; step++) {
        for (int i = 0; i < rbOpt.buffer_size; i++) {
            torch::NoGradGuard guard {};
            auto obs_tensor = torch::from_blob(obs.data(), {(int)obs.size()}).to(device);
            auto [action_dist, value_tensor] = policy->forward(obs_tensor);
            float value = value_tensor.item<float>();
            float action = action_dist.sample().item<float>();
            float log_prob = action_dist.log_prob(value_tensor).item<float>();
            auto [new_obs, reward, new_done, info] = env.step(action);
            rollout_buffer.add(0, obs.data(), &action, reward, done, value, log_prob);

            obs = new_obs;
            done = new_done;
        }

        {
            torch::NoGradGuard guard {};
            auto obs_tensor = torch::from_blob(obs.data(), obs.size()).to(device);
            auto [_, value_tensor] = policy->forward(obs_tensor);
            float value = value_tensor.item<float>();
            rollout_buffer.compute_returns_and_advantage(&value, &done);
        }

        auto batches = rollout_buffer.get_samples(128);

        ppo.train(batches.data(), batches.size());
        rollout_buffer.reset();

        // Evaluation
        int eval_trials = 300;
        float avg_reward = 0.0f;
        for (int i = 0; i < eval_trials; i++) {
            torch::NoGradGuard guard {};
            auto obs_tensor = torch::from_blob(obs.data(), {(int)obs.size()}).to(device);
            auto [action_dist, value_tensor] = policy->forward(obs_tensor);
            float value = value_tensor.item<float>();
            float action = action_dist.sample().item<float>();
            float log_prob = action_dist.log_prob(value_tensor).item<float>();
            auto [new_obs, reward, new_done, info] = env.step(action);
            avg_reward += reward;
            obs = new_obs;
            done = new_done;
        }
        avg_reward /= eval_trials;
        std::printf("Average reward: %f\n", avg_reward);
    }

    return 0;
}