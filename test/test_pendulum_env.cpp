//
// Created by lasagnaphil on 21. 1. 3..
//

#include "pendulum_env.h"
#include "fastrl/fastrl.h"

int main(int argc, char** argv) {
    // google::InitGoogleLogging(argv[0]);
    // google::SetStderrLogging(google::ERROR);

    auto device = torch::kCPU;
    int num_envs = 8;
    bool eval_enabled = false;

    auto policy_opt = fastrl::PolicyOptions();
    policy_opt.actor_hidden_dim = {64, 64};
    policy_opt.critic_hidden_dim = {64, 64};
    policy_opt.activation_type = fastrl::NNActivationType::Tanh;
    policy_opt.device = device;

    auto rb_opt = fastrl::RolloutBufferOptions();
    rb_opt.gae_lambda = 0.95f;
    rb_opt.gamma = 0.99f;
    rb_opt.buffer_size = 2048;
    rb_opt.num_envs = num_envs;

    auto ppo_opt = fastrl::PPOOptions();
    ppo_opt.learning_rate = 3e-4f;
    ppo_opt.num_epochs = 10;
    ppo_opt.ent_coef = 0.0f;
    ppo_opt.device = device;

    int sgd_minibatch_size = 64;

    auto logger = std::make_shared<TensorBoardLogger>("logs/tfevents.pb");
    auto policy = std::make_shared<fastrl::Policy>(3, 1, policy_opt);
    auto rollout_buffer = fastrl::RolloutBuffer(3, 1, rb_opt);
    auto ppo = fastrl::PPO(ppo_opt, policy, logger);

    std::vector<PendulumEnv> env(num_envs);
    PendulumEnv eval_env;

    int max_steps = 1000;
    for (int step = 1; step <= max_steps; step++) {
        std::vector<float> last_values(num_envs);
        std::vector<int8_t> last_dones(num_envs);

        policy->eval();
        for (int e = 0; e < num_envs; e++) {
            auto obs = env[e].reset();
            bool done;
            for (int i = 0; i < rb_opt.buffer_size; i++) {
                auto obs_tensor = torch::from_blob(obs.data(), {(int)obs.size()}).to(device);
                auto [action_dist, value_tensor] = policy->forward(obs_tensor);
                float value = value_tensor.item<float>();
                float action = action_dist.sample().item<float>();
                float log_prob = action_dist.log_prob(obs_tensor).item<float>();
                auto [new_obs, reward, new_done] = env[e].step(action);
                rollout_buffer.add(e, obs.data(), &action, reward, done, value, log_prob);
                obs = new_obs;
                done = new_done;
                if (done) {
                    obs = env[e].reset();
                }
            }

            {
                torch::NoGradGuard guard {};
                auto obs_tensor = torch::from_blob(obs.data(), obs.size()).to(device);
                auto [_, value_tensor] = policy->forward(obs_tensor);
                last_values[e] = value_tensor.item<float>();
                last_dones[e] = (int8_t)done;
            }
        }

        rollout_buffer.compute_returns_and_advantage(last_values.data(), last_dones.data());
        float avg_episode_reward = rollout_buffer.get_average_episode_reward();
        logger->add_scalar("train/avg_episode_reward", ppo.iter, avg_episode_reward);
        std::printf("Average reward: %f\n", avg_episode_reward);

        auto batches = rollout_buffer.get_samples(sgd_minibatch_size);

        policy->train();
        ppo.train(batches.data(), batches.size());
        rollout_buffer.reset();

        // Evaluation
        if (eval_enabled) {
            policy->eval();
            auto obs = eval_env.reset();
            auto done = false;
            int num_episodes = 0;
            float avg_episode_reward = 0.0f;
            float episode_reward = 0.0f;
            for (int i = 0; i < eval_env.max_time; i++) {
                torch::NoGradGuard guard {};
                auto obs_tensor = torch::from_blob(obs.data(), {(int)obs.size()}).to(device);
                auto [action_dist, value_tensor] = policy->forward(obs_tensor);
                float action = action_dist.sample().item<float>();
                auto [new_obs, reward, new_done] = eval_env.step(action);
                std::printf("State: [%f %f %f]\n", obs[0], obs[1], obs[2]);
                std::printf("Reward: %f\n", reward);
                episode_reward += reward;
                obs = new_obs;
                done = new_done;
                if (done || i == eval_env.max_time - 1) {
                    num_episodes++;
                    avg_episode_reward += episode_reward;
                    episode_reward = 0.0f;
                    eval_env.reset();
                }
            }
            avg_episode_reward /= num_episodes;
            std::printf("Average eval reward: %f\n", avg_episode_reward);
        }

    }

    return 0;
}