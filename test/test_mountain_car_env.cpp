//
// Created by lasagnaphil on 21. 1. 3..
//

#define USE_RENDERER

// #define USE_MPI
// #define USE_GLOO

#if defined(USE_MPI)
#include <c10d/ProcessGroupMPI.hpp>
#elif defined(USE_GLOO)
#include <c10d/ProcessGroupGloo.hpp>
#endif

#include "mountaincar_env.h"
#include "fastrl/fastrl.h"

int main(int argc, char** argv) {
    // google::InitGoogleLogging(argv[0]);
    // google::SetStderrLogging(google::ERROR);

    auto device = torch::kCPU;
    int num_envs = 20;
    bool eval_enabled = false;

    auto policy_opt = fastrl::PolicyOptions();
    policy_opt.action_dist_type = fastrl::DistributionType::DiagGaussian;
    policy_opt.actor_hidden_dim = {256, 256};
    policy_opt.critic_hidden_dim = {256, 256};
    policy_opt.activation_type = fastrl::NNActivationType::Tanh;
    policy_opt.log_std_init = -3.29f;
    // policy_opt.use_sde = true     TODO: Implement SDE
    policy_opt.ortho_init = false;
    policy_opt.device = device;

    auto rb_opt = fastrl::RolloutBufferOptions();
    rb_opt.gae_lambda = 0.9f;
    rb_opt.gamma = 0.9999f;
    rb_opt.buffer_size = 400;
    rb_opt.num_envs = num_envs;

    auto ppo_opt = fastrl::PPOOptions();
    ppo_opt.learning_rate = 7.77e-5f;
    ppo_opt.num_sgd_iters = 10;
    ppo_opt.ent_coeff = 0.00429f;
    ppo_opt.clip_range_vf_enabled = false;
    ppo_opt.device = device;

    int sgd_minibatch_size = 64;

    auto policy = std::make_shared<fastrl::Policy>(MountainCarEnv::obs_dim, MountainCarEnv::act_dim, policy_opt);
    auto rollout_buffer = fastrl::RolloutBuffer(MountainCarEnv::obs_dim, MountainCarEnv::act_dim, rb_opt);
#if defined(USE_MPI)
    auto process_group = c10d::ProcessGroupMPI::createProcessGroupMPI();
    auto logger = process_group->getRank() == 0? std::make_shared<TensorBoardLogger>("logs/tfevents.pb") : nullptr;
    auto ppo = fastrl::PPO(ppo_opt, policy, logger, process_group);
#elif defined(USE_GLOO)
    auto file_store = std::make_shared<c10d::Store>("tmp/c10d_gloo");
    auto process_group = c10d::ProcessGroupGloo(file_store, atoi(getenv("RANK")), atoi(getenv("SIZE")));
    auto logger = process_group->getRank() == 0? std::make_shared<TensorBoardLogger>("logs/tfevents.pb") : nullptr;
    auto ppo = fastrl::PPO(ppo_opt, policy, logger, process_group);
#else
    auto logger = std::make_shared<TensorBoardLogger>("logs/tfevents.pb");
    auto ppo = fastrl::PPO(ppo_opt, policy, logger);
#endif

    std::vector<MountainCarEnv> env(num_envs);
    MountainCarEnv eval_env;

    for (int e = 0; e < num_envs; e++) {
#if defined(USE_MPI) or defined(USE_GLOO)
        int seed = process_group->getRank() * num_envs + e;
#else
        int seed = e;
#endif
        env[e].seed(seed);
    }

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
                auto action_tensor  = action_dist->sample();
                float action = action_tensor.item<float>();
                float log_prob = action_dist->log_prob(obs_tensor).item<float>();
                auto [new_obs, reward, new_done] = env[e].step(action);
                rollout_buffer.add(e, obs_tensor, action_tensor, *action_dist, reward, done, value, log_prob);
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
        if (logger) {
            logger->add_scalar("train/avg_episode_reward", ppo.iter, avg_episode_reward);
        }
        std::printf("Average reward: %f\n", avg_episode_reward);

        auto batches = rollout_buffer.get_samples(sgd_minibatch_size);

        policy->train();
        ppo.train(batches.data(), batches.size());
        rollout_buffer.reset();

        if (logger && step % 10 == 0) {
            // Save model
            torch::serialize::OutputArchive output_archive;
            policy->save(output_archive);
            output_archive.save_to(std::string("policy_") + std::to_string(step) + ".pt");
        }
    }

    return 0;
}