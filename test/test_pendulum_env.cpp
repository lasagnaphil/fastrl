//
// Created by lasagnaphil on 21. 1. 3..
//

// #define USE_EVALUATION

#define USE_MPI
// #define USE_GLOO

#if defined(USE_MPI)
#include <c10d/ProcessGroupMPI.hpp>
#elif defined(USE_GLOO)
#include <c10d/ProcessGroupGloo.hpp>
#endif

#include "pendulum_env.h"
#include "fastrl/fastrl.h"
#ifdef USE_EVALUATION
#include <raylib.h>
#endif

int main(int argc, char** argv) {
    // google::InitGoogleLogging(argv[0]);
    // google::SetStderrLogging(google::ERROR);

    auto device = torch::kCPU;
    int num_envs = 20;
    bool eval_enabled = true;

    auto policy_opt = fastrl::PolicyOptions();
    policy_opt.actor_hidden_dim = {256, 256};
    policy_opt.critic_hidden_dim = {256, 256};
    policy_opt.activation_type = fastrl::NNActivationType::Tanh;
    policy_opt.device = device;

    auto rb_opt = fastrl::RolloutBufferOptions();
    rb_opt.gae_lambda = 0.1f;
    rb_opt.gamma = 0.95f;
    rb_opt.buffer_size = 512;
    rb_opt.num_envs = num_envs;

    auto ppo_opt = fastrl::PPOOptions();
    ppo_opt.learning_rate = 1e-5f;
    ppo_opt.num_epochs = 6;
    ppo_opt.ent_coef = 0.0f;
    ppo_opt.clip_range_vf_enabled = true;
    ppo_opt.clip_range_vf = 100.0f;
    ppo_opt.device = device;

    int sgd_minibatch_size = 64;

    torch::manual_seed(0);

    auto logger = std::make_shared<TensorBoardLogger>("logs/tfevents.pb");
    auto policy = std::make_shared<fastrl::Policy>(3, 1, policy_opt);
    auto rollout_buffer = fastrl::RolloutBuffer(3, 1, rb_opt);
#if defined(USE_MPI)
    auto process_group = c10d::ProcessGroupMPI::createProcessGroupMPI();
    auto ppo = fastrl::PPO(ppo_opt, policy, logger, process_group);
#elif defined(USE_GLOO)
    auto file_store = std::make_shared<c10d::Store>("tmp/c10d_gloo");
    auto process_group = c10d::ProcessGroupGloo(file_store, atoi(getenv("RANK")), atoi(getenv("SIZE")));
    auto ppo = fastrl::PPO(ppo_opt, policy, logger, process_group);
#else
    auto ppo = fastrl::PPO(ppo_opt, policy, logger);
#endif

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

        if (step % 10 == 0) {
            // Save model
            torch::serialize::OutputArchive output_archive;
            policy->save(output_archive);
            output_archive.save_to(std::string("policy_") + std::to_string(step) + ".pt");

            // Evaluation
            if (eval_enabled) {

#ifdef USE_EVALUATION
                InitWindow(800, 600, "PendulumV0");
                SetTargetFPS(60);

                policy->eval();
                auto obs = eval_env.reset();
                auto done = false;
                int num_episodes = 0;
                float avg_episode_reward = 0.0f;
                float episode_reward = 0.0f;
                int time = 0;

                while (!WindowShouldClose()) {
                    torch::NoGradGuard guard {};
                    auto obs_tensor = torch::from_blob(obs.data(), {(int)obs.size()}).to(device);
                    auto [action_dist, value_tensor] = policy->forward(obs_tensor);
                    float action = action_dist.sample().item<float>();
                    auto [new_obs, reward, new_done] = eval_env.step(action);
                    // auto [new_obs, reward, new_done] = eval_env.step(-1.f * (eval_env.state[0]) - 1.f * (eval_env.state[1]));
                    // std::printf("State: [%f %f %f]\n", obs[0], obs[1], obs[2]);
                    // std::printf("Reward: %f\n", reward);
                    episode_reward += reward;
                    obs = new_obs;
                    done = new_done;
                    if (done) {
                        num_episodes++;
                        avg_episode_reward += episode_reward;
                        episode_reward = 0.0f;
                        eval_env.reset();
                    }
                    BeginDrawing();
                    ClearBackground(RAYWHITE);
                    Rectangle pendulum_rect {390, 300, 20, 150};
                    Vector2 pendulum_origin {10, 0};
                    DrawRectanglePro(pendulum_rect, pendulum_origin, RAD2DEG * (eval_env.state[0] + M_PI), RED);
                    DrawCircle(390, 300, 10, GRAY);
                    DrawFPS(10, 10);
                    DrawText(TextFormat("Reward: %f", reward), 10, 40, 20, DARKGRAY);
                    DrawText(TextFormat("Action: %f", action), 10, 70, 20, DARKGRAY);
                    EndDrawing();

                    time++;
                    if (time == eval_env.max_time) break;
                }

                avg_episode_reward /= num_episodes;
                std::printf("Average eval reward: %f\n", avg_episode_reward);

                CloseWindow();
#endif
            }
        }


    }

    return 0;
}