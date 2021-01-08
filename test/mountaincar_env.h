//
// Created by lasagnaphil on 21. 1. 8..
//

#ifndef FASTRL_MOUNTAINCAR_CONTINUOUS_ENV_H
#define FASTRL_MOUNTAINCAR_CONTINUOUS_ENV_H

#include <random>
#include <tuple>
#include <raylib.h>

inline float angle_normalize(float x) {
    return std::fmod(x + (float)M_PI, 2*(float)M_PI) - (float)M_PI;
}

#define SQUARE(x) ((x)*(x))

struct MountainCarEnv {
    constexpr static int obs_dim = 2;
    constexpr static int act_dim = 1;

    float min_action = -1.f;
    float max_action = 1.f;
    float min_position = -1.2f;
    float max_position = 0.6f;
    float max_speed = 0.67f;
    float goal_position = 0.45f;
    float goal_velocity = 0.0f;
    float power = 0.0015f;

    std::array<float, obs_dim> state;
    std::default_random_engine random_engine;

    MountainCarEnv(float goal_velocity = 0.0f) : goal_velocity(goal_velocity) {
        seed();
        reset();
    }

    void seed(unsigned long seed = 0) {
        random_engine.seed(seed);
    }

    std::tuple<std::array<float, obs_dim>, float, bool> step(float action) {
        auto [position, velocity] = state;
        float force = std::min(std::max(action, min_action), max_action);

        velocity += force * power - 0.0025f * std::cos(3*position);
        if (velocity > max_speed) velocity = max_speed;
        if (velocity < -max_speed) velocity = -max_speed;
        position += velocity;
        if (position > max_position) position = max_position;
        if (position < min_position) position = min_position;
        if (position == min_position && velocity < 0) velocity = 0;

        bool done = position >= goal_position && velocity >= goal_velocity;
        // The reward is a bit different from the original MountainCar, but this is because
        // we are mainly testing out online RL algorithms (PPO), which does not work well with sparse rewards
        float reward = 0;
        if (done) reward = 1000.0f;
        reward += 100.0f * velocity * velocity;
        state = {position, velocity};
        return {state, reward, done};
    }

    std::array<float, obs_dim> reset() {
        state[0] = std::uniform_real_distribution<float>(-0.6f, -0.4f)(random_engine);
        state[1] = 0.0f;
        return state;
    }

    void render() {

    }

};


#endif //FASTRL_MOUNTAINCAR_CONTINUOUS_ENV_H
