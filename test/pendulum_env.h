//
// Created by lasagnaphil on 21. 1. 3..
//

#ifndef FASTRL_PENDULUMENV_H
#define FASTRL_PENDULUMENV_H

#include <random>
#include <tuple>
#include <map>
#include <any>

inline float angle_normalize(float x) {
    return std::fmod(x + M_PI, 2*M_PI) - M_PI;
}

#define SQUARE(x) ((x)*(x))

struct PendulumEnv {
    PendulumEnv(float g = 10.0) : g(g) {
        seed();
    }

    float max_speed = 8;
    float max_torque = 2;
    float dt = 0.05;
    float g;
    float m = 1;
    float l = 1;

    float state[2];

    std::default_random_engine random_engine;

    void seed(unsigned long seed = 0) {
        random_engine.seed(seed);
    }

    std::tuple<std::array<float, 3>, float, bool, std::map<std::string, std::any>> step(float action) {
        auto [th, thdot] = state;
        float u = std::min(std::max(action, -max_torque), max_torque);
        float costs = SQUARE(angle_normalize(th)) + 0.1f * SQUARE(thdot) + 0.001f * SQUARE(u);

        float newthdot = thdot + (-3.0 * g / (2.0 * l) * std::sin(th + M_PI) + 3.0 / SQUARE(m * l) * u) * dt;
        float newth = th + newthdot * dt;
        newthdot = std::min(std::max(newthdot, -max_speed), max_speed);

        state[0] = newth;
        state[1] = newthdot;
        return {_get_obs(), -costs, false, {}};
    }

    std::array<float, 3> reset() {
        state[0] = std::uniform_real_distribution<float>(-M_PI, M_PI)(random_engine);
        state[1] = std::uniform_real_distribution<float>(-1, 1)(random_engine);
        return _get_obs();
    }

    std::array<float, 3> _get_obs() {
        auto [theta, theta_dot] = state;
        return {std::cos(theta), std::sin(theta), theta_dot};
    }
};

#endif //FASTRL_PENDULUMENV_H
