//
// Created by lasagnaphil on 21. 1. 3..
//

#ifndef FASTRL_PENDULUMENV_H
#define FASTRL_PENDULUMENV_H

#include <random>
#include <tuple>
#include <raylib.h>

inline float angle_normalize(float x) {
    return std::fmod(x + (float)M_PI, 2*(float)M_PI) - (float)M_PI;
}

#define SQUARE(x) ((x)*(x))

struct PendulumEnv {
    constexpr static int obs_dim = 3;
    constexpr static int act_dim = 1;

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
    int time = 0;
    int max_time = 200;

    std::default_random_engine random_engine;

    void seed(unsigned long seed = 0) {
        random_engine.seed(seed);
    }

    std::tuple<std::array<float, 3>, float, bool> step(float action) {
        auto [th, thdot] = state;
        float u = std::min(std::max(action, -max_torque), max_torque);
        float costs = SQUARE(angle_normalize(th)) + 0.1f * SQUARE(thdot) + 0.001f * SQUARE(u);

        float newthdot = thdot + (-3.f * g / (2.f * l) * std::sin(th + M_PI) + 3.f / (m * SQUARE(l)) * u) * dt;
        float newth = th + newthdot * dt;
        newthdot = std::min(std::max(newthdot, -max_speed), max_speed);

        state[0] = newth;
        state[1] = newthdot;
        time++;
        return {_get_obs(), -costs, false};
    }

    std::array<float, 3> reset() {
        state[0] = std::uniform_real_distribution<float>(-M_PI, M_PI)(random_engine);
        state[1] = std::uniform_real_distribution<float>(-1, 1)(random_engine);
        time = 0;
        return _get_obs();
    }

    std::array<float, 3> _get_obs() {
        auto [theta, theta_dot] = state;
        return {std::cos(theta), std::sin(theta), theta_dot};
    }

    void render() {
        Rectangle pendulum_rect {390, 300, 20, 150};
        Vector2 pendulum_origin {10, 0};
        DrawRectanglePro(pendulum_rect, pendulum_origin, RAD2DEG * (state[0] + M_PI), RED);
        DrawCircle(390, 300, 10, GRAY);
    }
};

#endif //FASTRL_PENDULUMENV_H
