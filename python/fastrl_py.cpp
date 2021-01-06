//
// Created by lasagnaphil on 21. 1. 2..
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "fastrl/fastrl.h"

namespace py = pybind11;
using namespace fastrl;

PYBIND11_MODULE(fastrl_py, m) {
    m.doc() = "Python bindings for the FastRL library";

    py::class_<DiagGaussianDistribution>(m, "NormalDistribution")
            .def(py::init<torch::Tensor, torch::Tensor>())
            .def_readwrite("mu", &DiagGaussianDistribution::mean)
            .def_readwrite("sigma", &DiagGaussianDistribution::logstd)
            .def("entropy", &DiagGaussianDistribution::entropy)
            .def("log_prob", &DiagGaussianDistribution::log_prob)
            .def("sample", &DiagGaussianDistribution::sample);

    py::enum_<NNActivationType>(m, "NNActivationType")
            .value("ReLU", NNActivationType::ReLU)
            .value("Tanh", NNActivationType::Tanh)
            .export_values();

    py::class_<PolicyOptions>(m, "PolicyOptions")
            .def(py::init<>())
            .def_readwrite("actor_hidden_dim", &PolicyOptions::actor_hidden_dim)
            .def_readwrite("critic_hidden_dim", &PolicyOptions::critic_hidden_dim)
            .def_readwrite("activation_type", &PolicyOptions::activation_type);

    py::class_<Policy, std::shared_ptr<Policy>>(m, "Policy")
            .def(py::init<int, int, const PolicyOptions&>())
            .def("forward", &Policy::forward);

    py::class_<RolloutBufferOptions>(m, "RolloutBufferOptions")
            .def(py::init<>())
            .def_readwrite("buffer_size", &RolloutBufferOptions::buffer_size)
            .def_readwrite("gae_lambda", &RolloutBufferOptions::gae_lambda)
            .def_readwrite("gamma", &RolloutBufferOptions::gamma)
            .def_readwrite("num_envs", &RolloutBufferOptions::num_envs);

    py::class_<RolloutBufferBatch>(m, "RolloutBufferBatch")
            .def(py::init<>())
            .def_readwrite("observations", &RolloutBufferBatch::observations)
            .def_readwrite("actions", &RolloutBufferBatch::actions)
            .def_readwrite("old_values", &RolloutBufferBatch::old_values)
            .def_readwrite("old_log_prob", &RolloutBufferBatch::old_log_prob)
            .def_readwrite("advantages", &RolloutBufferBatch::advantages)
            .def_readwrite("returns", &RolloutBufferBatch::returns);

    class PyRolloutBuffer : public RolloutBuffer {
    public:
        using RolloutBuffer::RolloutBuffer;
        void compute_returns_and_advantage(const py::array_t<float>& last_values, const py::array_t<float>& last_dones) {
            assert(last_values.size() == opt.num_envs);
            assert(last_dones.size() == opt.num_envs);
            RolloutBuffer::compute_returns_and_advantage(last_values.data(), last_dones.data());
        }

        void add(int env_id, const py::array_t<float>& obs, const py::array_t<float>& action,
                 float reward, bool done, float value, float log_prob) {
            assert(obs.size() == state_size);
            assert(action.size() == action_size);
            RolloutBuffer::add(env_id, obs.data(), action.data(), reward, done, value, log_prob);
        }
    };

    py::class_<PyRolloutBuffer>(m, "RolloutBuffer")
            .def(py::init<int, int, RolloutBufferOptions>())
            .def("compute_returns_and_advantages", &PyRolloutBuffer::compute_returns_and_advantage)
            .def("add", &PyRolloutBuffer::add)
            .def("get_samples", &PyRolloutBuffer::get_samples);

    py::class_<PPOOptions>(m, "PPOOptions")
            .def(py::init<>())
            .def_readwrite("num_epochs", &PPOOptions::num_epochs)
            .def_readwrite("learning_rate", &PPOOptions::learning_rate)
            .def_readwrite("clip_range", &PPOOptions::clip_range)
            .def_readwrite("clip_range_vf_enabled", &PPOOptions::clip_range_vf_enabled)
            .def_readwrite("clip_range_vf", &PPOOptions::clip_range_vf)
            .def_readwrite("ent_coef", &PPOOptions::ent_coef)
            .def_readwrite("vf_coef", &PPOOptions::vf_coef)
            .def_readwrite("max_grad_norm", &PPOOptions::max_grad_norm)
            .def_readwrite("target_kl_enabled", &PPOOptions::target_kl_enabled)
            .def_readwrite("target_kl", &PPOOptions::target_kl);

    py::class_<PPO>(m, "PPO")
            .def(py::init<PPOOptions, std::shared_ptr<Policy>>())
            .def("train", &PPO::train);
}

#include "fastrl/fastrl.h"


