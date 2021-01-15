//
// Created by lasagnaphil on 21. 1. 16..
//

#ifndef FASTRL_DIST_UTILS_H
#define FASTRL_DIST_UTILS_H

#include <fastrl/fastrl.h>

#if defined(FASTRL_MPI)
#include <c10d/ProcessGroupMPI.hpp>
#endif
#if defined(FASTRL_GLOO)
#include <c10d/ProcessGroupGloo.hpp>
#endif
#if defined(FASTRL_NCCL)
#include <c10d/ProcessGroupNCCL.hpp>
#endif

namespace fastrl {

inline void average_gradients(std::vector<torch::Tensor>& params, c10d::ProcessGroup* pg) {
    std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> works;
    for (auto& param : params) {
        std::vector<torch::Tensor> tmp = {param.grad().data()};
        works.push_back(std::move(pg->allreduce(tmp)));
    }

    for (auto& work : works) {
        try {
            work->wait();
        } catch (const std::exception& ex) {
            std::cerr << "Exception received: " << ex.what() << std::endl;
#if defined(FASTRL_MPI)
            if (auto pg_mpi = dynamic_cast<c10d::ProcessGroupMPI*>(pg)) {
                pg_mpi->abort();
            }
#endif
#if defined(FASTRL_GLOO)
            if (auto pg_gloo = dynamic_cast<c10d::ProcessGroupGloo*>(pg)) {
                exit(EXIT_FAILURE);
            }
#endif
#if defined(FASTRL_NCCL)
            if (auto pg_nccl = dynamic_cast<c10d::ProcessGroupNCCL*>(pg)) {
                exit(EXIT_FAILURE);
            }
#endif
        }
    }
    for (auto& param : params) {
        param.grad().div_(pg->getSize());
    }
}

}

#endif //FASTRL_DIST_UTILS_H
