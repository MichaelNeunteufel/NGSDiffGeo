#include "kforms.hpp"
#include "riemannian_manifold.hpp"

#include <atomic>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
    using namespace ngfem;

    void ExpectException(const std::function<void()> &action)
    {
        try
        {
            action();
        }
        catch (const ngstd::Exception &)
        {
            return;
        }
        throw std::runtime_error("expected an NGSolve exception");
    }

    void TestValidation()
    {
        ExpectException([] {
            shared_ptr<CoefficientFunction> null_metric;
            RiemannianManifold invalid(null_metric);
        });

        RiemannianManifold manifold(IdentityCF(2));
        auto scalar = ScalarFieldCF(make_shared<ConstantCoefficientFunction>(1.0), 2);
        auto scalar_double = DoubleFormCF(
            make_shared<ConstantCoefficientFunction>(1.0), 0, 0, 2);
        auto sigma = DoubleFormCF(IdentityCF(2), 1, 1, 2);
        auto wrong_dimension = TensorFieldCF(IdentityCF(3), "11");

        // Invalid modes must not pass through rank-zero/empty-slot shortcuts.
        ExpectException([&] { manifold.ProjectTensor(scalar, 99); });
        ExpectException([&] {
            manifold.ProjectDoubleForm(scalar_double, 99, 0);
        });
        ExpectException([&] { manifold.Trace(scalar_double, 0, VorB(99)); });
        ExpectException([&] {
            manifold.TraceSigma(scalar_double, sigma, VorB(99));
        });
        ExpectException([&] { manifold.Raise(wrong_dimension); });

        shared_ptr<TensorFieldCoefficientFunction> null_tensor;
        shared_ptr<VectorFieldCoefficientFunction> null_vector;
        ExpectException([&] { manifold.Raise(null_tensor); });
        ExpectException([&] { manifold.Lower(null_tensor); });
        ExpectException([&] { manifold.ProjectTensor(null_tensor, 0); });
        ExpectException([&] { manifold.CovHessian(null_tensor); });
        ExpectException([&] { manifold.CovDivergence(null_tensor); });
        ExpectException([&] { manifold.Trace(null_tensor); });
        ExpectException([&] { manifold.Contraction(null_tensor, null_vector); });
        ExpectException([&] { manifold.Transpose(null_tensor); });
        ExpectException([&] { manifold.S_op(null_tensor); });
        ExpectException([&] { manifold.J_op(null_tensor); });

        ExpectException([&] { manifold.GetVolumeForm(VorB(99)); });
    }

    void TestConcurrentCurvatureInitialization()
    {
        auto manifold = make_shared<RiemannianManifold>(IdentityCF(3));
        constexpr int thread_count = 12;
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::atomic<bool> failed{false};
        std::vector<shared_ptr<DoubleFormCoefficientFunction>> results(thread_count);
        std::vector<std::thread> threads;
        threads.reserve(thread_count);

        for (int i = 0; i < thread_count; ++i)
            threads.emplace_back([&, i]
            {
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                try
                {
                    results[i] = manifold->GetRicciTensor();
                }
                catch (...)
                {
                    failed.store(true, std::memory_order_release);
                }
            });

        while (ready.load(std::memory_order_acquire) != thread_count)
            std::this_thread::yield();
        start.store(true, std::memory_order_release);
        for (auto &thread : threads)
            thread.join();

        if (failed.load(std::memory_order_acquire) || !results[0])
            throw std::runtime_error("concurrent curvature initialization failed");
        for (auto &result : results)
            if (result != results[0])
                throw std::runtime_error("curvature cache was initialized more than once");
    }
}

int main()
{
    TestValidation();
    TestConcurrentCurvatureInitialization();
    return 0;
}
