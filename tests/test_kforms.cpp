#include "kforms.hpp"
#include "coefficient_grad.hpp"
#include "riemannian_manifold.hpp"

#include <elementtransformation.hpp>
#include <symbolicintegrator.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
    using namespace ngfem;
    using Pattern = AutoDiffDiff<1, NonZero>;

    void Require(bool condition, const char *message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

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

    class Probe : public CoefficientFunction
    {
        bool defined;
        int hot;

    public:
        Probe(Array<int> dims, bool adefined, bool constant, int ahot)
            : CoefficientFunction(1), defined(adefined), hot(ahot)
        {
            SetDimensions(dims);
            elementwise_constant = constant;
        }

        double Evaluate(const BaseMappedIntegrationPoint &) const override
        {
            throw Exception("probe is only used for metadata and patterns");
        }

        bool DefinedOn(const ElementTransformation &) override { return defined; }

        void CalcEquivalenceKey() override
        {
            equivalence_key = "probe[" + ToString(Dimensions()) + "," +
                              ToString(hot) + "]";
        }

        void NonZeroPattern(const ProxyUserData &, FlatVector<Pattern> values) const override
        {
            values = Pattern(false);
            values(hot) = Pattern(NonZero(true), 0);
            values(hot).DDValue(0) = NonZero(true);
        }
    };

    Array<shared_ptr<CoefficientFunction>> Nodes()
    {
        auto matrix = make_shared<Probe>(Array<int>{2, 2}, false, true, 1);
        auto left = OneFormCF(make_shared<Probe>(Array<int>{2}, false, true, 0));
        auto right = OneFormCF(make_shared<Probe>(Array<int>{2}, true, true, 1));
        return {
            SymbolicEinsumCF("ab->ba", {matrix}),
            AlternationCF(matrix, 2, 2),
            BlockAlternationByPermutationCF(matrix, 2, 0, 2),
            Wedge(DoubleFormCF(left, 1, 0, 2),
                  DoubleFormCF(right, 1, 0, 2))->GetFullCoefficient()
        };
    }

    void TestProperties()
    {
        FE_ElementTransformation<2, 2> transformation(ET_TRIG);
        for (const auto &node : Nodes())
        {
            Require(!node->DefinedOn(transformation), "child domain was lost");
            Require(node->ElementwiseConstant(), "constant metadata was lost");
        }
        auto defined = DoubleFormCF(
            make_shared<Probe>(Array<int>{2}, true, true, 0), 1, 0, 2);
        auto restricted = DoubleFormCF(
            make_shared<Probe>(Array<int>{2}, false, false, 1), 1, 0, 2);
        for (const auto &node : {Wedge(defined, restricted), Wedge(restricted, defined)})
        {
            Require(!node->DefinedOn(transformation), "binary child domain was lost");
            Require(!node->ElementwiseConstant(), "nonconstant child was ignored");
        }
    }

    void TestEquivalence()
    {
        // Reuse exact child identities: only the operation instances differ.
        auto first = Nodes();
        for (const auto &node : first)
        {
            CoefficientFunction::T_Transform transformation;
            auto second = node->Transform(transformation);
            Require(node->EquivalenceKey() == second->EquivalenceKey(),
                    "equivalent form operations have different keys");
        }
        auto tensor = make_shared<Probe>(Array<int>{2, 2, 2}, true, true, 1);
        auto left = BlockAlternationByPermutationCF(tensor, 3, 0, 2);
        auto right = BlockAlternationByPermutationCF(tensor, 3, 1, 2);
        Require(left->EquivalenceKey() != right->EquivalenceKey(),
                "block positions are missing from the key");
        auto scalar = make_shared<ConstantCoefficientFunction>(2.0);
        auto product = SymbolicEinsumCF(",->", {scalar, scalar});
        auto unrelated = make_shared<ConstantCoefficientFunction>(3.0);
        CoefficientFunction::T_DJC scalar_cache, vector_cache;
        auto scalar_jacobian = product->DiffJacobi(unrelated.get(), scalar_cache);
        auto vector_variable = unrelated->Reshape(Array<int>{1});
        auto vector_jacobian = product->DiffJacobi(vector_variable.get(), vector_cache);
        Require(scalar_jacobian->EquivalenceKey() != vector_jacobian->EquivalenceKey(),
                "Jacobian variable axes are missing from the equivalence key");
    }

    void TestPatterns()
    {
        ProxyUserData userdata;
        for (const auto &node : Nodes())
        {
            auto children = node->InputCoefficientFunctions();
            std::vector<Vector<Pattern>> storage;
            storage.reserve(children.Size());
            for (const auto &child : children)
            {
                storage.emplace_back(child->Dimension());
                child->NonZeroPattern(userdata, storage.back());
            }
            // Construct views; assigning a FlatVector copies values rather than rebinding.
            std::vector<FlatVector<Pattern>> input_views;
            for (auto &values : storage)
                input_views.emplace_back(values.Size(), values.Data());
            FlatArray<FlatVector<Pattern>> inputs(input_views.size(), input_views.data());
            Vector<Pattern> direct(node->Dimension()), precomputed(node->Dimension());
            node->NonZeroPattern(userdata, direct);
            node->NonZeroPattern(userdata, inputs, precomputed);
            bool has_derivative = false;
            for (size_t i = 0; i < node->Dimension(); ++i)
            {
                Require(bool(direct(i).Value()) == bool(precomputed(i).Value()),
                        "nonzero values differ between overloads");
                Require(bool(direct(i).DValue(0)) == bool(precomputed(i).DValue(0)),
                        "first derivative dependencies differ between overloads");
                Require(bool(direct(i).DDValue(0)) == bool(precomputed(i).DDValue(0)),
                        "second derivative dependencies differ between overloads");
                has_derivative |= bool(direct(i).DValue(0));
            }
            Require(has_derivative, "all derivative dependencies were erased");
            Require(!bool(direct(0).Value()) && !bool(direct(3).Value()),
                    "repeated-index outputs must have zero pattern");
        }
    }

    void TestCodeGeneration()
    {
        for (const auto &node : Nodes())
            for (bool simd : {false, true})
                for (int deriv : {0, 1, 2})
                {
                    Code code;
                    code.is_simd = simd;
                    code.deriv = deriv;
                    code.res_type = simd ? "SIMD<double>" : "double";
                    Array<int> inputs;
                    for (size_t i = 0; i < node->InputCoefficientFunctions().Size(); ++i)
                        inputs.Append(int(i));
                    node->GenerateCode(code, inputs, 7);
                    Require(code.pointer.empty(),
                            "generated form code embeds an evaluation fallback pointer");
                    Require(!code.body.empty(), "no generated assignments");
                }
    }

    void TestValidation()
    {
        auto matrix = make_shared<Probe>(Array<int>{2, 2}, true, true, 1);
        for (int block_len : {0, 1})
        {
            ExpectException([&] {
                BlockAlternationByPermutationCF(matrix, 1, 0, block_len);
            });
            ExpectException([&] {
                BlockAlternationByPermutationCF(
                    TensorFieldCF(matrix, "00"), 2, 0, block_len);
            });
            ExpectException([&] {
                BlockAlternationByPermutationCF(
                    make_shared<Probe>(Array<int>{5}, true, true, 0),
                    1, 0, block_len);
            });
        }
        ExpectException([&] {
            BlockAlternationByPermutationCF(
                matrix, 2, std::numeric_limits<int>::max(), 1);
        });
        Require(BlockAlternationByPermutationCF(matrix, 2, 0, 1) == matrix,
                "valid no-op alternation should reuse its input");
        ExpectException([] {
            SwapDoubleFormSlots(nullptr);
        });
        auto scalar = make_shared<ConstantCoefficientFunction>(1.0);
        for (int invalid : {-1, 256})
        {
            ExpectException([&] {
                make_shared<KFormCoefficientFunction>(scalar, invalid, 2);
            });
            ExpectException([&] {
                make_shared<DoubleFormCoefficientFunction>(scalar, invalid, 0, 2);
            });
        }
        ExpectException([&] {
            make_shared<KFormCoefficientFunction>(scalar, 0, 258);
        });
        ExpectException([&] {
            make_shared<DoubleFormCoefficientFunction>(scalar, 0, 0, 258);
        });
        RiemannianManifold manifold(IdentityCF(2));
        shared_ptr<KFormCoefficientFunction> null_kform;
        shared_ptr<DoubleFormCoefficientFunction> null_double;
        ExpectException([&] { HodgeStar(null_kform, manifold); });
        ExpectException([&] { InverseHodgeStar(null_kform, manifold); });
        ExpectException([&] { HodgeStar(null_double, manifold); });
        ExpectException([&] { InverseHodgeStar(null_double, manifold); });
        auto zero = ZeroDoubleForm(0, 0, 2);
        ExpectException([&] { HodgeStar(zero, manifold, VOL, 2); });
        ExpectException([&] { InverseHodgeStar(zero, manifold, VOL, -2); });
    }

    class TraversalProbe : public ConstantCoefficientFunction
    {
    public:
        mutable size_t visits = 0;
        mutable size_t pattern_visits = 0;
        mutable size_t zero_visits = 0;
        bool IsZeroCF() const override { ++zero_visits; return false; }
        TraversalProbe() : ConstantCoefficientFunction(1.0) {}

        Array<shared_ptr<CoefficientFunction>> InputCoefficientFunctions() const override
        {
            ++visits;
            return Array<shared_ptr<CoefficientFunction>>();
        }

        void TraverseTree(const function<void(CoefficientFunction &)> &func) override
        {
            ++visits;
            func(*this);
        }

        using ConstantCoefficientFunction::NonZeroPattern;
        void NonZeroPattern(const ProxyUserData &userdata, FlatVector<Pattern> values) const override
        {
            ++pattern_visits;
            ConstantCoefficientFunction::NonZeroPattern(userdata, values);
        }
    };

    void TestSharedTransforms()
    {
        constexpr int depth = 12;
        auto probe = make_shared<TraversalProbe>();
        auto scalar = ScalarFieldCF(probe, 2);
        shared_ptr<TensorFieldCoefficientFunction> graph = scalar;
        for (int i = 0; i < depth; ++i)
            graph = TensorProduct(graph, graph);

        for (bool affected : {true, false})
        {
            CoefficientFunction::T_Transform transformation;
            auto target = affected ? static_pointer_cast<CoefficientFunction>(scalar)
                                   : make_shared<ConstantCoefficientFunction>(2.0);
            auto replacement = make_shared<TraversalProbe>();
            transformation.replace[target] = replacement;
            probe->visits = 0;
            auto result = graph->Transform(transformation);
            Require(result->Dimensions().Size() == 0, "scalar product shape was lost");
            // A shared DAG has O(depth) nodes. Count visits, not wall-clock time.
            Require(probe->visits <= 4 * depth,
                    "replacement repeatedly scans shared operand subtrees");
            Require(replacement->pattern_visits <= 8 * depth,
                    "rebuilding einsum repeatedly evaluates shared nonzero patterns");
        }
    }

    void TestSharedGradientTraversal()
    {
        constexpr int depth = 12;
        auto probe = make_shared<TraversalProbe>();
        shared_ptr<TensorFieldCoefficientFunction> graph = ScalarFieldCF(probe, 2);
        for (int i = 0; i < depth; ++i)
            graph = TensorProduct(graph, graph);
        probe->visits = 0;
        GradCF(graph, 2);
        Require(probe->visits <= 4 * depth,
                "gradient discovery repeatedly visits shared subtrees");
    }

    void TestSharedWedgeZero()
    {
        constexpr int depth = 12;
        auto probe = make_shared<TraversalProbe>();
        auto graph = DoubleFormCF(probe, 0, 0, 2);
        for (int i = 0; i < depth; ++i)
            graph = Wedge(graph, graph);
        probe->zero_visits = 0;
        Require(!graph->IsZeroCF(), "nonzero wedge was marked zero");
        Require(probe->zero_visits <= 4 * depth,
                "wedge zero detection repeatedly visits shared subtrees");
    }

    void TestSIMD()
    {
        LocalHeapMem<100000> heap("kforms-simd-test");
        FE_ElementTransformation<2, 2> transformation(ET_TRIG);
        IntegrationRule rule(ET_TRIG, 3);
        MappedIntegrationRule<2, 2> mapped(rule, transformation, heap);
        SIMD_IntegrationRule simd_rule(ET_TRIG, 3);
        SIMD_MappedIntegrationRule<2, 2> simd_mapped(simd_rule, transformation, heap);
        auto vector = MakeVectorialCoefficientFunction(
            Array<shared_ptr<CoefficientFunction>>{
                make_shared<ConstantCoefficientFunction>(2.0),
                make_shared<ConstantCoefficientFunction>(3.0)});
        auto other = MakeVectorialCoefficientFunction(
            Array<shared_ptr<CoefficientFunction>>{
                make_shared<ConstantCoefficientFunction>(5.0),
                make_shared<ConstantCoefficientFunction>(7.0)});
        auto tensor = TensorProduct(OneFormCF(vector), OneFormCF(other));
        Array<shared_ptr<CoefficientFunction>> nodes = {
            AlternationCF(tensor, 2, 2),
            BlockAlternationByPermutationCF(tensor, 2, 0, 2),
            Wedge(DoubleFormCF(vector, 1, 0, 2),
                  DoubleFormCF(other, 1, 0, 2))->GetFullCoefficient()
        };
        nodes.Append(SymbolicEinsumCF("ab->ba", {nodes[0]}));
        const double expected[] = {0.0, -1.0, 1.0, 0.0};
        for (size_t ni : Range(nodes))
        {
            const auto &node = nodes[ni];
            const double sign = ni == nodes.Size()-1 ? -1.0 : 1.0;
            FlatMatrix<double> scalar_values(mapped.Size(), 4, heap);
            node->Evaluate(mapped, scalar_values);
            FlatMatrix<SIMD<double>> simd_values(4, simd_mapped.Size(), heap);
            node->Evaluate(simd_mapped, simd_values);
            for (int component = 0; component < 4; ++component)
            {
                for (size_t ip = 0; ip < mapped.Size(); ++ip)
                    Require(std::abs(scalar_values(ip, component) - sign * expected[component]) < 1e-12,
                            "scalar component differs from the independent reference");
                for (size_t ip = 0; ip < simd_mapped.Size(); ++ip)
                {
                    auto error = simd_values(component, ip) - SIMD<double>(sign * expected[component]);
                    Require(HSum(error * error) < 1e-24,
                            "native SIMD differs from the independent reference");
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    const std::pair<const char *, std::function<void()>> checks[] = {
        {"properties", TestProperties},
        {"equivalence", TestEquivalence},
        {"patterns", TestPatterns},
        {"codegen", TestCodeGeneration},
        {"validation", TestValidation},
        {"transforms", TestSharedTransforms},
        {"shared_gradient", TestSharedGradientTraversal},
        {"shared_wedge", TestSharedWedgeZero},
        {"simd", TestSIMD}
    };
    int failures = 0;
    for (const auto &[name, check] : checks)
    {
        if (argc > 1 && std::string(argv[1]) != name)
            continue;
        try
        {
            check();
        }
        catch (const std::exception &error)
        {
            std::cerr << name << ": " << error.what() << std::endl;
            ++failures;
        }
    }
    return failures ? 1 : 0;
}
