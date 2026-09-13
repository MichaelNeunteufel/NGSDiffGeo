#include "kforms.hpp"
#include "kforms_detail.hpp"
#include "coefficient_grad.hpp"
#include "riemannian_manifold.hpp"
#include "symbolic_expression.hpp"

#include <core/archive.hpp>
#include <elementtransformation.hpp>
#include <symbolicintegrator.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

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
        nodes.Append(ngfem::kforms_internal::KFormFromIndependentCF(
                         make_shared<ConstantCoefficientFunction>(1.0), 2, 2)
                         ->GetFullCoefficient());
        nodes.Append(Compile(nodes.Last(), false, 0, true));
        const double expected[] = {0.0, -1.0, 1.0, 0.0};
        for (size_t ni : Range(nodes))
        {
            const auto &node = nodes[ni];
            const double sign = ni == 3 ? -1.0 : 1.0;
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

    shared_ptr<CoefficientFunction> ConstantVector(std::initializer_list<double> values)
    {
        Array<shared_ptr<CoefficientFunction>> components;
        for (double value : values)
            components.Append(make_shared<ConstantCoefficientFunction>(value));
        return MakeVectorialCoefficientFunction(std::move(components));
    }

    shared_ptr<CoefficientFunction> ConstantVector(const std::vector<double> &values)
    {
        Array<shared_ptr<CoefficientFunction>> components;
        for (double value : values)
            components.Append(make_shared<ConstantCoefficientFunction>(value));
        return MakeVectorialCoefficientFunction(std::move(components));
    }

    shared_ptr<CoefficientFunction> ConstantComplexVector(
        std::initializer_list<Complex> values)
    {
        Array<shared_ptr<CoefficientFunction>> components;
        for (Complex value : values)
            components.Append(MakeConstantCoefficientFunction(value));
        return MakeVectorialCoefficientFunction(std::move(components));
    }

    shared_ptr<CoefficientFunction> AlternatingTwoForm3D(
        double c01, double c02, double c12)
    {
        return ConstantVector({0.0, -c01, -c02,
                               c01, 0.0, -c12,
                               c02, c12, 0.0})
            ->Reshape(Array<int>{3, 3});
    }

    void RequireSameValues(const shared_ptr<CoefficientFunction> &actual,
                           const shared_ptr<CoefficientFunction> &expected,
                           const char *message)
    {
        Require(actual->Dimensions() == expected->Dimensions(), message);
        Require(actual->IsComplex() == expected->IsComplex(), message);
        LocalHeapMem<100000> heap("compact-form-values");
        FE_ElementTransformation<2, 2> transformation(ET_TRIG);
        IntegrationRule rule(ET_TRIG, 2);
        MappedIntegrationRule<2, 2> mapped(rule, transformation, heap);
        if (actual->IsComplex())
        {
            FlatMatrix<Complex> actual_values(mapped.Size(), actual->Dimension(), heap);
            FlatMatrix<Complex> expected_values(mapped.Size(), expected->Dimension(), heap);
            actual->Evaluate(mapped, actual_values);
            expected->Evaluate(mapped, expected_values);
            for (size_t ip = 0; ip < mapped.Size(); ++ip)
                for (int component = 0; component < actual->Dimension(); ++component)
                    Require(std::abs(actual_values(ip, component) -
                                     expected_values(ip, component)) < 1e-12,
                            message);
        }
        else
        {
            FlatMatrix<double> actual_values(mapped.Size(), actual->Dimension(), heap);
            FlatMatrix<double> expected_values(mapped.Size(), expected->Dimension(), heap);
            actual->Evaluate(mapped, actual_values);
            expected->Evaluate(mapped, expected_values);
            for (size_t ip = 0; ip < mapped.Size(); ++ip)
                for (int component = 0; component < actual->Dimension(); ++component)
                    Require(std::abs(actual_values(ip, component) -
                                     expected_values(ip, component)) < 1e-12,
                            message);
        }
    }

    template <int D>
    void RequireSameSpatialValues(
        const shared_ptr<CoefficientFunction> &actual,
        const shared_ptr<CoefficientFunction> &expected,
        ELEMENT_TYPE element_type, const char *message)
    {
        Require(actual->Dimensions() == expected->Dimensions(), message);
        Require(!actual->IsComplex() && !expected->IsComplex(), message);
        LocalHeapMem<100000> heap("compact-spatial-values");
        FE_ElementTransformation<D, D> transformation(element_type);
        IntegrationRule rule(element_type, 2);
        MappedIntegrationRule<D, D> mapped(rule, transformation, heap);
        FlatMatrix<double> actual_values(
            mapped.Size(), actual->Dimension(), heap);
        FlatMatrix<double> expected_values(
            mapped.Size(), expected->Dimension(), heap);
        actual->Evaluate(mapped, actual_values);
        expected->Evaluate(mapped, expected_values);
        for (size_t ip = 0; ip < mapped.Size(); ++ip)
            for (int component = 0; component < actual->Dimension(); ++component)
                Require(std::abs(actual_values(ip, component) -
                                 expected_values(ip, component)) < 1e-11,
                        message);
    }

    void RequireSameSpatialValues(
        const shared_ptr<CoefficientFunction> &actual,
        const shared_ptr<CoefficientFunction> &expected,
        int dim, const char *message)
    {
        if (dim == 1)
            return RequireSameSpatialValues<1>(actual, expected, ET_SEGM,
                                               message);
        if (dim == 2)
            return RequireSameSpatialValues<2>(actual, expected, ET_TRIG,
                                               message);
        if (dim == 3)
            return RequireSameSpatialValues<3>(actual, expected, ET_TET,
                                               message);
        throw std::runtime_error("unsupported spatial test dimension");
    }

    void TestCompactKFormSymbolics()
    {
        using namespace ngfem::kforms_internal;

        auto compact_a = KFormFromIndependentCF(
            ConstantVector({2.0, 3.0, 5.0}), 2, 3);
        auto compact_b = KFormFromIndependentCF(
            ConstantVector({7.0, 11.0, 13.0}), 2, 3);
        auto sum = AddKForms(compact_a, compact_b);
        auto full_sum = sum->GetFullCoefficient();
        auto is_compact = [](const shared_ptr<CoefficientFunction> &value) {
            auto form = dynamic_pointer_cast<KFormCoefficientFunction>(value);
            return form && TryGetCompactView(*form).has_value();
        };

        Require(TryGetCompactView(*sum).has_value(),
                "compact sum lost its facade");
        Require(full_sum->Dimensions() == Array<int>({3, 3}),
                "compact facade lost its full public shape");
        auto sum_view = TryGetCompactView(*sum);
        Require(sum_view->independent->Dimension() == 3,
                "compact sum does not retain three independent components");
        RequireSameValues(sum, SymbolicSumCF(compact_a, compact_b),
                          "compact sum differs from its dense semantic value");

        bool saw_left_evaluation_facade = false;
        bool saw_right_evaluation_facade = false;
        bool saw_dense_alternation = false;
        full_sum->TraverseDAG([&](CoefficientFunction &node) {
            saw_left_evaluation_facade |= &node == compact_a->GetFullCoefficient().get();
            saw_right_evaluation_facade |= &node == compact_b->GetFullCoefficient().get();
            saw_dense_alternation |=
                node.GetDescription() == "AlternationCF" ||
                node.GetDescription() == "BlockAlternationCF";
        });
        Require(!saw_left_evaluation_facade && !saw_right_evaluation_facade,
                "compact chain evaluates a full-shaped intermediate facade");
        Require(!saw_dense_alternation,
                "compact chain contains a dense alternation node");

        bool saw_left_semantic_operand = false;
        bool saw_right_semantic_operand = false;
        TraverseSemanticDAG(full_sum, [&](CoefficientFunction &node) {
            saw_left_semantic_operand |= &node == compact_a.get();
            saw_right_semantic_operand |= &node == compact_b.get();
        });
        Require(saw_left_semantic_operand && saw_right_semantic_operand,
                "compact chain discarded semantic operand identities");

        auto nonalternating = ConstantVector({1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0,
                                              7.0, 8.0, 9.0})
                                  ->Reshape(Array<int>{3, 3});
        for (bool replace_left : {true, false})
            for (bool replace_wrapper : {true, false})
            {
                auto operand = replace_left ? compact_a : compact_b;
                auto other = replace_left ? compact_b : compact_a;
                CoefficientFunction::T_Transform transformation;
                auto target = replace_wrapper
                                  ? static_pointer_cast<CoefficientFunction>(operand)
                                  : operand->GetFullCoefficient();
                transformation.replace[target] = nonalternating;
                auto transformed = sum->Transform(transformation);
                Require(!is_compact(transformed),
                        "non-alternating replacement remained marked compact");
                auto expected = replace_left
                                    ? SymbolicSumCF(nonalternating, other)
                                    : SymbolicSumCF(other, nonalternating);
                RequireSameValues(transformed, expected,
                                  "compact replacement changed non-alternating values");

                auto differentiated = sum->Diff(target.get(), nonalternating);
                RequireSameValues(differentiated, nonalternating,
                                  "compact differentiation compressed a full direction");

                CoefficientFunction::T_DJC jacobian_cache;
                auto jacobian = full_sum->DiffJacobi(target.get(), jacobian_cache);
                Array<int> expected_dims(full_sum->Dimensions());
                expected_dims.Append(target->Dimensions());
                Require(jacobian->Dimensions() == expected_dims,
                        "compact tensor Jacobian has the wrong shape");
                RequireSameValues(jacobian, IdentityCF(full_sum->Dimensions()),
                                  "compact tensor Jacobian is not the identity");
            }

        auto scalar = make_shared<ConstantCoefficientFunction>(2.0);
        auto scalar_compact = KFormFromIndependentCF(
            scalar * ConstantVector({2.0, 3.0, 5.0}), 2, 3);
        auto scalar_sum = AddKForms(scalar_compact, compact_b);
        auto scalar_direction = make_shared<ConstantCoefficientFunction>(4.0);
        RequireSameValues(
            scalar_sum->Diff(scalar.get(), scalar_direction),
            scalar_direction * AlternatingTwoForm3D(2.0, 3.0, 5.0),
            "compact scalar directional derivative lost its semantic source");
        CoefficientFunction::T_DJC scalar_jacobian_cache;
        RequireSameValues(
            scalar_sum->GetFullCoefficient()->DiffJacobi(
                scalar.get(), scalar_jacobian_cache),
            AlternatingTwoForm3D(2.0, 3.0, 5.0),
            "compact scalar Jacobian lost its semantic source");

        auto self_replacement = ConstantVector({9.0, 8.0, 7.0,
                                                6.0, 5.0, 4.0,
                                                3.0, 2.0, 1.0})
                                    ->Reshape(Array<int>{3, 3});
        CoefficientFunction::T_Transform self_transform;
        self_transform.replace[full_sum] = self_replacement;
        RequireSameValues(full_sum->Transform(self_transform), self_replacement,
                          "compact facade self replacement failed");
        RequireSameValues(full_sum->Diff(full_sum.get(), self_replacement),
                          self_replacement,
                          "compact facade self differentiation failed");

        for (bool realcompile : {false, true})
        {
            auto compiled = Compile(full_sum, realcompile, 0, true);
            RequireSameValues(compiled, full_sum,
                              "compiled compact facade changed values");
        }

        Code code;
        Array<int> code_inputs{0};
        full_sum->GenerateCode(code, code_inputs, 1);
        Require(code.pointer.empty(),
                "compact facade code generation uses an evaluation fallback");

        auto stream = make_shared<std::stringstream>();
        {
            BinaryOutArchive archive(stream);
            shared_ptr<CoefficientFunction> stored = sum;
            archive & stored;
        }
        shared_ptr<CoefficientFunction> restored;
        {
            auto input = make_shared<std::stringstream>(stream->str());
            BinaryInArchive archive(input);
            archive & restored;
        }
        auto restored_form = dynamic_pointer_cast<KFormCoefficientFunction>(restored);
        Require(restored_form != nullptr, "archive lost the K-form wrapper");
        Require(TryGetCompactView(*restored_form).has_value(),
                "archive lost compact operation state");
        RequireSameValues(restored, sum, "archived compact facade changed values");

        const Complex z01(2.0, 1.0), z02(3.0, -2.0), z12(5.0, 4.0);
        auto complex_compact = KFormFromIndependentCF(
            ConstantComplexVector({z01, z02, z12}), 2, 3);
        auto complex_semantic = KFormCF(
            ConstantComplexVector(
                {Complex(0.0), -z01, -z02,
                 z01, Complex(0.0), -z12,
                 z02, z12, Complex(0.0)})
                ->Reshape(Array<int>{3, 3}),
            2, 3);
        RequireSameValues(complex_compact, complex_semantic,
                          "complex compact facade changed values");
        RequireSameValues(Compile(complex_compact, false, 0, true),
                          complex_semantic,
                          "graph-compiled complex compact facade changed values");
        RequireSameValues(Compile(complex_compact, true, 0, true),
                          complex_semantic,
                          "native-compiled complex compact facade changed values");

        ExpectException([&] {
            KFormFromIndependentCF(ConstantVector({1.0, 2.0}), 2, 3);
        });
    }

    size_t BinomialReference(int n, int k)
    {
        if (k < 0 || k > n)
            return 0;
        size_t result = 1;
        for (int i = 1; i <= k; ++i)
            result = result * size_t(n - k + i) / size_t(i);
        return result;
    }

    void TestFormBasis()
    {
        using namespace ngfem::kforms_internal;
        for (int dim = 1; dim <= 4; ++dim)
            for (int degree = 0; degree <= dim; ++degree)
            {
                auto basis = GetFormBasis(dim, degree);
                Require(basis == GetFormBasis(dim, degree),
                        "FormBasis cache did not reuse its table");
                Require(basis->IndependentSize() == BinomialReference(dim, degree),
                        "FormBasis has the wrong canonical component count");
                size_t expected_dense = 1;
                for (int i = 0; i < degree; ++i)
                    expected_dense *= size_t(dim);
                Require(basis->DenseSize() == expected_dense,
                        "FormBasis has the wrong dense component count");
                Require(basis->OwnedStorageBytesLowerBound() >=
                            sizeof(FormBasis),
                        "FormBasis memory report is incomplete");

                for (size_t compact = 0; compact < basis->IndependentSize(); ++compact)
                {
                    const auto &canonical = basis->CanonicalIndex(compact);
                    Require(canonical.size() == size_t(degree),
                            "canonical form index has the wrong degree");
                    for (int slot = 1; slot < degree; ++slot)
                        Require(canonical[size_t(slot - 1)] < canonical[size_t(slot)],
                                "canonical form index is not strictly increasing");
                    if (compact > 0)
                        Require(std::lexicographical_compare(
                                    basis->CanonicalIndex(compact - 1).begin(),
                                    basis->CanonicalIndex(compact - 1).end(),
                                    canonical.begin(), canonical.end()),
                                "canonical form indices are not lexicographic");
                    Require(basis->CompactIndex(canonical) == int(compact),
                            "canonical inverse map returns the wrong component");
                }

                std::vector<int> axes(static_cast<size_t>(degree));
                for (size_t dense = 0; dense < basis->DenseSize(); ++dense)
                {
                    size_t remainder = dense;
                    int inversions = 0;
                    bool repeated = false;
                    for (int slot = 0; slot < degree; ++slot)
                    {
                        axes[size_t(slot)] = int(remainder % size_t(dim));
                        remainder /= size_t(dim);
                        for (int previous = 0; previous < slot; ++previous)
                        {
                            repeated |= axes[size_t(previous)] == axes[size_t(slot)];
                            inversions += axes[size_t(previous)] > axes[size_t(slot)];
                        }
                    }
                    auto entry = basis->DenseEntry(dense);
                    if (repeated)
                    {
                        Require(entry.sign == 0 && entry.independent_index == -1,
                                "repeated dense index did not map to zero");
                        continue;
                    }
                    auto sorted = axes;
                    std::sort(sorted.begin(), sorted.end());
                    int expected_index = -1;
                    for (size_t compact = 0; compact < basis->IndependentSize(); ++compact)
                        if (basis->CanonicalIndex(compact) == sorted)
                            expected_index = int(compact);
                    Require(entry.independent_index == expected_index,
                            "dense index maps to the wrong compact component");
                    Require(entry.sign == (inversions % 2 ? -1 : 1),
                            "dense index has the wrong permutation sign");
                }

                for (int left_degree = 0; left_degree <= degree; ++left_degree)
                {
                    const auto &shuffles = basis->Shuffles(left_degree);
                    Require(shuffles.size() == BinomialReference(degree, left_degree),
                            "FormBasis has the wrong shuffle count");
                    for (const auto &shuffle : shuffles)
                    {
                        Require(shuffle.left_positions.size() == size_t(left_degree) &&
                                    shuffle.right_positions.size() ==
                                        size_t(degree - left_degree),
                                "FormBasis shuffle has the wrong partition sizes");
                        int inversions = 0;
                        for (size_t i = 0; i < shuffle.left_positions.size(); ++i)
                            inversions += shuffle.left_positions[i] - int(i);
                        Require(shuffle.sign == (inversions % 2 ? -1 : 1),
                                "FormBasis shuffle has the wrong sign");
                    }
                }
            }

        std::vector<shared_ptr<const FormBasis>> concurrent(16);
        std::vector<std::thread> threads;
        for (size_t i = 0; i < concurrent.size(); ++i)
            threads.emplace_back([&, i] { concurrent[i] = GetFormBasis(4, 2); });
        for (auto &thread : threads)
            thread.join();
        for (auto basis : concurrent)
            Require(basis == concurrent[0],
                    "concurrent FormBasis lookup created multiple tables");

        ExpectException([] { GetFormBasis(0, 0); });
        ExpectException([] { GetFormBasis(5, 1); });
        ExpectException([] { GetFormBasis(2, 3); });
    }

    shared_ptr<CoefficientFunction> DenseReferenceFromBasis(
        const ngfem::kforms_internal::FormBasis &basis,
        const std::vector<double> &independent)
    {
        std::vector<double> dense(basis.DenseSize(), 0.0);
        for (size_t i = 0; i < dense.size(); ++i)
        {
            auto entry = basis.DenseEntry(i);
            if (entry.sign)
                dense[i] = entry.sign * independent[size_t(entry.independent_index)];
        }
        if (basis.Degree() == 0)
            return make_shared<ConstantCoefficientFunction>(dense[0]);
        Array<int> dims(static_cast<size_t>(basis.Degree()));
        dims = basis.DimensionOfSpace();
        return ConstantVector(dense)->Reshape(dims);
    }

    void TestCompactKFormFacade()
    {
        using namespace ngfem::kforms_internal;
        for (int dim = 1; dim <= 4; ++dim)
            for (int degree = 0; degree <= dim; ++degree)
            {
                auto basis = GetFormBasis(dim, degree);
                std::vector<double> values(basis->IndependentSize());
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = double(i + 2);
                auto form = KFormFromIndependentCF(ConstantVector(values), degree, dim);
                auto view = TryGetCompactView(*form);
                Require(view.has_value(), "independent factory did not create a compact facade");
                Require(view->degree == degree && view->dim == dim,
                        "compact view has the wrong form metadata");
                Require(view->independent->Dimension() == int(values.size()),
                        "compact view exposes the wrong independent shape");
                Array<int> full_dims(static_cast<size_t>(degree));
                full_dims = dim;
                Require(form->Dimensions() == full_dims,
                        "compact facade has the wrong public shape");
                auto reference = DenseReferenceFromBasis(*basis, values);
                RequireSameValues(form, reference,
                                  "compact facade differs from its basis expansion");
                RequireSameValues(Compile(form, false, 0, true), reference,
                                  "graph-compiled compact facade changed values");

                auto zero = ZeroKForm(degree, dim);
                Require(TryGetCompactView(*zero).has_value(),
                        "valid zero K-form did not use compact storage");
                RequireSameValues(zero, ZeroCF(full_dims),
                                  "compact zero K-form changed its full value");

                auto dense_form = KFormCF(reference, degree, dim);
                Require(!TryGetCompactView(*dense_form).has_value(),
                        "public dense constructor silently marked input compact");
            }

        Require(!TryGetCompactView(*ZeroKForm(0, 0)).has_value(),
                "unknown-dimension scalar zero acquired a fictitious basis");
        Require(!TryGetCompactView(*ZeroKForm(3, 2)).has_value(),
                "overdegree zero constructed an empty compact child");

        auto independent = ConstantVector({2.0, 3.0, 5.0});
        auto form = KFormFromIndependentCF(independent, 2, 3);
        auto arbitrary_direction = ConstantVector({1.0, 2.0, 3.0,
                                                   4.0, 5.0, 6.0,
                                                   7.0, 8.0, 9.0})
                                       ->Reshape(Array<int>{3, 3});
        RequireSameValues(
            form->GetFullCoefficient()->Diff(
                form->GetFullCoefficient().get(), arbitrary_direction),
            arbitrary_direction,
            "compact facade compressed its full-shaped self direction");

        auto scale = make_shared<ConstantCoefficientFunction>(2.0);
        auto scaled_form = KFormFromIndependentCF(
            scale * ConstantVector({2.0, 3.0, 5.0}), 2, 3);
        auto scalar_direction = make_shared<ConstantCoefficientFunction>(4.0);
        auto differentiated = dynamic_pointer_cast<KFormCoefficientFunction>(
            scaled_form->Diff(scale.get(), scalar_direction));
        Require(differentiated && TryGetCompactView(*differentiated).has_value(),
                "scalar differentiation unnecessarily lost compact storage");
        RequireSameValues(
            differentiated,
            scalar_direction * AlternatingTwoForm3D(2.0, 3.0, 5.0),
            "compact facade scalar differentiation changed values");
        CoefficientFunction::T_DJC scalar_cache;
        auto jacobian = scaled_form->GetFullCoefficient()->DiffJacobi(
            scale.get(), scalar_cache);
        auto jacobian_form = KFormCF(jacobian, 2, 3);
        Require(TryGetCompactView(*jacobian_form).has_value(),
                "scalar Jacobian unnecessarily lost compact storage");
        RequireSameValues(jacobian, AlternatingTwoForm3D(2.0, 3.0, 5.0),
                          "compact facade scalar Jacobian changed values");

        CoefficientFunction::T_Transform scalar_transform;
        scalar_transform.replace[scale] =
            make_shared<ConstantCoefficientFunction>(7.0);
        auto transformed = dynamic_pointer_cast<KFormCoefficientFunction>(
            scaled_form->Transform(scalar_transform));
        Require(transformed && TryGetCompactView(*transformed).has_value(),
                "scalar replacement unnecessarily lost compact storage");
        RequireSameValues(
            transformed, 7.0 * AlternatingTwoForm3D(2.0, 3.0, 5.0),
            "compact facade scalar replacement changed values");

        CoefficientFunction::T_Transform full_transform;
        full_transform.replace[scaled_form->GetFullCoefficient()] =
            arbitrary_direction;
        auto dense_replacement = dynamic_pointer_cast<KFormCoefficientFunction>(
            scaled_form->Transform(full_transform));
        Require(dense_replacement &&
                    !TryGetCompactView(*dense_replacement).has_value(),
                "full-shaped facade replacement remained marked compact");
        RequireSameValues(dense_replacement, arbitrary_direction,
                          "full-shaped facade replacement changed values");

        auto pattern_child = make_shared<Probe>(Array<int>{3}, false, true, 1);
        auto pattern_form = KFormFromIndependentCF(pattern_child, 2, 3);
        FE_ElementTransformation<2, 2> transformation(ET_TRIG);
        Require(!pattern_form->DefinedOn(transformation),
                "compact facade lost child domain restrictions");
        Require(pattern_form->ElementwiseConstant(),
                "compact facade lost constant metadata");
        ProxyUserData userdata;
        Vector<Pattern> pattern(pattern_form->Dimension());
        pattern_form->NonZeroPattern(userdata, pattern);
        auto pattern_basis = GetFormBasis(3, 2);
        for (size_t dense = 0; dense < pattern_basis->DenseSize(); ++dense)
        {
            auto entry = pattern_basis->DenseEntry(dense);
            Require(bool(pattern(dense).Value()) ==
                        (entry.sign != 0 && entry.independent_index == 1),
                    "compact facade propagated the wrong nonzero pattern");
        }

        Code code;
        Array<int> code_inputs{0};
        form->GetFullCoefficient()->GenerateCode(code, code_inputs, 1);
        Require(code.pointer.empty(),
                "compact facade code generation uses an evaluation fallback");

        auto stream = make_shared<std::stringstream>();
        {
            BinaryOutArchive archive(stream);
            shared_ptr<CoefficientFunction> stored = form;
            archive & stored;
        }
        shared_ptr<CoefficientFunction> restored;
        {
            auto input = make_shared<std::stringstream>(stream->str());
            BinaryInArchive archive(input);
            archive & restored;
        }
        auto restored_form = dynamic_pointer_cast<KFormCoefficientFunction>(restored);
        Require(restored_form && TryGetCompactView(*restored_form).has_value(),
                "archive lost production compact facade state");
        RequireSameValues(restored, form,
                          "archived production compact facade changed values");

        ExpectException([] {
            KFormFromIndependentCF(ConstantVector({1.0, 2.0}), 2, 3);
        });
    }

    shared_ptr<CoefficientFunction> DenseDoubleFormReference(
        const ngfem::kforms_internal::FormBasis &left_basis,
        const ngfem::kforms_internal::FormBasis &right_basis,
        const std::vector<double> &independent)
    {
        std::vector<double> dense(
            left_basis.DenseSize() * right_basis.DenseSize(), 0.0);
        for (size_t left_dense = 0;
             left_dense < left_basis.DenseSize(); ++left_dense)
            for (size_t right_dense = 0;
                 right_dense < right_basis.DenseSize(); ++right_dense)
            {
                const auto left = left_basis.DenseEntry(left_dense);
                const auto right = right_basis.DenseEntry(right_dense);
                if (left.sign == 0 || right.sign == 0)
                    continue;
                const size_t compact =
                    size_t(left.independent_index) *
                        right_basis.IndependentSize() +
                    size_t(right.independent_index);
                dense[left_dense * right_basis.DenseSize() + right_dense] =
                    left.sign * right.sign * independent[compact];
            }
        const int rank = left_basis.Degree() + right_basis.Degree();
        if (rank == 0)
            return make_shared<ConstantCoefficientFunction>(dense[0]);
        Array<int> dimensions(static_cast<size_t>(rank));
        dimensions = left_basis.DimensionOfSpace();
        return ConstantVector(dense)->Reshape(dimensions);
    }

    void TestCompactDoubleFormFacade()
    {
        using namespace ngfem::kforms_internal;
        for (int dim = 1; dim <= 4; ++dim)
            for (int left_degree = 0; left_degree <= dim; ++left_degree)
                for (int right_degree = 0; right_degree <= dim;
                     ++right_degree)
                {
                    auto left_basis = GetFormBasis(dim, left_degree);
                    auto right_basis = GetFormBasis(dim, right_degree);
                    const size_t compact_size =
                        left_basis->IndependentSize() *
                        right_basis->IndependentSize();
                    std::vector<double> values(compact_size);
                    for (size_t i = 0; i < compact_size; ++i)
                        values[i] = double(i + 2);
                    auto form = DoubleFormFromIndependentCF(
                        ConstantVector(values), left_degree,
                        right_degree, dim);
                    auto view = TryGetCompactView(*form);
                    Require(view.has_value(),
                            "independent double-form factory stayed dense");
                    Require(view->left_degree == left_degree &&
                                view->right_degree == right_degree &&
                                view->dim == dim,
                            "compact double-form view has wrong metadata");
                    Require(view->independent->Dimension() ==
                                int(compact_size),
                            "compact double-form view has wrong size");
                    Array<int> dimensions(
                        size_t(left_degree + right_degree));
                    dimensions = dim;
                    Require(form->Dimensions() == dimensions,
                            "compact double-form facade has wrong public shape");

                    if (left_degree + right_degree <= 4)
                    {
                        auto reference = DenseDoubleFormReference(
                            *left_basis, *right_basis, values);
                        RequireSameValues(
                            form, reference,
                            "compact double-form facade changed values");
                        RequireSameValues(
                            Compile(form, false, 0, true), reference,
                            "compiled compact double-form facade changed values");
                    }

                    auto zero = ZeroDoubleForm(
                        left_degree, right_degree, dim);
                    Require(TryGetCompactView(*zero).has_value(),
                            "valid zero double form did not use compact storage");

                    auto dense = DoubleFormCF(
                        DenseDoubleFormReference(
                            *left_basis, *right_basis, values),
                        left_degree, right_degree, dim);
                    Require(!TryGetCompactView(*dense).has_value(),
                            "public double-form constructor silently compressed input");
                }

        Require(!TryGetCompactView(*ZeroDoubleForm(0, 0, 0)).has_value(),
                "unknown-dimension double scalar acquired a fictitious basis");
        Require(!TryGetCompactView(*ZeroDoubleForm(3, 0, 2)).has_value(),
                "overdegree double zero constructed an empty compact child");

        auto singleton = DoubleFormFromIndependentCF(
            make_shared<ConstantCoefficientFunction>(3.0), 0, 0, 2);
        Require(singleton->Dimensions().Size() == 0,
                "compact double scalar lost scalar shape");
        RequireSameValues(singleton, ConstantCF(3.0),
                          "compact double scalar changed value");

        auto factor = make_shared<ConstantCoefficientFunction>(2.0);
        auto scaled = DoubleFormFromIndependentCF(
            factor * ConstantVector({2.0, 3.0, 5.0, 7.0}),
            1, 1, 2);
        auto differentiated = dynamic_pointer_cast<
            DoubleFormCoefficientFunction>(
            scaled->Diff(factor.get(), ConstantCF(4.0)));
        Require(differentiated && TryGetCompactView(*differentiated),
                "double-form scalar derivative lost compact storage");
        RequireSameValues(
            differentiated,
            4.0 * ConstantVector({2.0, 3.0, 5.0, 7.0})
                      ->Reshape(Array<int>{2, 2}),
            "double-form scalar derivative changed values");

        auto arbitrary = ConstantVector({1.0, 2.0, 3.0, 4.0})
                             ->Reshape(Array<int>{2, 2});
        CoefficientFunction::T_Transform replacement;
        replacement.replace[scaled->GetFullCoefficient()] = arbitrary;
        auto replaced = dynamic_pointer_cast<DoubleFormCoefficientFunction>(
            scaled->Transform(replacement));
        Require(replaced && !TryGetCompactView(*replaced),
                "full double-form facade replacement stayed compact");
        RequireSameValues(replaced, arbitrary,
                          "full double-form facade replacement changed values");

        auto complex_values = ConstantComplexVector(
            {Complex(1, 2), Complex(3, -1),
             Complex(-2, 4), Complex(5, 1)});
        auto complex_form = DoubleFormFromIndependentCF(
            complex_values, 1, 1, 2);
        RequireSameValues(
            Compile(complex_form, true, 0, true),
            complex_values->Reshape(Array<int>{2, 2}),
            "native-compiled complex double-form facade changed values");

        LocalHeapMem<100000> heap("compact-double-form-simd");
        FE_ElementTransformation<2, 2> transformation(ET_TRIG);
        IntegrationRule rule(ET_TRIG, 2);
        MappedIntegrationRule<2, 2> mapped(rule, transformation, heap);
        SIMD_IntegrationRule simd_rule(ET_TRIG, 2);
        SIMD_MappedIntegrationRule<2, 2> simd_mapped(
            simd_rule, transformation, heap);
        FlatMatrix<double> scalar_values(mapped.Size(), 4, heap);
        scaled->Evaluate(mapped, scalar_values);
        FlatMatrix<SIMD<double>> simd_values(4, simd_mapped.Size(), heap);
        scaled->Evaluate(simd_mapped, simd_values);
        const double expected[] = {4.0, 6.0, 10.0, 14.0};
        for (int component = 0; component < 4; ++component)
        {
            for (size_t ip = 0; ip < mapped.Size(); ++ip)
                Require(std::abs(scalar_values(ip, component) -
                                 expected[component]) < 1e-12,
                        "compact double-form scalar evaluation changed value");
            for (size_t ip = 0; ip < simd_mapped.Size(); ++ip)
            {
                auto error = simd_values(component, ip) -
                             SIMD<double>(expected[component]);
                Require(HSum(error * error) < 1e-24,
                        "compact double-form SIMD evaluation changed value");
            }
        }

        auto stream = make_shared<std::stringstream>();
        {
            BinaryOutArchive archive(stream);
            shared_ptr<CoefficientFunction> stored = scaled;
            archive & stored;
        }
        shared_ptr<CoefficientFunction> restored;
        {
            auto input = make_shared<std::stringstream>(stream->str());
            BinaryInArchive archive(input);
            archive & restored;
        }
        auto restored_form = dynamic_pointer_cast<
            DoubleFormCoefficientFunction>(restored);
        Require(restored_form && TryGetCompactView(*restored_form),
                "archive lost compact double-form facade state");
        RequireSameValues(restored, scaled,
                          "archived compact double-form facade changed values");

        ExpectException([] {
            DoubleFormFromIndependentCF(
                ConstantVector({1.0, 2.0}), 1, 1, 2);
        });
    }

    shared_ptr<KFormCoefficientFunction> SequentialCompactForm(
        int dim, int degree, double offset)
    {
        auto basis = ngfem::kforms_internal::GetFormBasis(dim, degree);
        std::vector<double> values(basis->IndependentSize());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = offset + double(i + 1);
        return ngfem::kforms_internal::KFormFromIndependentCF(
            ConstantVector(values), degree, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> SequentialCompactDoubleForm(
        int dim, int left_degree, int right_degree, double offset)
    {
        auto left_basis = ngfem::kforms_internal::GetFormBasis(
            dim, left_degree);
        auto right_basis = ngfem::kforms_internal::GetFormBasis(
            dim, right_degree);
        std::vector<double> values(left_basis->IndependentSize() *
                                   right_basis->IndependentSize());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = offset + double(i + 1);
        return ngfem::kforms_internal::DoubleFormFromIndependentCF(
            ConstantVector(values), left_degree, right_degree, dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> DenseDoubleForm(
        int dim, int left_degree, int right_degree, double offset)
    {
        size_t count = 1;
        for (int axis = 0; axis < left_degree + right_degree; ++axis)
            count *= size_t(dim);
        std::vector<double> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = offset + double(i + 1);
        shared_ptr<CoefficientFunction> value;
        if (left_degree + right_degree == 0)
            value = ConstantCF(values[0]);
        else
        {
            Array<int> dimensions(
                size_t(left_degree + right_degree));
            dimensions = dim;
            value = ConstantVector(values)->Reshape(dimensions);
        }
        return DoubleFormCF(value, left_degree, right_degree, dim);
    }

    shared_ptr<KFormCoefficientFunction> DenseCopy(
        const shared_ptr<KFormCoefficientFunction> &form)
    {
        // Put an ordinary full-value operation at the root so this copy is
        // deliberately outside the compact representation contract.
        return KFormCF(ScaleCoefficientCF(form, ConstantCF(1.0)), form->Degree(),
                       form->DimensionOfSpace());
    }

    void TestCompactKFormArithmetic()
    {
        using namespace ngfem::kforms_internal;
        for (int dim = 1; dim <= 4; ++dim)
            for (int left_degree = 0; left_degree <= dim; ++left_degree)
                for (int right_degree = 0;
                     right_degree <= dim - left_degree; ++right_degree)
                {
                    auto left = SequentialCompactForm(dim, left_degree, 1.0);
                    auto right = SequentialCompactForm(dim, right_degree, 7.0);
                    auto compact_wedge = Wedge(left, right);
                    auto dense_wedge = WedgeDenseKForms(
                        DenseCopy(left), DenseCopy(right));
                    Require(TryGetCompactView(*compact_wedge).has_value(),
                            "compact wedge lost compact storage");
                    auto wedge_context =
                        "compact wedge differs from dense wedge for dim=" +
                        ToString(dim) + ", degrees=" +
                        ToString(left_degree) + "," + ToString(right_degree);
                    RequireSameValues(compact_wedge, dense_wedge,
                                      wedge_context.c_str());
                    RequireSameValues(Compile(compact_wedge, false, 0, true),
                                      dense_wedge,
                                      "graph-compiled compact wedge changed values");
                    if (dim == 2 && left_degree == 1 && right_degree == 1)
                        RequireSameValues(
                            Compile(compact_wedge, true, 0, true), dense_wedge,
                            "native-compiled compact wedge changed values");

                    auto reverse = Wedge(right, left);
                    const double sign =
                        (left_degree * right_degree) % 2 ? -1.0 : 1.0;
                    RequireSameValues(compact_wedge,
                                      ScaleKForm(reverse, ConstantCF(sign)),
                                      "compact wedge violates graded commutativity");
                }

        auto a = SequentialCompactForm(4, 1, 1.0);
        auto b = SequentialCompactForm(4, 1, 5.0);
        auto c = SequentialCompactForm(4, 1, 11.0);
        auto two_a = Wedge(a, b);
        RequireSameValues(Wedge(Wedge(a, b), c), Wedge(a, Wedge(b, c)),
                          "compact wedge is not associative");

        auto sum = AddKForms(a, b);
        auto difference = AddKForms(a, b, true);
        auto scaled = ScaleKForm(a, ConstantCF(3.0));
        Require(TryGetCompactView(*sum).has_value() &&
                    TryGetCompactView(*difference).has_value() &&
                    TryGetCompactView(*scaled).has_value(),
                "compact arithmetic lost compact storage");
        RequireSameValues(sum, SymbolicSumCF(a, b),
                          "compact addition changed values");
        RequireSameValues(
            difference,
            SymbolicSumCF(a, ScaleCoefficientCF(b, ConstantCF(-1.0))),
            "compact subtraction changed values");
        RequireSameValues(scaled, ScaleCoefficientCF(a, ConstantCF(3.0)),
                          "compact scaling changed values");

        auto dense_a = DenseCopy(a);
        auto dense_b = DenseCopy(b);
        Require(!TryGetCompactView(*dense_a).has_value() &&
                    !TryGetCompactView(*dense_b).has_value(),
                "dense one-form fixture is unexpectedly compact");
        Require(TryGetCompactView(*AddKForms(dense_a, dense_b)).has_value() &&
                    TryGetCompactView(*ScaleKForm(dense_a, ConstantCF(2.0))).has_value() &&
                    TryGetCompactView(*Wedge(dense_a, dense_b)).has_value(),
                "intrinsically alternating one-form arithmetic stayed dense");
        RequireSameValues(Wedge(dense_a, dense_b), Wedge(a, b),
                          "one-form compact promotion changed wedge values");

        auto dense_nonalternating = KFormCF(
            ConstantVector({1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16})
                ->Reshape(Array<int>{4, 4}),
            2, 4);
        auto dense_one_form = KFormCF(
            ConstantVector({1.0, 2.0, 3.0, 4.0})
                ->Reshape(Array<int>{4}),
            1, 4);
        Require(!TryGetCompactView(*AddKForms(two_a, dense_nonalternating)).has_value(),
                "mixed addition did not use dense fallback");
        Require(!TryGetCompactView(*Wedge(dense_nonalternating, dense_one_form)).has_value(),
                "mixed wedge did not use dense fallback");

        auto two_b = Wedge(b, c);
        auto compact_chain = AddKForms(
            ScaleKForm(two_a, ConstantCF(2.0)), two_b, true);
        bool saw_dense_alternation = false;
        bool saw_full_operand = false;
        compact_chain->GetFullCoefficient()->TraverseDAG(
            [&](CoefficientFunction &node) {
                saw_dense_alternation |=
                    node.GetDescription() == "AlternationCF" ||
                    node.GetDescription() == "BlockAlternationCF";
                saw_full_operand |=
                    &node == two_a->GetFullCoefficient().get() ||
                    &node == two_b->GetFullCoefficient().get();
            });
        Require(!saw_dense_alternation && !saw_full_operand,
                "compact arithmetic evaluates a dense intermediate form");

        auto arbitrary_two_direction =
            ConstantVector({1, 2, 3, 4,
                            5, 6, 7, 8,
                            9, 10, 11, 12,
                            13, 14, 15, 16})
                ->Reshape(Array<int>{4, 4});
        auto wedge = Wedge(two_a, c);
        auto differentiated = wedge->Diff(two_a.get(), arbitrary_two_direction);
        RequireSameValues(
            differentiated,
            Wedge(KFormCF(arbitrary_two_direction, 2, 4), c),
            "compact wedge compressed a non-alternating direction");
        auto differentiated_form = KFormCF(differentiated, 3, 4);
        Require(!TryGetCompactView(*differentiated_form).has_value(),
                "non-alternating wedge direction remained marked compact");

        CoefficientFunction::T_Transform replacement;
        replacement.replace[two_a] = arbitrary_two_direction;
        auto replaced = KFormCF(wedge->Transform(replacement), 3, 4);
        Require(!TryGetCompactView(*replaced).has_value(),
                "non-alternating wedge replacement remained compact");
        RequireSameValues(
            replaced, Wedge(KFormCF(arbitrary_two_direction, 2, 4), c),
                          "compact wedge replacement changed values");

        auto factor = make_shared<ConstantCoefficientFunction>(2.0);
        auto factor_scaled = ScaleKForm(a, factor);
        CoefficientFunction::T_Transform factor_replacement;
        factor_replacement.replace[factor] = ConstantCF(5.0);
        RequireSameValues(factor_scaled->Transform(factor_replacement),
                          ScaleCoefficientCF(a, ConstantCF(5.0)),
                          "compact scale factor replacement changed values");
        RequireSameValues(factor_scaled->Diff(factor.get(), ConstantCF(3.0)),
                          ScaleCoefficientCF(a, ConstantCF(3.0)),
                          "compact scale factor derivative changed values");
        CoefficientFunction::T_DJC factor_jacobian_cache;
        auto factor_jacobian = factor_scaled->GetFullCoefficient()->DiffJacobi(
            factor.get(), factor_jacobian_cache);
        auto factor_jacobian_form = KFormCF(factor_jacobian, 1, 4);
        Require(TryGetCompactView(*factor_jacobian_form).has_value(),
                "compact scale factor Jacobian lost compact storage");
        RequireSameValues(factor_jacobian, a,
                          "compact scale factor Jacobian changed values");

        auto stream = make_shared<std::stringstream>();
        {
            BinaryOutArchive archive(stream);
            shared_ptr<CoefficientFunction> stored = compact_chain;
            archive & stored;
        }
        shared_ptr<CoefficientFunction> restored;
        {
            auto input = make_shared<std::stringstream>(stream->str());
            BinaryInArchive archive(input);
            archive & restored;
        }
        auto restored_form = dynamic_pointer_cast<KFormCoefficientFunction>(restored);
        Require(restored_form && TryGetCompactView(*restored_form).has_value(),
                "archive lost compact arithmetic state");
        RequireSameValues(restored, compact_chain,
                          "archived compact arithmetic changed values");
    }

    void TestCompactDoubleFormArithmetic()
    {
        using namespace ngfem::kforms_internal;
        for (int dim = 1; dim <= 4; ++dim)
            for (int p = 0; p <= dim; ++p)
                for (int r = 0; r <= dim - p; ++r)
                    for (int q = 0; q <= dim; ++q)
                        for (int s = 0; s <= dim - q; ++s)
                        {
                            auto left = SequentialCompactDoubleForm(
                                dim, p, q, 1.0);
                            auto right = SequentialCompactDoubleForm(
                                dim, r, s, 7.0);
                            auto compact = Wedge(left, right);
                            auto compact_view = TryGetCompactView(*compact);
                            Require(compact_view.has_value(),
                                    "compact double wedge lost compact storage");

                            if (p + q + r + s <= 4)
                            {
                                auto dense = WedgeDenseDoubleForms(
                                    left, right);
                                RequireSameValues(
                                    compact, dense,
                                    "compact double wedge differs from dense wedge");
                                RequireSameValues(
                                    Compile(compact, false, 0, true), dense,
                                    "compiled compact double wedge changed values");
                                if (dim == 2 && p == 1 && q == 0 &&
                                    r == 1 && s == 0)
                                    RequireSameValues(
                                        Compile(compact, true, 0, true), dense,
                                        "native-compiled compact double wedge changed values");
                            }

                            auto reverse = Wedge(right, left);
                            const double sign =
                                (p * r + q * s) % 2 ? -1.0 : 1.0;
                            auto signed_reverse = ScaleDoubleForm(
                                reverse, ConstantCF(sign));
                            auto reverse_view =
                                TryGetCompactView(*signed_reverse);
                            Require(reverse_view.has_value(),
                                    "reversed double wedge lost compact storage");
                            RequireSameValues(
                                compact_view->independent,
                                reverse_view->independent,
                                "compact double wedge violates blockwise graded commutativity");
                        }

        auto a = SequentialCompactDoubleForm(3, 1, 1, 1.0);
        auto b = SequentialCompactDoubleForm(3, 1, 0, 5.0);
        auto c = SequentialCompactDoubleForm(3, 0, 1, 11.0);
        RequireSameValues(
            Wedge(Wedge(a, b), c), Wedge(a, Wedge(b, c)),
            "compact double wedge is not associative");

        auto left = SequentialCompactDoubleForm(4, 2, 2, 1.0);
        auto right = SequentialCompactDoubleForm(4, 2, 2, 9.0);
        auto sum = AddDoubleForms(left, right);
        auto difference = AddDoubleForms(left, right, true);
        auto scaled = ScaleDoubleForm(left, ConstantCF(3.0));
        Require(TryGetCompactView(*sum) &&
                    TryGetCompactView(*difference) &&
                    TryGetCompactView(*scaled),
                "compact double-form arithmetic lost compact storage");
        RequireSameValues(sum, SymbolicSumCF(left, right),
                          "compact double-form addition changed values");
        RequireSameValues(
            difference,
            SymbolicSumCF(
                left, ScaleCoefficientCF(right, ConstantCF(-1.0))),
            "compact double-form subtraction changed values");
        RequireSameValues(
            scaled, ScaleCoefficientCF(left, ConstantCF(3.0)),
            "compact double-form scaling changed values");

        auto dense_11_a = DenseDoubleForm(3, 1, 1, 1.0);
        auto dense_11_b = DenseDoubleForm(3, 1, 1, 10.0);
        Require(!TryGetCompactView(*dense_11_a) &&
                    !TryGetCompactView(*dense_11_b),
                "dense (1,1) fixture is unexpectedly compact");
        Require(TryGetCompactView(
                    *AddDoubleForms(dense_11_a, dense_11_b)) &&
                    TryGetCompactView(
                        *ScaleDoubleForm(dense_11_a, ConstantCF(2.0))) &&
                    TryGetCompactView(*Wedge(dense_11_a, dense_11_b)) &&
                    TryGetCompactView(
                        *SwapDoubleFormSlots(dense_11_a)),
                "intrinsically alternating (1,1) operations stayed dense");
        RequireSameValues(
            Wedge(dense_11_a, dense_11_b),
            WedgeDenseDoubleForms(dense_11_a, dense_11_b),
            "(1,1) compact promotion changed wedge values");
        auto fused_seed = Wedge(dense_11_a, dense_11_b);
        auto fused_seed_evaluator = NativeCoefficientValue(fused_seed);
        Require(
            fused_seed_evaluator->GetDescription() ==
                "CompactDoubleWedgeFullCF",
            "compact seed wedge does not use the fused full-output evaluator");
        for (auto input : fused_seed_evaluator->InputCoefficientFunctions())
            Require(input->Dimension() == dense_11_a->Dimension(),
                    "fused seed wedge input is not independent-sized");

        auto dense_22 = DenseDoubleForm(4, 2, 2, 1.0);
        Require(!TryGetCompactView(*AddDoubleForms(left, dense_22)),
                "mixed double-form addition did not use dense fallback");
        auto dense_20 = DenseDoubleForm(4, 2, 0, 3.0);
        auto compact_01 = SequentialCompactDoubleForm(4, 0, 1, 2.0);
        Require(!TryGetCompactView(*Wedge(dense_20, compact_01)),
                "mixed double wedge did not use dense fallback");
        Require(!TryGetCompactView(*SwapDoubleFormSlots(dense_20)),
                "unchecked double-form swap did not use dense fallback");

        auto two_one = Wedge(a, b);
        auto chained_wedge = Wedge(two_one, c);
        auto chained_evaluator = NativeCoefficientValue(chained_wedge);
        Require(
            chained_evaluator->GetDescription() ==
                "CompactDoubleWedgeFullCF",
            "compact wedge chain does not use the fused full-output evaluator");
        for (auto input : chained_evaluator->InputCoefficientFunctions())
            Require(input->Dimension() < two_one->Dimension(),
                    "compact wedge chain exposes a full intermediate input");
        auto compact_chain = AddDoubleForms(
            ScaleDoubleForm(two_one, ConstantCF(2.0)),
            Wedge(a, b), true);
        bool saw_dense_wedge = false;
        bool saw_full_operand = false;
        compact_chain->GetFullCoefficient()->TraverseDAG(
            [&](CoefficientFunction &node) {
                saw_dense_wedge |=
                    node.GetDescription() == "DoubleFormWedgeCF";
                saw_full_operand |=
                    &node == two_one->GetFullCoefficient().get();
            });
        Require(!saw_dense_wedge && !saw_full_operand,
                "compact double arithmetic evaluates a dense intermediate");

        auto swapped = SwapDoubleFormSlots(two_one);
        auto restored_slots = SwapDoubleFormSlots(swapped);
        Require(TryGetCompactView(*swapped) &&
                    TryGetCompactView(*restored_slots),
                "compact slot swap lost compact storage");
        RequireSameValues(restored_slots, two_one,
                          "swapping double-form slots twice changed values");

        auto arbitrary_direction = DenseDoubleForm(3, 2, 1, 2.0)
                                       ->GetFullCoefficient();
        auto wedge = Wedge(two_one, c);
        auto differentiated = DoubleFormCF(
            wedge->Diff(two_one.get(), arbitrary_direction), 2, 2, 3);
        Require(!TryGetCompactView(*differentiated),
                "non-alternating double-wedge direction stayed compact");
        RequireSameValues(
            differentiated,
            WedgeDenseDoubleForms(
                DoubleFormCF(arbitrary_direction, 2, 1, 3), c),
            "compact double-wedge direction changed values");

        CoefficientFunction::T_Transform replacement;
        replacement.replace[two_one] = arbitrary_direction;
        auto replaced = DoubleFormCF(
            wedge->Transform(replacement), 2, 2, 3);
        Require(!TryGetCompactView(*replaced),
                "non-alternating double-wedge replacement stayed compact");
        RequireSameValues(
            replaced,
            WedgeDenseDoubleForms(
                DoubleFormCF(arbitrary_direction, 2, 1, 3), c),
            "compact double-wedge replacement changed values");

        auto factor = make_shared<ConstantCoefficientFunction>(2.0);
        auto factor_scaled = ScaleDoubleForm(two_one, factor);
        CoefficientFunction::T_Transform factor_replacement;
        factor_replacement.replace[factor] = ConstantCF(5.0);
        auto transformed_factor = factor_scaled->Transform(
            factor_replacement);
        auto transformed_form = dynamic_pointer_cast<
            DoubleFormCoefficientFunction>(transformed_factor);
        Require(transformed_form && TryGetCompactView(*transformed_form),
                "double-form factor replacement lost compact storage");
        RequireSameValues(
            transformed_form,
            ScaleCoefficientCF(two_one, ConstantCF(5.0)),
            "double-form factor replacement changed values");
        RequireSameValues(
            factor_scaled->Diff(factor.get(), ConstantCF(3.0)),
            ScaleCoefficientCF(two_one, ConstantCF(3.0)),
            "double-form factor derivative changed values");
        CoefficientFunction::T_DJC factor_cache;
        auto factor_jacobian =
            factor_scaled->GetFullCoefficient()->DiffJacobi(
                factor.get(), factor_cache);
        auto factor_jacobian_form = DoubleFormCF(
            factor_jacobian, two_one->LeftDegree(),
            two_one->RightDegree(), 3);
        Require(TryGetCompactView(*factor_jacobian_form).has_value(),
                "double-form factor Jacobian lost compact storage");
        RequireSameValues(factor_jacobian, two_one,
                          "double-form factor Jacobian changed values");

        RequireSameValues(
            Compile(compact_chain, false, 0, true), compact_chain,
            "compiled compact double-form chain changed values");
        auto stream = make_shared<std::stringstream>();
        {
            BinaryOutArchive archive(stream);
            shared_ptr<CoefficientFunction> stored = compact_chain;
            archive & stored;
        }
        shared_ptr<CoefficientFunction> restored;
        {
            auto input = make_shared<std::stringstream>(stream->str());
            BinaryInArchive archive(input);
            archive & restored;
        }
        auto restored_form = dynamic_pointer_cast<
            DoubleFormCoefficientFunction>(restored);
        Require(restored_form && TryGetCompactView(*restored_form),
                "archive lost compact double-form arithmetic state");
        RequireSameValues(restored, compact_chain,
                          "archived compact double-form chain changed values");
    }

    shared_ptr<CoefficientFunction> SpatialCompactComponents(
        int dim, size_t count)
    {
        Array<shared_ptr<CoefficientFunction>> components;
        for (size_t component = 0; component < count; ++component)
        {
            shared_ptr<CoefficientFunction> value = ConstantCF(
                double(component + 1));
            for (int axis = 0; axis < dim; ++axis)
                value = value +
                        double((component + 1) * size_t(axis + 2)) *
                            MakeCoordinateCoefficientFunction(axis);
            components.Append(value);
        }
        if (count == 1)
            return components[0];
        return MakeVectorialCoefficientFunction(std::move(components));
    }

    shared_ptr<CoefficientFunction> SpatialDenseTensor(int dim, int rank)
    {
        size_t count = 1;
        for (int axis = 0; axis < rank; ++axis)
            count *= size_t(dim);
        auto value = SpatialCompactComponents(dim, count);
        Array<int> dimensions(static_cast<size_t>(rank));
        dimensions = dim;
        return value->Reshape(dimensions);
    }

    void TestCompactProvenZeroSimplification()
    {
        using namespace ngfem::kforms_internal;
        auto contains_native_node = [](
            const shared_ptr<CoefficientFunction> &value,
            const std::string &description) {
            bool found = false;
            value->TraverseDAG([&](CoefficientFunction &node) {
                found |= node.GetDescription() == description;
            });
            return found;
        };

        auto zero_one = ZeroKForm(1, 3);
        auto one = SequentialCompactForm(3, 1, 2.0);
        auto wedge = Wedge(zero_one, one);
        auto wedge_view = TryGetCompactView(*wedge);
        Require(wedge_view && IsConstantZero(wedge_view->independent),
                "zero K-form wedge retained a nonzero native evaluator");
        Require(!contains_native_node(
                    wedge->GetFullCoefficient(),
                    "CompactWedgeIndependentCF"),
                "zero K-form wedge retained the product kernel");
        auto one_direction = SpatialDenseTensor(3, 1);
        RequireSameSpatialValues(
            wedge->Diff(zero_one.get(), one_direction),
            Wedge(KFormCF(one_direction, 1, 3), one), 3,
            "zero K-form wedge discarded symbolic dependence");

        auto zero_double = ZeroDoubleForm(1, 0, 3);
        auto double_one = SequentialCompactDoubleForm(3, 1, 0, 3.0);
        auto double_wedge = Wedge(zero_double, double_one);
        auto double_view = TryGetCompactView(*double_wedge);
        Require(double_view && IsConstantZero(double_view->independent),
                "zero double-form wedge retained a nonzero native evaluator");
        Require(!contains_native_node(
                    double_wedge->GetFullCoefficient(),
                    "CompactDoubleWedgeFullCF") &&
                    !contains_native_node(
                        double_wedge->GetFullCoefficient(),
                        "CompactWedgeIndependentCF"),
                "zero double-form wedge retained a product kernel");
        RequireSameSpatialValues(
            double_wedge->Diff(zero_double.get(), one_direction),
            Wedge(DoubleFormCF(one_direction, 1, 0, 3), double_one), 3,
            "zero double-form wedge discarded symbolic dependence");

        auto exterior = ExteriorDerivative(zero_one);
        auto exterior_view = TryGetCompactView(*exterior);
        Require(exterior_view && IsConstantZero(exterior_view->independent),
                "exterior derivative of zero retained a nonzero evaluator");
        Require(!contains_native_node(
                    exterior->GetFullCoefficient(),
                    "CompactExteriorDerivativeIndependentCF"),
                "exterior derivative of zero retained its gradient kernel");
        RequireSameSpatialValues(
            exterior->Diff(zero_one.get(), one_direction),
            ExteriorDerivativeDenseKForm(
                KFormCF(one_direction, 1, 3)),
            3, "zero exterior derivative discarded symbolic dependence");

        auto zero_factor = make_shared<ConstantCoefficientFunction>(0.0);
        auto scaled = ScaleKForm(one, zero_factor);
        auto scaled_view = TryGetCompactView(*scaled);
        Require(scaled_view && IsConstantZero(scaled_view->independent),
                "zero scale factor retained a nonzero native evaluator");
        RequireSameValues(
            scaled->Diff(zero_factor.get(), ConstantCF(1.0)), one,
            "zero scale factor discarded symbolic dependence");
    }

    void TestCompactExteriorDerivative()
    {
        using namespace ngfem::kforms_internal;
        for (int dim = 1; dim <= 3; ++dim)
            for (int degree = 0; degree < dim; ++degree)
            {
                auto basis = GetFormBasis(dim, degree);
                auto form = KFormFromIndependentCF(
                    SpatialCompactComponents(dim, basis->IndependentSize()),
                    degree, dim);
                auto compact = ExteriorDerivative(form);
                auto dense = ExteriorDerivativeDenseKForm(DenseCopy(form));
                Require(TryGetCompactView(*compact).has_value(),
                        "exterior derivative lost compact storage");
                RequireSameSpatialValues(
                    compact, dense, dim,
                    "compact exterior derivative differs from dense reference");
                RequireSameSpatialValues(
                    Compile(compact, false, 0, true), dense, dim,
                    "compiled compact exterior derivative changed values");
                const auto native_message =
                    "native-compiled compact exterior derivative changed "
                    "values for dim=" + ToString(dim) +
                    ", degree=" + ToString(degree);
                RequireSameSpatialValues(
                    Compile(compact, true, 0, true), dense, dim,
                    native_message.c_str());

                bool saw_dense_alternation = false;
                compact->GetFullCoefficient()->TraverseDAG(
                    [&](CoefficientFunction &node) {
                        saw_dense_alternation |=
                            node.GetDescription() == "AlternationCF";
                    });
                Require(!saw_dense_alternation,
                        "compact exterior derivative evaluates dense alternation");

                if (degree + 2 <= dim)
                {
                    auto second = ExteriorDerivative(compact);
                    Require(TryGetCompactView(*second).has_value(),
                            "second exterior derivative lost compact storage");
                    Array<int> dimensions(size_t(degree + 2));
                    dimensions = dim;
                    RequireSameSpatialValues(
                        second, ZeroCF(dimensions), dim,
                        "compact exterior derivative does not square to zero");
                }
            }

        auto parameter = make_shared<ConstantCoefficientFunction>(2.0);
        auto parameter_input = KFormFromIndependentCF(
            SpatialCompactComponents(3, GetFormBasis(3, 1)->IndependentSize()),
            1, 3);
        auto parameter_derivative = ExteriorDerivative(
            ScaleKForm(parameter_input, parameter));
        CoefficientFunction::T_DJC parameter_cache;
        auto parameter_jacobian =
            parameter_derivative->GetFullCoefficient()->DiffJacobi(
                parameter.get(), parameter_cache);
        auto parameter_jacobian_form = KFormCF(parameter_jacobian, 2, 3);
        Require(TryGetCompactView(*parameter_jacobian_form).has_value(),
                "compact exterior scalar Jacobian lost compact storage");
        RequireSameSpatialValues(
            parameter_jacobian, ExteriorDerivative(parameter_input), 3,
            "compact exterior scalar Jacobian changed values");

        auto alternating_input = KFormFromIndependentCF(
            SpatialCompactComponents(3, GetFormBasis(3, 2)->IndependentSize()),
            2, 3);
        auto compact = ExteriorDerivative(alternating_input);
        auto arbitrary_direction = SpatialDenseTensor(3, 2);

        CoefficientFunction::T_Transform replacement;
        replacement.replace[alternating_input] = arbitrary_direction;
        auto replaced = KFormCF(compact->Transform(replacement), 3, 3);
        Require(!TryGetCompactView(*replaced).has_value(),
                "non-alternating exterior-derivative replacement stayed compact");
        RequireSameSpatialValues(
            replaced,
            ExteriorDerivativeDenseKForm(
                KFormCF(arbitrary_direction, 2, 3)),
            3, "exterior-derivative replacement changed values");

        auto differentiated = KFormCF(
            compact->Diff(alternating_input.get(), arbitrary_direction), 3, 3);
        Require(!TryGetCompactView(*differentiated).has_value(),
                "non-alternating exterior-derivative direction stayed compact");
        RequireSameSpatialValues(
            differentiated,
            ExteriorDerivativeDenseKForm(
                KFormCF(arbitrary_direction, 2, 3)),
            3, "exterior-derivative direction changed values");

        auto dense_input = KFormCF(SpatialDenseTensor(3, 2), 2, 3);
        auto dense_result = ExteriorDerivative(dense_input);
        Require(!TryGetCompactView(*dense_result).has_value(),
                "unchecked exterior-derivative input was silently compressed");

        auto stream = make_shared<std::stringstream>();
        {
            BinaryOutArchive archive(stream);
            shared_ptr<CoefficientFunction> stored = compact;
            archive & stored;
        }
        shared_ptr<CoefficientFunction> restored;
        {
            auto input = make_shared<std::stringstream>(stream->str());
            BinaryInArchive archive(input);
            archive & restored;
        }
        auto restored_form = dynamic_pointer_cast<KFormCoefficientFunction>(
            restored);
        Require(restored_form &&
                    TryGetCompactView(*restored_form).has_value(),
                "archive lost compact exterior-derivative state");
        RequireSameSpatialValues(
            restored, compact, 3,
            "archived compact exterior derivative changed values");
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
        {"simd", TestSIMD},
        {"compact_symbolics", TestCompactKFormSymbolics},
        {"form_basis", TestFormBasis},
        {"compact_facade", TestCompactKFormFacade},
        {"compact_double_facade", TestCompactDoubleFormFacade},
        {"compact_arithmetic", TestCompactKFormArithmetic},
        {"compact_double_arithmetic", TestCompactDoubleFormArithmetic},
        {"compact_zero", TestCompactProvenZeroSimplification},
        {"compact_exterior", TestCompactExteriorDerivative}
    };
    if (argc > 2)
    {
        std::cerr << "usage: " << argv[0] << " [mode]" << std::endl;
        return 2;
    }
    if (argc == 2)
    {
        const std::string requested = argv[1];
        const bool known = std::any_of(
            std::begin(checks), std::end(checks),
            [&](const auto &check) { return requested == check.first; });
        if (!known)
        {
            std::cerr << "unknown test mode: " << requested << std::endl;
            return 2;
        }
    }
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
