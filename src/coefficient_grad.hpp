#ifndef COEFFICIENT_GRAD
#define COEFFICIENT_GRAD

#include <coefficient.hpp>
#include <diffop.hpp>
#include <symbolicintegrator.hpp>

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

/**
 * @file coefficient_grad.hpp
 * @brief Numerical gradients for general coefficient functions and proxies.
 */

namespace ngfem
{
    /**
     * Differentiate a coefficient function with respect to physical coordinates.
     *
     * The result dimensions are ``(dim, *cf->Dimensions())``: the derivative
     * direction is always the first component axis. Proxy expressions use
     * NGSolve's symbolic operator path. Pure coefficient graphs use
     * GradCoefficientFunction.
     */
    std::shared_ptr<CoefficientFunction> GradCF(
        const std::shared_ptr<CoefficientFunction> &cf,
        int dim,
        bool surface = false);

    /**
     * Return the physical Hessian with two derivative axes prepended.
     *
     * Pure coefficient graphs may be tensor valued. Direct VectorH1 and
     * H1(dim=...) proxies are supported. Composite proxy expressions require
     * a native NGSolve Hessian operator for the complete expression.
     */
    std::shared_ptr<CoefficientFunction> HesseCF(
        const std::shared_ptr<CoefficientFunction> &cf,
        size_t dim,
        bool boundary = false);

    /**
     * Numerical gradient of an arbitrary coefficient-function value graph.
     *
     * Derivatives are computed by a fourth-order centered stencil in reference
     * coordinates and mapped to physical coordinates. In surface mode the
     * reference derivative is mapped through the tangential pseudoinverse.
     */
    template <int D>
    class GradCoefficientFunction
        : public T_CoefficientFunction<GradCoefficientFunction<D>>
    {
        static_assert(D >= 1 && D <= 3, "GradCF supports dimensions 1,2,3");

        using BASE = T_CoefficientFunction<GradCoefficientFunction<D>>;

        std::shared_ptr<CoefficientFunction> c1;
        bool surface;

        static constexpr size_t stencil_size = 4;
        static constexpr std::array<int, stencil_size> stencil_offsets = {
            -1, 1, -2, 2};

        static void FillCenteredStencil(
            IntegrationRule &rule,
            size_t offset,
            const IntegrationPoint &point,
            int direction)
        {
            for (size_t stencil_point = 0;
                 stencil_point < stencil_size;
                 ++stencil_point)
            {
                rule[offset + stencil_point] = point;
                rule[offset + stencil_point](direction) +=
                    stencil_offsets[stencil_point] * eps();
            }
        }

        template <typename T>
        static T CenteredDerivative(
            T minus_h,
            T plus_h,
            T minus_2h,
            T plus_2h)
        {
            return (8.0 * plus_h - 8.0 * minus_h - plus_2h + minus_2h) / (12.0 * eps());
        }

        static const CoefficientFunction &CheckedInput(
            const std::shared_ptr<CoefficientFunction> &cf)
        {
            if (!cf)
                throw Exception("GradCF: input coefficient is null");
            return *cf;
        }

        template <typename MAPSCAL, typename T, ORDERING ORD>
        void T_EvaluateMapped(
            const BaseMappedIntegrationRule &bmir,
            BareSliceMatrix<T, ORD> values) const
        {
            auto &lh = TLHeap();
            HeapReset hr_lh(lh);
            const size_t hd = c1->Dimension();

            if (!surface)
            {
                const auto &mir = static_cast<
                    const MappedIntegrationRule<D, D, MAPSCAL> &>(bmir);
                const auto &ir = mir.IR();
                const size_t nip = mir.Size();

                FlatMatrix<T, ORD> values_c1(hd, stencil_size * nip, lh);
                FlatMatrix<T, ORD> derivatives(hd * nip, D, lh);

                // Batch all four stencil points for one reference direction.
                // This reduces child evaluations from D*nip to D per element.
                for (int direction = 0; direction < D; ++direction)
                {
                    HeapReset hr(lh);
                    IntegrationRule perturbed_rule(stencil_size * nip, lh);
                    for (size_t i = 0; i < nip; ++i)
                        FillCenteredStencil(
                            perturbed_rule, stencil_size * i, ir[i], direction);

                    MappedIntegrationRule<D, D, MAPSCAL> perturbed_mir(
                        perturbed_rule, mir.GetTransformation(), lh);
                    {
                        // Cached values belong to the original rule, not to
                        // the finite-difference stencil points.
                        auto reset_userdata =
                            mir.GetTransformation().PushUserData();
                        c1->Evaluate(perturbed_mir, values_c1);
                    }

                    for (size_t i = 0; i < nip; ++i)
                        for (size_t component = 0; component < hd; ++component)
                            derivatives(i * hd + component, direction) =
                                CenteredDerivative(
                                    values_c1(component, stencil_size * i),
                                    values_c1(component, stencil_size * i + 1),
                                    values_c1(component, stencil_size * i + 2),
                                    values_c1(component, stencil_size * i + 3));
                }

                for (size_t i = 0; i < nip; ++i)
                {
                    const auto jacobian_inverse = mir[i].GetJacobianInverse();
                    for (size_t component = 0; component < hd; ++component)
                        for (int physical_direction = 0;
                             physical_direction < D;
                             ++physical_direction)
                        {
                            T value = 0.0;
                            for (int reference_direction = 0;
                                 reference_direction < D;
                                 ++reference_direction)
                                value += derivatives(
                                             i * hd + component,
                                             reference_direction) *
                                         jacobian_inverse(
                                             reference_direction,
                                             physical_direction);
                            values(
                                // GradCF stores the derivative index first.
                                physical_direction * hd + component,
                                i) = value;
                        }
                }
                return;
            }

            if constexpr (D < 2)
            {
                throw Exception(
                    "GradCF(surface): only dimensions 2,3 supported");
            }
            else
            {
                const size_t reference_dim = D - 1;

                if (bmir.DimElement() == D - 1)
                {
                    // A boundary-element transformation exposes an intrinsic
                    // (D-1)-dimensional rule. Its Jacobian inverse already is
                    // the tangential pseudoinverse into ambient coordinates.
                    const auto &mir = static_cast<const MappedIntegrationRule<
                        D - 1, D, MAPSCAL> &>(bmir);
                    const auto &ir = mir.IR();
                    const size_t nip = mir.Size();
                    FlatMatrix<T, ORD> values_c1(
                        hd, stencil_size * nip, lh);
                    FlatMatrix<T, ORD> derivatives(
                        hd * nip, reference_dim, lh);

                    for (int direction = 0; direction < D - 1; ++direction)
                    {
                        HeapReset hr(lh);
                        IntegrationRule perturbed_rule(
                            stencil_size * nip, lh);
                        for (size_t i = 0; i < nip; ++i)
                            FillCenteredStencil(
                                perturbed_rule,
                                stencil_size * i,
                                ir[i],
                                direction);

                        MappedIntegrationRule<D - 1, D, MAPSCAL> perturbed_mir(
                            perturbed_rule, mir.GetTransformation(), lh);
                        {
                            auto reset_userdata =
                                mir.GetTransformation().PushUserData();
                            c1->Evaluate(perturbed_mir, values_c1);
                        }

                        for (size_t i = 0; i < nip; ++i)
                            for (size_t component = 0;
                                 component < hd;
                                 ++component)
                                derivatives(i * hd + component, direction) =
                                    CenteredDerivative(
                                        values_c1(
                                            component, stencil_size * i),
                                        values_c1(
                                            component, stencil_size * i + 1),
                                        values_c1(
                                            component, stencil_size * i + 2),
                                        values_c1(
                                            component, stencil_size * i + 3));
                    }

                    for (size_t i = 0; i < nip; ++i)
                    {
                        const auto jacobian_inverse =
                            mir[i].GetJacobianInverse();
                        for (size_t component = 0; component < hd; ++component)
                            for (int physical_direction = 0;
                                 physical_direction < D;
                                 ++physical_direction)
                            {
                                T value = 0.0;
                                for (int reference_direction = 0;
                                     reference_direction < D - 1;
                                     ++reference_direction)
                                    value += derivatives(
                                                 i * hd + component,
                                                 reference_direction) *
                                             jacobian_inverse(
                                                 reference_direction,
                                                 physical_direction);
                                values(
                                    physical_direction * hd + component,
                                    i) = value;
                            }
                    }
                    return;
                }

                const auto &mir = static_cast<
                    const MappedIntegrationRule<D, D, MAPSCAL> &>(bmir);
                const auto &ir = mir.IR();
                const size_t nip = mir.Size();
                FlatMatrix<T, ORD> values_c1(hd, stencil_size * nip, lh);
                FlatMatrix<T, ORD> derivatives(
                    hd * nip, reference_dim, lh);
                Facet2ElementTrafo facet_to_element(
                    mir.GetTransformation().GetElementType());

                // Element-boundary integrators pass points which have already
                // been mapped from the facet into the volume reference
                // element. Recover their intrinsic facet coordinates before
                // applying tangential finite-difference offsets.
                IntegrationRule facet_rule(nip, lh);
                for (size_t i = 0; i < nip; ++i)
                {
                    HeapReset hr(lh);
                    const IntegrationPoint &volume_ip = ir[i];
                    const int facet_number = volume_ip.FacetNr();
                    if (facet_number < 0)
                        throw Exception(
                            "GradCF(surface): missing FacetNr for "
                            "boundary evaluation");

                    IntegrationPoint facet_origin;
                    const IntegrationPoint volume_origin =
                        facet_to_element(facet_number, facet_origin);
                    Mat<D, D - 1> reference_jacobian =
                        facet_to_element.GetJacobian(facet_number, lh);
                    Mat<D - 1, D, double> reference_inverse =
                        Inv(Trans(reference_jacobian) * reference_jacobian) * Trans(reference_jacobian);
                    Vec<D> reference_offset;
                    for (int direction = 0; direction < D; ++direction)
                        reference_offset(direction) =
                            volume_ip(direction) - volume_origin(direction);
                    Vec<D - 1> facet_coordinates =
                        reference_inverse * reference_offset;

                    IntegrationPoint facet_ip;
                    for (int direction = 0; direction < D - 1; ++direction)
                        facet_ip(direction) = facet_coordinates(direction);
                    facet_ip.SetWeight(volume_ip.Weight());
                    facet_ip.SetFacetNr(facet_number);
                    facet_rule[i] = facet_ip;
                }

                // Element-boundary integration uses a D-dimensional volume
                // transformation. Perturb the recovered facet-local point,
                // then map each stencil point into the volume exactly once.
                for (int direction = 0; direction < D - 1; ++direction)
                {
                    HeapReset hr(lh);
                    IntegrationRule perturbed_volume_rule(
                        stencil_size * nip, lh);

                    for (size_t i = 0; i < nip; ++i)
                    {
                        const IntegrationPoint &facet_ip = facet_rule[i];
                        const int facet_number = facet_ip.FacetNr();

                        IntegrationRule perturbed_facet_rule(
                            stencil_size, lh);
                        FillCenteredStencil(
                            perturbed_facet_rule, 0, facet_ip, direction);
                        const IntegrationRule &mapped_rule = facet_to_element(
                            facet_number, perturbed_facet_rule, lh);
                        for (size_t point = 0;
                             point < stencil_size;
                             ++point)
                            perturbed_volume_rule[stencil_size * i + point] =
                                mapped_rule[point];
                    }

                    MappedIntegrationRule<D, D, MAPSCAL> perturbed_mir(
                        perturbed_volume_rule,
                        mir.GetTransformation(),
                        lh);
                    {
                        auto reset_userdata =
                            mir.GetTransformation().PushUserData();
                        c1->Evaluate(perturbed_mir, values_c1);
                    }

                    for (size_t i = 0; i < nip; ++i)
                        for (size_t component = 0; component < hd; ++component)
                            derivatives(i * hd + component, direction) =
                                CenteredDerivative(
                                    values_c1(component, stencil_size * i),
                                    values_c1(component, stencil_size * i + 1),
                                    values_c1(component, stencil_size * i + 2),
                                    values_c1(component, stencil_size * i + 3));
                }

                for (size_t i = 0; i < nip; ++i)
                {
                    const int facet_number = ir[i].FacetNr();
                    Mat<D, D - 1, MAPSCAL> tangential_jacobian =
                        mir[i].GetJacobian() * facet_to_element.GetJacobian(facet_number, lh);
                    Mat<D - 1, D - 1, MAPSCAL> gram_matrix =
                        Trans(tangential_jacobian) * tangential_jacobian;
                    Mat<D - 1, D, MAPSCAL> tangential_inverse =
                        Inv(gram_matrix) * Trans(tangential_jacobian);

                    for (size_t component = 0; component < hd; ++component)
                        for (int physical_direction = 0;
                             physical_direction < D;
                             ++physical_direction)
                        {
                            T value = 0.0;
                            for (int reference_direction = 0;
                                 reference_direction < D - 1;
                                 ++reference_direction)
                                value += derivatives(
                                             i * hd + component,
                                             reference_direction) *
                                         tangential_inverse(
                                             reference_direction,
                                             physical_direction);
                            values(
                                physical_direction * hd + component,
                                i) = value;
                        }
                }
            }
        }

        template <ORDERING ORD>
        void T_EvaluateSIMDVolume(
            const SIMD_BaseMappedIntegrationRule &bmir,
            BareSliceMatrix<SIMD<double>, ORD> values) const
        {
            const auto &mir = static_cast<
                const SIMD_MappedIntegrationRule<D, D> &>(bmir);
            const auto &ir = mir.IR();
            const size_t packed_nip = mir.Size();
            const size_t hd = c1->Dimension();
            constexpr size_t lanes = SIMD<double>::Size();

            if (packed_nip > size_t(std::numeric_limits<int>::max()) / (stencil_size * lanes))
                throw Exception("GradCF: SIMD integration rule is too large");
            if (hd != 0 && packed_nip > std::numeric_limits<size_t>::max() / (stencil_size * hd))
                throw Exception("GradCF: SIMD value buffer is too large");

            auto &lh = TLHeap();
            HeapReset hr_lh(lh);
            FlatMatrix<SIMD<double>> values_c1(
                hd, stencil_size * packed_nip, lh);
            FlatMatrix<SIMD<double>> derivatives(
                hd * packed_nip, D, lh);

            // Keep the four stencil offsets in separate packed blocks. This
            // preserves SIMD lane layout when the scalar rule is repacked.
            for (int direction = 0; direction < D; ++direction)
            {
                HeapReset hr(lh);
                const int stencil_nip =
                    int(stencil_size * packed_nip * lanes);
                SIMD_IntegrationRule perturbed_rule(stencil_nip, lh);
                for (size_t i = 0; i < packed_nip; ++i)
                    for (size_t stencil_point = 0;
                         stencil_point < stencil_size;
                         ++stencil_point)
                    {
                        const size_t index =
                            stencil_point * packed_nip + i;
                        perturbed_rule[index] = ir[i];
                        perturbed_rule[index](direction) +=
                            stencil_offsets[stencil_point] * eps();
                    }

                SIMD_MappedIntegrationRule<D, D> perturbed_mir(
                    perturbed_rule,
                    mir.GetTransformation(),
                    lh);
                {
                    auto reset_userdata =
                        mir.GetTransformation().PushUserData();
                    c1->Evaluate(perturbed_mir, values_c1);
                }

                for (size_t i = 0; i < packed_nip; ++i)
                    for (size_t component = 0; component < hd; ++component)
                        derivatives(i * hd + component, direction) =
                            CenteredDerivative(
                                values_c1(component, i),
                                values_c1(component, packed_nip + i),
                                values_c1(component, 2 * packed_nip + i),
                                values_c1(component, 3 * packed_nip + i));
            }

            for (size_t i = 0; i < packed_nip; ++i)
            {
                const auto jacobian_inverse = mir[i].GetJacobianInverse();
                for (size_t component = 0; component < hd; ++component)
                    for (int physical_direction = 0;
                         physical_direction < D;
                         ++physical_direction)
                    {
                        SIMD<double> value = 0.0;
                        for (int reference_direction = 0;
                             reference_direction < D;
                             ++reference_direction)
                            value += derivatives(
                                         i * hd + component,
                                         reference_direction) *
                                     jacobian_inverse(
                                         reference_direction,
                                         physical_direction);
                        values(
                            physical_direction * hd + component,
                            i) = value;
                    }
            }
        }

    public:
        // Reference-space stencil step shared by scalar and SIMD evaluation.
        static constexpr double eps() { return 1e-4; }

        GradCoefficientFunction(
            std::shared_ptr<CoefficientFunction> ac1,
            bool asurface = false)
            : BASE(
                  CheckedInput(ac1).Dimension() * D,
                  CheckedInput(ac1).IsComplex()),
              c1(std::move(ac1)),
              surface(asurface)
        {
            if (surface && D < 2)
                throw Exception(
                    "GradCF(surface): only dimensions 2,3 supported");

            Array<int> tensor_dims(c1->Dimensions().Size() + 1);
            tensor_dims[0] = D;
            for (size_t i = 0; i < c1->Dimensions().Size(); ++i)
                tensor_dims[i + 1] = c1->Dimensions()[i];
            this->SetDimensions(tensor_dims);
            this->elementwise_constant = c1->ElementwiseConstant();
        }

        std::string GetDescription() const override
        {
            return surface ? "SurfaceGradCF" : "GradCF";
        }

        auto GetCArgs() const { return std::tuple{c1, surface}; }

        void DoArchive(Archive &ar) override
        {
            BASE::DoArchive(ar);
        }

        bool DefinedOn(const ElementTransformation &trafo) override
        {
            return c1->DefinedOn(trafo);
        }

        void CalcEquivalenceKey() override
        {
            this->equivalence_key =
                std::string(surface ? "SurfaceGradCF<" : "GradCF<") + ToString(D) + ">(" + c1->EquivalenceKey() + "," + ToString(this->Dimensions()) + ")";
        }

        void TraverseTree(
            const std::function<void(CoefficientFunction &)> &func) override
        {
            c1->TraverseTree(func);
            func(*this);
        }

        Array<std::shared_ptr<CoefficientFunction>>
        InputCoefficientFunctions() const override
        {
            return Array<std::shared_ptr<CoefficientFunction>>({c1});
        }

        void NonZeroPattern(
            const ProxyUserData &ud,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            // Dependency propagation is conservative: any input dependency may
            // contribute to every physical derivative of that component.
            Vector<AutoDiffDiff<1, NonZero>> input_values(c1->Dimension());
            c1->NonZeroPattern(ud, input_values);
            for (size_t direction = 0; direction < D; ++direction)
                for (size_t component = 0;
                     component < c1->Dimension();
                     ++component)
                    values[direction * c1->Dimension() + component] =
                        input_values[component];
        }

        void NonZeroPattern(
            const ProxyUserData &,
            FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> input,
            FlatVector<AutoDiffDiff<1, NonZero>> values) const override
        {
            for (size_t direction = 0; direction < D; ++direction)
                for (size_t component = 0;
                     component < c1->Dimension();
                     ++component)
                    values[direction * c1->Dimension() + component] =
                        input[0][component];
        }

        std::shared_ptr<CoefficientFunction> Transform(
            CoefficientFunction::T_Transform &transformation) const override
        {
            auto thisptr = std::const_pointer_cast<CoefficientFunction>(
                this->shared_from_this());
            if (transformation.cache.count(thisptr))
                return transformation.cache[thisptr];
            if (transformation.replace.count(thisptr))
                return transformation.replace[thisptr];
            auto newcf = GradCF(c1->Transform(transformation), D, surface);
            transformation.cache[thisptr] = newcf;
            return newcf;
        }

        using BASE::Evaluate;

        double Evaluate(const BaseMappedIntegrationPoint &) const override
        {
            throw Exception("GradCF: scalar evaluation is not supported");
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(
            const MIR &bmir,
            BareSliceMatrix<T, ORD> values) const
        {
            // Unsupported SIMD scalar types must use ExceptionNOSIMD so the
            // enclosing NGSolve integrator can retry its scalar path.
            if constexpr (std::is_same_v<MIR, SIMD_BaseMappedIntegrationRule>)
            {
                if constexpr (std::is_same_v<T, SIMD<double>>)
                {
                    if (surface)
                        throw ExceptionNOSIMD(
                            "GradCF(surface) has no native SIMD evaluator");
                    if (bmir.DimSpace() != D || bmir.DimElement() != D)
                        throw Exception(
                            "GradCF: SIMD mapped integration rule has "
                            "incompatible dimensions");
                    T_EvaluateSIMDVolume(bmir, values);
                }
                else
                    throw ExceptionNOSIMD(
                        "GradCF has no native SIMD evaluator for this scalar "
                        "type");
            }
            else if constexpr (
                std::is_same_v<T, double> || std::is_same_v<T, Complex>)
            {
                if (!surface)
                {
                    if (bmir.DimSpace() != D || bmir.DimElement() != D)
                        throw Exception(
                            "GradCF: mapped integration rule has incompatible "
                            "dimensions");
                }
                else
                {
                    if constexpr (D < 2)
                        throw Exception(
                            "GradCF(surface): only dimensions 2,3 supported");
                    if (bmir.DimSpace() != D || (bmir.DimElement() != D && bmir.DimElement() != D - 1))
                        throw Exception(
                            "GradCF(surface): mapped integration rule has "
                            "incompatible dimensions");
                }

                if (bmir.IsComplex())
                {
                    if constexpr (std::is_same_v<T, Complex>)
                        T_EvaluateMapped<Complex>(bmir, values);
                    else
                        throw Exception(
                            "GradCF: complex mapped integration rule requires "
                            "complex output");
                }
                else
                    T_EvaluateMapped<double>(bmir, values);
            }
            else
                throw Exception(
                    "GradCF: this scalar evaluation type is not supported");
        }

        template <typename MIR, typename T, ORDERING ORD>
        void T_Evaluate(
            const MIR &ir,
            FlatArray<BareSliceMatrix<T, ORD>>,
            BareSliceMatrix<T, ORD> values) const
        {
            T_Evaluate(ir, values);
        }

        std::shared_ptr<CoefficientFunction> Diff(
            const CoefficientFunction *var,
            std::shared_ptr<CoefficientFunction> dir) const override
        {
            if (this == var)
                return dir;
            return GradCF(c1->Diff(var, std::move(dir)), D, surface);
        }

        bool IsZeroCF() const override { return c1->IsZeroCF(); }
    };

}

#include <python_ngstd.hpp>
void ExportGradCF(py::module m);

#endif // COEFFICIENT_GRAD
