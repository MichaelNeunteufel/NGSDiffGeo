// Implementation fragment included only by ../kforms.cpp.
// Semantic wrapper construction, factories, and archive ownership.

namespace
{

    shared_ptr<KFormCoefficientFunction> WrapKFormImpl(shared_ptr<CoefficientFunction> cf, int k, int dim)
    {
        int used_dim = InferWrappedOrTensorDimension(cf, dim);
        ValidateFormSpaceDimension(used_dim, k == 0, "KFormCF");
        ValidateKFormInput(*cf, k);

        if (k == 0)
        {
            if (auto sf = ReuseIfCompatible<ScalarFieldCoefficientFunction>(cf, used_dim, "KFormCF"))
                return sf;
            return make_shared<ScalarFieldCoefficientFunction>(
                UnwrapTensorFieldFullValue(std::move(cf)), used_dim);
        }
        if (k == 1)
        {
            if (auto of = ReuseIfCompatible<OneFormCoefficientFunction>(cf, used_dim, "KFormCF"))
                return of;
            return make_shared<OneFormCoefficientFunction>(
                UnwrapTensorFieldFullValue(std::move(cf)), used_dim);
        }
        if (k == 2)
        {
            if (auto tf = ReuseIfCompatible<TwoFormCoefficientFunction>(cf, used_dim, "KFormCF"))
                return tf;
            return make_shared<TwoFormCoefficientFunction>(
                UnwrapTensorFieldFullValue(std::move(cf)), used_dim);
        }
        if (k == 3)
        {
            if (auto tf = ReuseIfCompatible<ThreeFormCoefficientFunction>(cf, used_dim, "KFormCF"))
                return tf;
            return make_shared<ThreeFormCoefficientFunction>(
                UnwrapTensorFieldFullValue(std::move(cf)), used_dim);
        }

        if (auto kf = dynamic_pointer_cast<KFormCoefficientFunction>(cf))
        {
            if (kf->Degree() != k)
                throw Exception("KFormCF: cannot reinterpret existing k-form with different degree");
            if (used_dim > 0 && kf->DimensionOfSpace() != used_dim)
                throw Exception("KFormCF: requested dim does not match wrapped dimension");
            return kf;
        }

        return make_shared<KFormCoefficientFunction>(
            UnwrapTensorFieldFullValue(std::move(cf)),
            k, used_dim);
    }

    shared_ptr<DoubleFormCoefficientFunction> WrapDoubleFormImpl(shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
    {
        int used_dim = InferWrappedOrTensorDimension(cf, dim);
        ValidateFormSpaceDimension(used_dim, p + q == 0, "DoubleFormCF");

        ValidateDoubleFormInput(*cf, p, q);

        if (auto df = dynamic_pointer_cast<DoubleFormCoefficientFunction>(cf))
        {
            if (df->LeftDegree() != p || df->RightDegree() != q)
                throw Exception("DoubleFormCF: cannot reinterpret existing DoubleForm with different left/right degrees");
            if (used_dim > 0 && df->DimensionOfSpace() != used_dim)
                throw Exception("DoubleFormCF: requested dim does not match wrapped double-form dimension");
            return df;
        }

        return make_shared<DoubleFormCoefficientFunction>(
            UnwrapTensorFieldFullValue(std::move(cf)),
            p, q, used_dim);
    }

    template <typename TWrapped>
    shared_ptr<TWrapped> WrapSpecializedKForm(shared_ptr<CoefficientFunction> cf, int k, int dim, const char *name)
    {
        int used_dim = InferWrappedOrTensorDimension(cf, dim);
        auto kf = KFormCF(cf, k, used_dim);
        auto wrapped = dynamic_pointer_cast<TWrapped>(kf);
        if (!wrapped)
            throw Exception(std::string(name) + ": internal type mismatch");
        return wrapped;
    }

} // namespace

KFormCoefficientFunction::KFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, int ak, int adim)
    : TensorFieldCoefficientFunction(ac1, std::string(size_t(CheckedKFormDegree(ak, "KFormCF")), '1')),
      degree(uint8_t(ak)), dim(CheckedFormSpaceDimension(adim, ak == 0, "KFormCF"))
{
    const auto &dims = ac1->Dimensions();
    if (dim > 0 && dims.Size() > 0 && dims[0] != dim)
        throw Exception("KFormCF: tensor dimensions must all equal dim. dim = " +
                        ToString(int(dim)) + ", but found dimension " +
                        ToString(int(dims[0])));

    if (dim > 0 && degree > dim && !ac1->IsZeroCF())
        throw Exception("KFormCF: degree exceeds dimension (only zero forms allowed in that case)");
}

shared_ptr<TensorFieldCoefficientFunction>
KFormCoefficientFunction::Rewrap(shared_ptr<CoefficientFunction> cf) const
{
    return KFormCF(std::move(cf), degree, dim);
}

DoubleFormCoefficientFunction::DoubleFormCoefficientFunction(shared_ptr<CoefficientFunction> ac1, int ap, int aq, int adim)
    : TensorFieldCoefficientFunction(ac1, std::string(size_t(CheckedDoubleFormRank(ap, aq, "DoubleFormCF")), '1')),
      degree_left(uint8_t(ap)), degree_right(uint8_t(aq)),
      dim(CheckedFormSpaceDimension(adim, ap == 0 && aq == 0, "DoubleFormCF"))
{
    if (ap + aq > MAX_FORM_RANK && !ac1->IsZeroCF())
        throw Exception("DoubleFormCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");

    const auto &dims = ac1->Dimensions();
    if (dim > 0 && dims.Size() > 0 && dims[0] != dim)
        throw Exception("DoubleFormCF: tensor dimensions must all equal dim. dim = " +
                        ToString(int(dim)) + ", but found dimension " +
                        ToString(int(dims[0])));

    if (dim > 0 && (ap > dim || aq > dim) && !ac1->IsZeroCF())
        throw Exception("DoubleFormCF: degree exceeds dimension (only zero forms allowed in that case)");
}

shared_ptr<TensorFieldCoefficientFunction>
DoubleFormCoefficientFunction::Rewrap(shared_ptr<CoefficientFunction> cf) const
{
    return DoubleFormCF(std::move(cf), degree_left, degree_right, dim);
}

ScalarFieldCoefficientFunction::ScalarFieldCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
    : KFormCoefficientFunction(
          cf, 0, CheckedFormSpaceDimension(dim, true, "ScalarFieldCF"))
{
}

OneFormCoefficientFunction::OneFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
    : KFormCoefficientFunction(
          cf, 1, CheckedFormSpaceDimension(dim, false, "OneFormCF"))
{
}

TwoFormCoefficientFunction::TwoFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
    : KFormCoefficientFunction(
          cf, 2, CheckedFormSpaceDimension(dim, false, "TwoFormCF"))
{
}

ThreeFormCoefficientFunction::ThreeFormCoefficientFunction(shared_ptr<CoefficientFunction> cf, int dim)
    : KFormCoefficientFunction(
          cf, 3, CheckedFormSpaceDimension(dim, false, "ThreeFormCF"))
{
}

// Public semantic wrappers own their archive registrations.
static ngcore::RegisterClassForArchive<KFormCoefficientFunction,
                                       TensorFieldCoefficientFunction>
    reg_kform_cf;
static ngcore::RegisterClassForArchive<DoubleFormCoefficientFunction,
                                       TensorFieldCoefficientFunction>
    reg_double_form_cf;
static ngcore::RegisterClassForArchive<ScalarFieldCoefficientFunction,
                                       KFormCoefficientFunction>
    reg_scalar_field_cf;
static ngcore::RegisterClassForArchive<OneFormCoefficientFunction,
                                       KFormCoefficientFunction>
    reg_one_form_cf;
static ngcore::RegisterClassForArchive<TwoFormCoefficientFunction,
                                       KFormCoefficientFunction>
    reg_two_form_cf;
static ngcore::RegisterClassForArchive<ThreeFormCoefficientFunction,
                                       KFormCoefficientFunction>
    reg_three_form_cf;

shared_ptr<KFormCoefficientFunction> KFormCF(shared_ptr<CoefficientFunction> cf, int k, int dim)
{
    cf = RequireNonNull(std::move(cf), "KFormCF");
    CheckedKFormDegree(k, "KFormCF");
    return WrapKFormImpl(cf, k, dim);
}

shared_ptr<DoubleFormCoefficientFunction> DoubleFormCF(shared_ptr<CoefficientFunction> cf, int p, int q, int dim)
{
    cf = RequireNonNull(std::move(cf), "DoubleFormCF");
    const int rank = CheckedDoubleFormRank(p, q, "DoubleFormCF");
    if (rank > MAX_FORM_RANK && !cf->IsZeroCF())
        throw Exception("DoubleFormCF: only ranks up to " + ToString(MAX_FORM_RANK) + " are supported");
    return WrapDoubleFormImpl(cf, p, q, dim);
}
shared_ptr<ScalarFieldCoefficientFunction> ScalarFieldCF(shared_ptr<CoefficientFunction> cf, int dim)
{
    cf = RequireNonNull(std::move(cf), "ScalarFieldCF");
    return WrapSpecializedKForm<ScalarFieldCoefficientFunction>(cf, 0, dim, "ScalarFieldCF");
}

shared_ptr<OneFormCoefficientFunction> OneFormCF(shared_ptr<CoefficientFunction> cf)
{
    cf = RequireNonNull(std::move(cf), "OneFormCF");
    if (cf->Dimensions().Size() != 1)
        throw Exception("OneFormCF: input coefficient must be vector valued");
    return WrapSpecializedKForm<OneFormCoefficientFunction>(cf, 1, -1, "OneFormCF");
}

shared_ptr<TwoFormCoefficientFunction> TwoFormCF(shared_ptr<CoefficientFunction> cf, int dim)
{
    cf = RequireNonNull(std::move(cf), "TwoFormCF");
    return WrapSpecializedKForm<TwoFormCoefficientFunction>(cf, 2, dim, "TwoFormCF");
}

shared_ptr<ThreeFormCoefficientFunction> ThreeFormCF(shared_ptr<CoefficientFunction> cf, int dim)
{
    cf = RequireNonNull(std::move(cf), "ThreeFormCF");
    return WrapSpecializedKForm<ThreeFormCoefficientFunction>(cf, 3, dim, "ThreeFormCF");
}
