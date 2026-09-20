// Implementation fragment included only by ../kforms.cpp.
// Shared validation and code-generation helpers.

namespace
{
    template <typename T>
    shared_ptr<T> RequireNonNull(shared_ptr<T> ptr, const char *name)
    {
        if (!ptr)
            throw Exception(std::string(name) + ": input coefficient is null");
        return ptr;
    }

    void AccumulateSignedCodeExpression(
        CodeExpr &sum, CodeExpr value, int sign, size_t term)
    {
        if (term == 0)
        {
            sum = sign < 0
                      ? CodeExpr("-(" + value.S() + ")")
                      : std::move(value);
            return;
        }
        if (sign < 0)
            sum -= value;
        else
            sum += value;
    }

    template <typename T>
    int CheckedHodgeDimension(const shared_ptr<T> &form,
                              const RiemannianManifold &M, VorB vb,
                              const char *name)
    {
        RequireNonNull(form, name);
        if (form->DimensionOfSpace() != M.Dimension())
            throw Exception(std::string(name) +
                            ": form dimension does not match manifold dimension");
        if (vb != VOL && vb != BND && vb != BBND)
            throw Exception(std::string(name) +
                            ": only implemented for VOL, BND, and BBND");
        const int n = M.Dimension() - (vb == VOL ? 0 : (vb == BND ? 1 : 2));
        if (n < 0)
            throw Exception(std::string(name) + ": invalid manifold codimension");
        return n;
    }

    void ValidateDoubleFormSlot(int slot)
    {
        if (slot != -1 && slot != 0 && slot != 1)
            throw Exception("slot must be -1 (both), 0 (left), or 1 (right)");
    }

    int InferWrappedOrTensorDimension(shared_ptr<CoefficientFunction> cf, int dim)
    {
        if (dim > 0)
            return dim;
        if (auto kf = dynamic_pointer_cast<KFormCoefficientFunction>(cf))
            return kf->DimensionOfSpace();
        if (auto df = dynamic_pointer_cast<DoubleFormCoefficientFunction>(cf))
            return df->DimensionOfSpace();
        if (cf->Dimensions().Size() > 0)
            return int(cf->Dimensions()[0]);
        return 0;
    }

    void ValidateFormSpaceDimension(int used_dim, bool allow_unknown,
                                    const char *name)
    {
        if ((allow_unknown && used_dim == 0) ||
            (used_dim >= 1 && used_dim <= MAX_SPACE_DIM))
            return;
        throw Exception(std::string(name) + ": dim must be in {1,...," +
                        ToString(MAX_SPACE_DIM) +
                        "}" + (allow_unknown ? " (or 0 for scalar forms)" : ""));
    }

    uint8_t CheckedFormSpaceDimension(int dim, bool allow_unknown,
                                      const char *name)
    {
        ValidateFormSpaceDimension(dim, allow_unknown, name);
        return uint8_t(dim);
    }

    int CheckedKFormDegree(int k, const char *name)
    {
        if (k < 0)
            throw Exception(std::string(name) + ": degree must be non-negative");
        if (k > MAX_FORM_RANK)
            throw Exception(std::string(name) + ": only ranks up to " +
                            ToString(MAX_FORM_RANK) + " are supported");
        return k;
    }

    int CheckedDoubleFormRank(int p, int q, const char *name)
    {
        if (p < 0 || q < 0)
            throw Exception(std::string(name) + ": degrees must be non-negative");
        if (size_t(p) > MAX_SIGNATURE_LABELS ||
            size_t(q) > MAX_SIGNATURE_LABELS - size_t(p))
            throw Exception(std::string(name) + ": combined rank must not exceed " +
                            ToString(MAX_SIGNATURE_LABELS));
        return p + q;
    }

    void ValidateTensorComponentCount(int dim, int rank, const char *name)
    {
        size_t count = 1;
        for (int i = 0; i < rank; ++i)
        {
            if (count > std::numeric_limits<size_t>::max() / size_t(dim))
                throw Exception(std::string(name) +
                                ": tensor component count exceeds size_t");
            count *= size_t(dim);
        }
    }

    void ValidateKFormInput(const CoefficientFunction &cf, int k)
    {
        if (cf.Dimensions().Size() != size_t(k))
            throw Exception("KFormCF: degree k requires rank-k coefficient");
    }

    void ValidateDoubleFormInput(const CoefficientFunction &cf, int p, int q)
    {
        if (cf.Dimensions().Size() != size_t(p + q))
            throw Exception("DoubleFormCF: degrees p,q require rank-(p+q) coefficient");
    }

    template <typename TWrapped>
    shared_ptr<TWrapped> ReuseIfCompatible(shared_ptr<CoefficientFunction> cf, int used_dim, const char *name)
    {
        auto existing = dynamic_pointer_cast<TWrapped>(cf);
        if (!existing)
            return nullptr;
        if (existing->DimensionOfSpace() == 0 && used_dim > 0)
            return nullptr;
        if (used_dim > 0 && existing->DimensionOfSpace() != used_dim)
            throw Exception(std::string(name) + ": requested dim does not match wrapped dimension");
        return existing;
    }

    shared_ptr<CoefficientFunction>
    UnwrapTensorFieldFullValue(shared_ptr<CoefficientFunction> cf)
    {
        // Only the full semantic value crosses a wrapper boundary. Future
        // compact form storage must remain behind a full-shaped value node.
        while (auto tf =
                   dynamic_pointer_cast<TensorFieldCoefficientFunction>(cf))
            cf = tf->GetFullCoefficient();
        return cf;
    }

} // namespace
