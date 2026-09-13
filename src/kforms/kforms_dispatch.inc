// Implementation fragment included only by ../kforms.cpp.
// Public operation dispatch between compact and dense implementations.

shared_ptr<KFormCoefficientFunction> ZeroKForm(int k, int dim)
{
    CheckedKFormDegree(k, "ZeroKForm");
    ValidateFormSpaceDimension(dim, k == 0, "ZeroKForm");
    ValidateTensorComponentCount(dim, k, "ZeroKForm");
    if (dim > 0 && k <= dim)
    {
        auto basis = kforms_internal::GetFormBasis(dim, k);
        return kforms_internal::KFormFromIndependentCF(
            ZeroCF(Array<int>{int(basis->IndependentSize())}), k, dim);
    }
    Array<int> dims;
    for (int i = 0; i < k; ++i)
        dims.Append(dim);
    auto zero_cf = ZeroCF(dims);
    return KFormCF(zero_cf, k, dim);
}

shared_ptr<DoubleFormCoefficientFunction> ZeroDoubleForm(int p, int q, int dim)
{
    const int rank = CheckedDoubleFormRank(p, q, "ZeroDoubleForm");
    ValidateFormSpaceDimension(dim, rank == 0, "ZeroDoubleForm");
    ValidateTensorComponentCount(dim, rank, "ZeroDoubleForm");
    if (dim > 0 && p <= dim && q <= dim)
    {
        auto left = kforms_internal::GetFormBasis(dim, p);
        auto right = kforms_internal::GetFormBasis(dim, q);
        const size_t compact_size =
            left->IndependentSize() * right->IndependentSize();
        return kforms_internal::DoubleFormFromIndependentCF(
            ZeroCF(Array<int>{int(compact_size)}), p, q, dim);
    }
    Array<int> dims;
    for (int i = 0; i < rank; ++i)
        dims.Append(dim);
    auto zero_cf = ZeroCF(dims);
    return DoubleFormCF(zero_cf, p, q, dim);
}

shared_ptr<KFormCoefficientFunction> Wedge(shared_ptr<KFormCoefficientFunction> a, shared_ptr<KFormCoefficientFunction> b)
{
    a = RequireNonNull(std::move(a), "Wedge");
    b = RequireNonNull(std::move(b), "Wedge");
    if (a->DimensionOfSpace() != b->DimensionOfSpace())
        throw Exception("Wedge: input forms must have the same dimension of space");
    int dim = a->DimensionOfSpace();
    int k = a->Degree();
    int l = b->Degree();
    if (k + l > dim)
        return ZeroKForm(k + l, dim);
    auto compact_a = kforms_internal::TryGetProvenCompactInput(a);
    auto compact_b = kforms_internal::TryGetProvenCompactInput(b);
    if (compact_a && compact_b)
        return kforms_internal::WedgeCompactKFormsWithViews(
            a, b, std::move(*compact_a), std::move(*compact_b));
    return kforms_internal::WedgeDenseKForms(a, b);
}

shared_ptr<DoubleFormCoefficientFunction> Wedge(shared_ptr<DoubleFormCoefficientFunction> a, shared_ptr<DoubleFormCoefficientFunction> b)
{
    a = RequireNonNull(std::move(a), "Wedge");
    b = RequireNonNull(std::move(b), "Wedge");
    if (a->DimensionOfSpace() != b->DimensionOfSpace())
        throw Exception("Wedge: input double-forms must have the same dimension of space");
    int dim = a->DimensionOfSpace();
    int p = a->LeftDegree();
    int q = a->RightDegree();
    int r = b->LeftDegree();
    int s = b->RightDegree();

    if (p + r > dim || q + s > dim)
        return ZeroDoubleForm(p + r, q + s, dim);
    auto compact_a = kforms_internal::TryGetProvenCompactInput(a);
    auto compact_b = kforms_internal::TryGetProvenCompactInput(b);
    if (compact_a && compact_b)
        return kforms_internal::WedgeCompactDoubleFormsWithViews(
            a, b, std::move(*compact_a), std::move(*compact_b));
    return kforms_internal::WedgeDenseDoubleForms(a, b);
}

shared_ptr<KFormCoefficientFunction> ExteriorDerivative(shared_ptr<KFormCoefficientFunction> a)
{
    a = RequireNonNull(std::move(a), "ExteriorDerivative");
    int dim = a->DimensionOfSpace();
    int k = a->Degree();
    if (k + 1 > dim)
        return ZeroKForm(k + 1, dim);
    auto compact = kforms_internal::TryGetProvenCompactInput(a);
    if (compact)
        return kforms_internal::ExteriorDerivativeCompactKFormWithView(
            a, std::move(*compact));
    return kforms_internal::ExteriorDerivativeDenseKForm(a);
}

shared_ptr<KFormCoefficientFunction> HodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb)
{
    const int n = CheckedHodgeDimension(a, M, vb, "HodgeStar");
    int k = a->Degree();
    if (k > n)
        throw Exception("HodgeStar: form degree exceeds manifold dimension");

    if (vb == BND)
        return BoundaryHodgeStarKForm(a, M);
    if (vb == BBND)
        return BBNDHodgeStarKForm(a, M);

    if (k == 0)
    {
        auto eps = M.GetLeviCivitaSymbol(true);
        return KFormCF(ScaleCoefficientCF(eps, a), n, n);
    }

    if (k == n)
    {
        auto selector = UnitVectorCF(a->Dimension(), TopFormComponentIndex(n))->Reshape(a->Dimensions());
        auto sig = a->GetSignature();
        auto component = SymbolicEinsumCF(sig + "," + sig + "->", {a, selector});
        return KFormCF(ScaleCoefficientCF(component, 1 / M.GetVolumeForm(VOL)), 0, n);
    }

    shared_ptr<TensorFieldCoefficientFunction> raised = a;
    for (int i = 0; i < k; ++i)
        raised = M.Raise(raised, i);

    auto eps = M.GetLeviCivitaSymbol(true);

    std::string alpha_sig = raised->GetSignature(); // length k
    std::string eps_sig = alpha_sig + SIGNATURE.substr(k, n - k);
    std::string out_sig = SIGNATURE.substr(k, n - k);
    auto contracted = SymbolicEinsumCF(alpha_sig + "," + eps_sig + "->" + out_sig,
                                       {raised, eps});
    auto scaled = ScaleCoefficientCF(contracted, ConstantCF(1 / double(Factorial(k))));

    return KFormCF(scaled, n - k, n);
}

shared_ptr<KFormCoefficientFunction> InverseHodgeStar(shared_ptr<KFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb)
{
    const int n = CheckedHodgeDimension(a, M, vb, "InverseHodgeStar");
    int k = a->Degree();
    int exponent = k * (n - k);
    int sign = (exponent % 2 == 0) ? 1 : -1;

    auto star = HodgeStar(a, M, vb);
    if (sign == 1)
        return star;
    return KFormCF(ScaleCoefficientCF(star, ConstantCF(-1)), n - k, vb == VOL ? n : M.Dimension());
}

shared_ptr<DoubleFormCoefficientFunction> HodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb, int slot)
{
    const int n = CheckedHodgeDimension(a, M, vb, "HodgeStar (double-form)");
    ValidateDoubleFormSlot(slot);
    int ambient_dim = M.Dimension();
    int p = a->LeftDegree();
    int q = a->RightDegree();
    if ((slot != 1 && p > n) || (slot != 0 && q > n))
        throw Exception("HodgeStar (double-form): selected form degree exceeds manifold dimension");

    if (vb == VOL)
        if (auto compact =
                kforms_internal::TryHodgeCompactDoubleForm(
                    a, M.GetMetricInverse(), M.GetVolumeForm(VOL),
                    slot))
            return compact;

    if (slot == 0)
    {
        if (vb == BBND)
        {
            if (n == 0)
                return a;
            auto n1 = M.GetEdgeNormal(0);
            auto n2 = M.GetEdgeConormal(0);
            auto star_vol_left = BlockHodgeStar(a, 0, p, ambient_dim, M);
            auto left_tf = TensorFieldCF(star_vol_left, std::string(size_t(ambient_dim - p + q), '1'));
            auto c1 = M.Contraction(left_tf, n1, 0);
            auto c2 = M.Contraction(c1, n2, 0);
            return DoubleFormCF(c2->GetFullCoefficient(), n - p, q, ambient_dim);
        }
        if (vb == BND)
        {
            auto star_vol_left = BlockHodgeStar(a, 0, p, ambient_dim, M);
            auto left_tf = TensorFieldCF(star_vol_left, std::string(size_t(ambient_dim - p + q), '1'));
            auto contracted = M.Contraction(left_tf, M.GetNV(), 0);
            double sign = (p % 2 == 0) ? 1.0 : -1.0;
            return DoubleFormCF(ScaleCoefficientCF(contracted, ConstantCF(sign)), n - p, q, ambient_dim);
        }
        auto left_star = BlockHodgeStar(a, 0, p, n, M);
        return DoubleFormCF(left_star, n - p, q, n);
    }

    if (slot == 1)
    {
        if (vb == BBND)
        {
            if (n == 0)
                return a;
            auto n1 = M.GetEdgeNormal(0);
            auto n2 = M.GetEdgeConormal(0);
            auto star_vol_right = BlockHodgeStar(a, p, q, ambient_dim, M);
            auto right_tf = TensorFieldCF(star_vol_right, std::string(size_t(p + ambient_dim - q), '1'));
            auto c1 = M.Contraction(right_tf, n1, size_t(p));
            auto c2 = M.Contraction(c1, n2, size_t(p));
            return DoubleFormCF(c2->GetFullCoefficient(), p, n - q, ambient_dim);
        }
        if (vb == BND)
        {
            auto star_vol_right = BlockHodgeStar(a, p, q, ambient_dim, M);
            auto right_tf = TensorFieldCF(star_vol_right, std::string(size_t(p + ambient_dim - q), '1'));
            auto contracted = M.Contraction(right_tf, M.GetNV(), size_t(p));
            double sign = (q % 2 == 0) ? 1.0 : -1.0;
            return DoubleFormCF(ScaleCoefficientCF(contracted, ConstantCF(sign)), p, n - q, ambient_dim);
        }
        auto right_star = BlockHodgeStar(a, p, q, n, M);
        return DoubleFormCF(right_star, p, n - q, n);
    }

    if (vb == BBND)
        return BBNDHodgeStarDoubleForm(a, M);

    if (vb == BND)
        return BoundaryHodgeStarDoubleForm(a, M);

    auto left_star = BlockHodgeStar(a, 0, p, n, M);
    auto left_tf = TensorFieldCF(left_star, std::string(size_t(n - p + q), '1'));
    auto right_star = BlockHodgeStar(left_tf, n - p, q, n, M);

    return DoubleFormCF(right_star, n - p, n - q, n);
}

shared_ptr<DoubleFormCoefficientFunction> InverseHodgeStar(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb, int slot)
{
    const int n = CheckedHodgeDimension(a, M, vb, "InverseHodgeStar (double-form)");
    ValidateDoubleFormSlot(slot);
    int p = a->LeftDegree();
    int q = a->RightDegree();
    int exponent = 0;
    if (slot == 0)
        exponent = p * (n - p);
    else if (slot == 1)
        exponent = q * (n - q);
    else
        exponent = p * (n - p) + q * (n - q);
    int sign = (exponent % 2 == 0) ? 1 : -1;

    auto star = HodgeStar(a, M, vb, slot);
    if (sign == 1)
        return star;
    return DoubleFormCF(ScaleCoefficientCF(star, ConstantCF(-1)), star->LeftDegree(), star->RightDegree(), vb == VOL ? n : M.Dimension());
}

shared_ptr<ScalarFieldCoefficientFunction> SlotInnerProduct(shared_ptr<DoubleFormCoefficientFunction> a, const RiemannianManifold &M, VorB vb, bool forms)
{
    return M.SlotInnerProduct(RequireNonNull(a, "SlotInnerProduct"), vb, forms);
}

shared_ptr<DoubleFormCoefficientFunction> SwapDoubleFormSlots(shared_ptr<DoubleFormCoefficientFunction> a)
{
    a = RequireNonNull(std::move(a), "SwapDoubleFormSlots");
    if (kforms_internal::TryGetProvenCompactInput(a))
        return kforms_internal::SwapCompactDoubleFormSlots(a);
    return kforms_internal::SwapDenseDoubleFormSlots(a);
}
