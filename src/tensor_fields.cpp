
shared_ptr<CoefficientFunction> TraceCF(shared_ptr<CoefficientFunction> coef)
{
    if (coef->IsZeroCF())
        return ZeroCF(coef->Dimensions());

    return make_shared<VectorFieldCoefficientFunction>(coef);
}