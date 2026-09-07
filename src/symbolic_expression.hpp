#ifndef NGSDIFFGEO_SYMBOLIC_EXPRESSION_HPP
#define NGSDIFFGEO_SYMBOLIC_EXPRESSION_HPP

#include <coefficient.hpp>

namespace ngfem
{
  // Native evaluation may simplify zeros and identities. Symbolic operations
  // retain their original operands for differentiation and reconstruction.
  class SymbolicExpressionCoefficientFunction
      : public T_CoefficientFunction<SymbolicExpressionCoefficientFunction>
  {
    using BASE = T_CoefficientFunction<SymbolicExpressionCoefficientFunction>;
  protected:
    Array<shared_ptr<CoefficientFunction>> operands;
    shared_ptr<CoefficientFunction> evaluator;
    // Evaluation graph layout (independent of the semantic operands above):
    // false: expose the evaluator's children and forward their precomputed
    //        values/patterns and code-generation indices to the evaluator.
    // true:  expose the evaluator itself as one child and copy its result.
    //        Proxy derivatives need this so NGSolve can discover a leaf proxy.
    // InputCoefficientFunctions, T_Evaluate, NonZeroPattern and GenerateCode
    // must all use the same layout. Direct evaluation always delegates.
    bool evaluator_as_input;
    virtual shared_ptr<CoefficientFunction> Rebuild(
        const Array<shared_ptr<CoefficientFunction>> &inputs) const = 0;

  private:
    static shared_ptr<CoefficientFunction> TransformOperand(
        const shared_ptr<CoefficientFunction> &operand, T_Transform &transformation);

  public:
    SymbolicExpressionCoefficientFunction(
        const Array<shared_ptr<CoefficientFunction>> &aoperands,
        shared_ptr<CoefficientFunction> aevaluator, bool as_input = false);

    void DoArchive(Archive &) override;
    bool IsZeroCF() const override;
    bool DefinedOn(const ElementTransformation &trafo) override;
    void CalcEquivalenceKey() override;
    Array<shared_ptr<CoefficientFunction>> InputCoefficientFunctions() const override;
    void TraverseTree(const function<void(CoefficientFunction &)> &func) override;
    using BASE::Evaluate;
    double Evaluate(const BaseMappedIntegrationPoint &ip) const override;
    template <typename MIR, typename T, ORDERING ORD>
    void T_Evaluate(const MIR &ir, BareSliceMatrix<T, ORD> values) const
    { evaluator->Evaluate(ir, values); }
    template <typename MIR, typename T, ORDERING ORD>
    void T_Evaluate(const MIR &ir, FlatArray<BareSliceMatrix<T, ORD>> inputs,
                    BareSliceMatrix<T, ORD> values) const
    {
      if (!evaluator_as_input) evaluator->Evaluate(ir, inputs, values);
      else
        for (size_t ip = 0; ip < ir.Size(); ++ip)
          for (int i = 0; i < Dimension(); ++i) values(i, ip) = inputs[0](i, ip);
    }
    void GenerateCode(Code &code, FlatArray<int> inputs, int index) const override;
    void NonZeroPattern(const ProxyUserData &ud,
                        FlatVector<AutoDiffDiff<1, NonZero>> values) const override;
    void NonZeroPattern(const ProxyUserData &ud,
                        FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
                        FlatVector<AutoDiffDiff<1, NonZero>> values) const override;
    shared_ptr<CoefficientFunction> Transform(T_Transform &transformation) const override;
    shared_ptr<CoefficientFunction> DiffJacobi(const CoefficientFunction *var,
                                              T_DJC &cache) const override;
  };

  // Only a childless native zero is safe to discard as a symbolic operand.
  bool IsConstantZero(const shared_ptr<CoefficientFunction> &cf);
}
#endif
