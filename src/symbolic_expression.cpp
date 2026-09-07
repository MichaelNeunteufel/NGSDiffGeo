#include "symbolic_expression.hpp"
#include "tensor_fields.hpp"

#include <core/register_archive.hpp>
#include <map>

namespace ngfem
{
  shared_ptr<CoefficientFunction> SymbolicExpressionCoefficientFunction::TransformOperand(
      const shared_ptr<CoefficientFunction> &operand, T_Transform &transformation)
  {
    if (auto it = transformation.cache.find(operand); it != transformation.cache.end())
      return it->second;
    if (auto it = transformation.replace.find(operand); it != transformation.replace.end())
      return transformation.cache[operand] = it->second;

    // Inspect semantic dependencies before asking native nodes to transform.
    // Cache unchanged nodes too: shared subgraphs must only be inspected once.
    auto symbolic = dynamic_pointer_cast<SymbolicExpressionCoefficientFunction>(operand);
    auto children = symbolic ? Array<shared_ptr<CoefficientFunction>>(symbolic->operands)
                             : operand->InputCoefficientFunctions();
    bool changed = false;
    for (auto child : children)
      if (child)
        changed |= TransformOperand(child, transformation) != child;

    // Untouched native auxiliary nodes (e.g. inverse metrics) need no Transform.
    auto result = changed ? operand->Transform(transformation) : operand;
    transformation.cache[operand] = result;
    return result;
  }

  SymbolicExpressionCoefficientFunction::SymbolicExpressionCoefficientFunction(
      const Array<shared_ptr<CoefficientFunction>> &aoperands,
      shared_ptr<CoefficientFunction> aevaluator, bool as_input)
      : BASE(1), operands(aoperands), evaluator(std::move(aevaluator)),
        evaluator_as_input(as_input)
  {
    for (auto operand : operands)
      if (!operand) throw Exception("SymbolicExpressionCF: null operand");
    SetDimensions(evaluator->Dimensions());
    is_complex = evaluator->IsComplex();
    elementwise_constant = true;
    for (auto operand : operands)
      elementwise_constant &= operand->ElementwiseConstant();
  }

  void SymbolicExpressionCoefficientFunction::DoArchive(Archive &)
  {}

  bool SymbolicExpressionCoefficientFunction::IsZeroCF() const
  { return evaluator->IsZeroCF(); }

  bool SymbolicExpressionCoefficientFunction::DefinedOn(const ElementTransformation &trafo)
  {
    for (auto operand : operands)
      if (!operand->DefinedOn(trafo)) return false;
    return true;
  }

  void SymbolicExpressionCoefficientFunction::CalcEquivalenceKey()
  {
    equivalence_key = GetDescription() + "(";
    for (auto operand : operands)
      equivalence_key += operand->EquivalenceKey() + ",";
    equivalence_key += ")";
  }

  Array<shared_ptr<CoefficientFunction>> SymbolicExpressionCoefficientFunction::InputCoefficientFunctions() const
  { return evaluator_as_input ? Array<shared_ptr<CoefficientFunction>>{evaluator}
                              : evaluator->InputCoefficientFunctions(); }

  void SymbolicExpressionCoefficientFunction::TraverseTree(const function<void(CoefficientFunction &)> &func)
  {
    for (auto input : InputCoefficientFunctions()) input->TraverseTree(func);
    func(*this);
  }

  double SymbolicExpressionCoefficientFunction::Evaluate(const BaseMappedIntegrationPoint &ip) const
  { return evaluator->Evaluate(ip); }

  void SymbolicExpressionCoefficientFunction::GenerateCode(Code &code, FlatArray<int> inputs, int index) const
  {
    if (!evaluator_as_input) { evaluator->GenerateCode(code, inputs, index); return; }
    DeclareTensorFieldGeneratedCoefficient(code, index, Dimensions(), IsComplex());
    for (int i = 0; i < Dimension(); ++i)
      code.body += Var(index, i, Dimensions()).Assign(Var(inputs[0], i, Dimensions()), false);
  }

  void SymbolicExpressionCoefficientFunction::NonZeroPattern(const ProxyUserData &ud,
                      FlatVector<AutoDiffDiff<1, NonZero>> values) const
  {
    using Pattern = AutoDiffDiff<1, NonZero>;
    // Native einsum queries operand patterns while rebuilding its evaluator.
    // Use precomputed child patterns to avoid expanding shared DAGs into trees.
    // Keep this cache local: patterns depend on the current proxy/component.
    std::map<CoefficientFunction *, Vector<Pattern>> patterns;
    evaluator->TraverseDAG([&](CoefficientFunction &node) {
      auto &output = patterns.try_emplace(&node, node.Dimension()).first->second;
      auto children = node.InputCoefficientFunctions();
      if (children.Size() == 0)
      {
        node.NonZeroPattern(ud, output);
        return;
      }
      std::vector<FlatVector<Pattern>> inputs;
      inputs.reserve(children.Size());
      for (auto child : children)
        inputs.emplace_back(child ? FlatVector<Pattern>(patterns.at(child.get()))
                                  : FlatVector<Pattern>(0, nullptr));
      node.NonZeroPattern(ud,
          FlatArray<FlatVector<Pattern>>(inputs.size(), inputs.data()), output);
    });
    values = patterns.at(evaluator.get());
  }

  void SymbolicExpressionCoefficientFunction::NonZeroPattern(const ProxyUserData &ud,
                      FlatArray<FlatVector<AutoDiffDiff<1, NonZero>>> inputs,
                      FlatVector<AutoDiffDiff<1, NonZero>> values) const
  {
    if (evaluator_as_input) values = inputs[0];
    else evaluator->NonZeroPattern(ud, inputs, values);
  }

  shared_ptr<CoefficientFunction> SymbolicExpressionCoefficientFunction::Transform(T_Transform &transformation) const
  {
    auto self = const_pointer_cast<CoefficientFunction>(shared_from_this());
    if (transformation.cache.count(self)) return transformation.cache[self];
    if (transformation.replace.count(self)) return transformation.replace[self];
    Array<shared_ptr<CoefficientFunction>> transformed;
    bool changed = false;
    for (auto operand : operands)
    {
      auto result = TransformOperand(operand, transformation);
      changed |= result != operand;
      transformed.Append(result);
    }
    auto result = changed ? Rebuild(transformed) : self;
    transformation.cache[self] = result;
    return result;
  }

  bool IsConstantZero(const shared_ptr<CoefficientFunction> &cf)
  {
    return cf->IsZeroCF() && cf->InputCoefficientFunctions().Size() == 0
        && !dynamic_pointer_cast<SymbolicExpressionCoefficientFunction>(cf);
  }

  // Retain Jacobian columns even when their current evaluations are all zero.
  class SymbolicJacobianCoefficientFunction : public SymbolicExpressionCoefficientFunction
  {
    Array<int> dimensions;
    static shared_ptr<CoefficientFunction> MakeEvaluator(
        const Array<shared_ptr<CoefficientFunction>> &columns, const Array<int> &dims)
    {
      return MakeVectorialCoefficientFunction(Array<shared_ptr<CoefficientFunction>>(columns))
          ->Reshape(columns.Size(), columns[0]->Dimension())->Transpose()->Reshape(dims);
    }
    shared_ptr<CoefficientFunction> Rebuild(
        const Array<shared_ptr<CoefficientFunction>> &inputs) const override
    { return make_shared<SymbolicJacobianCoefficientFunction>(inputs, dimensions); }
  public:
    SymbolicJacobianCoefficientFunction(
        const Array<shared_ptr<CoefficientFunction>> &columns, const Array<int> &dims)
        : SymbolicExpressionCoefficientFunction(columns, MakeEvaluator(columns, dims)), dimensions(dims) {}
    auto GetCArgs() const { return tuple{Array<shared_ptr<CoefficientFunction>>(operands), Array<int>(dimensions)}; }
    string GetDescription() const override { return "SymbolicJacobianCF " + ToString(dimensions); }
    shared_ptr<CoefficientFunction> Diff(const CoefficientFunction *var,
                                         shared_ptr<CoefficientFunction> dir) const override
    {
      if (this == var) return dir;
      Array<shared_ptr<CoefficientFunction>> columns;
      for (auto operand : operands) columns.Append(operand->Diff(var, dir));
      return Rebuild(columns);
    }
  };

  shared_ptr<CoefficientFunction> SymbolicExpressionCoefficientFunction::DiffJacobi(
      const CoefficientFunction *var, T_DJC &cache) const
  {
    auto self = const_pointer_cast<CoefficientFunction>(shared_from_this());
    if (auto it = cache.find(self); it != cache.end()) return it->second;
    if (this == var) return IdentityCF(Dimensions());
    Array<shared_ptr<CoefficientFunction>> columns(var->Dimension());
    for (size_t i : Range(columns))
      columns[i] = Diff(var, UnitVectorCF(var->Dimension(), int(i))->Reshape(var->Dimensions()));
    Array<int> dims(Dimensions());
    dims.Append(var->Dimensions());
    return cache[self] = make_shared<SymbolicJacobianCoefficientFunction>(columns, dims);
  }

  static ngcore::RegisterClassForArchive<SymbolicJacobianCoefficientFunction,
                                         CoefficientFunction> reg_symbolic_jacobian;

}
