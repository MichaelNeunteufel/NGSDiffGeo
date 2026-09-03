#include "tensor_fields.hpp"

#include <array>
#include <functional>
#include <limits>
#include <stdexcept>

namespace
{
  void Require(bool condition)
  {
    if (!condition)
      throw std::runtime_error("tensor metadata unit-test assertion failed");
  }

  void ExpectException(const std::function<void()> &func)
  {
    bool threw = false;
    try
    {
      func();
    }
    catch (const ngstd::Exception &)
    {
      threw = true;
    }
    Require(threw);
  }
}

int main()
{
  using ngfem::Factorial;
  using ngfem::MAX_SIGNATURE_LABELS;
  using ngfem::TensorMeta;

  constexpr std::array<int, 13> factorials = {
      1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
      3628800, 39916800, 479001600};
  for (size_t i = 0; i < factorials.size(); ++i)
    Require(Factorial(int(i)) == factorials[i]);
  ExpectException([]() { Factorial(-1); });
  ExpectException([]() { Factorial(13); });

  const TensorMeta scalar;
  Require(scalar.Rank() == 0);
  Require(scalar.CovarianceMask() == 0);
  Require(scalar.CovString().empty());
  Require(scalar.Sig().empty());

  // Exhaustively check the raw-mask/string round trip for small ranks.
  for (size_t rank = 0; rank <= 10; ++rank)
    for (uint64_t mask = 0; mask < (uint64_t(1) << rank); ++mask)
    {
      const auto raw = TensorMeta::FromRaw(rank, mask);
      const auto reconstructed = TensorMeta::FromCovString(raw.CovString());
      Require(reconstructed == raw);
      Require(reconstructed.Rank() == rank);
      Require(reconstructed.CovarianceMask() == mask);
    }

  const auto meta = TensorMeta::FromCovString("101");
  Require(meta.Rank() == 3);
  Require(meta.CovarianceMask() == 5);
  Require(meta.CovString() == "101");
  Require(meta.Sig() == "abc");
  Require(meta.Label(0) == 'a');
  Require(meta.Label(1) == 'b');
  Require(meta.Label(2) == 'c');
  Require(meta.FreshLabel() == 'd');
  Require(meta.FreshLabel(1) == 'e');
  Require(meta.WithCovariant(1, true).CovString() == "111");
  Require(meta.Appended(false).CovString() == "1010");
  Require(meta.Prepended(true).CovString() == "1101");
  Require(meta.Erased(1).CovString() == "11");

  for (size_t i = 0; i < meta.Rank(); ++i)
  {
    const auto covariant = meta.WithCovariant(i, true);
    Require(covariant.Covariant(i));
    Require(!covariant.WithCovariant(i, false).Covariant(i));
    Require(meta.CovString() == "101");
  }

  for (bool covariant : {false, true})
  {
    Require(meta.Appended(covariant).Erased(meta.Rank()) == meta);
    Require(meta.Prepended(covariant).Erased(0) == meta);
  }

  const auto erase_meta = TensorMeta::FromCovString("1010");
  Require(erase_meta.Erased(0).CovString() == "010");
  Require(erase_meta.Erased(1).CovString() == "110");
  Require(erase_meta.Erased(2).CovString() == "100");
  Require(erase_meta.Erased(3).CovString() == "101");

  const auto erase2_meta = TensorMeta::FromCovString("10110");
  Require(erase2_meta.Erased2(1, 3).CovString() == "110");
  Require(erase2_meta.Erased2(3, 1).CovString() == "110");
  ExpectException([&]() { erase2_meta.Erased2(2, 2); });
  ExpectException([&]() { erase2_meta.Erased2(1, erase2_meta.Rank()); });

  const auto left = TensorMeta::FromCovString("01");
  const auto right = TensorMeta::FromCovString("110");
  Require(left.Concatenated(right).CovString() == "01110");
  Require(scalar.Concatenated(left) == left);
  Require(left.Concatenated(scalar) == left);

  const auto labels = TensorMeta::FromCovString(std::string(27, '0'));
  Require(labels.Label(25) == 'z');
  Require(labels.Label(26) == 'A');
  const auto maximum =
      TensorMeta::FromCovString(std::string(MAX_SIGNATURE_LABELS, '0'));
  Require(maximum.Label(MAX_SIGNATURE_LABELS - 1) == 'Z');
  Require(maximum.Sig() == ngfem::SIGNATURE);

  ExpectException([]() { TensorMeta::FromCovString("10x1"); });
  ExpectException([]() { TensorMeta::FromRaw(0, 1); });
  ExpectException([]() { TensorMeta::FromRaw(1, 2); });
  ExpectException([&]() { meta.Covariant(meta.Rank()); });
  ExpectException([&]() { meta.Label(meta.Rank()); });
  ExpectException([&]() { meta.WithCovariant(meta.Rank(), true); });
  ExpectException([&]() {
    meta.FreshLabel(std::numeric_limits<size_t>::max());
  });
  ExpectException([&]() { maximum.FreshLabel(); });
  ExpectException([]() {
    TensorMeta::FromRaw(MAX_SIGNATURE_LABELS + 1, 0);
  });
  ExpectException([]() {
    TensorMeta::FromCovString(std::string(MAX_SIGNATURE_LABELS, '0'))
        .Appended(false);
  });
  ExpectException([]() {
    TensorMeta::FromCovString(std::string(MAX_SIGNATURE_LABELS, '0'))
        .Concatenated(TensorMeta::FromCovString("0"));
  });
}
