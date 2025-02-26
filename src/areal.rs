use crate::field::Field;
use algebraics::RealAlgebraicNumber;
use num_rational::Ratio;
use num_traits::{Pow, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Algebraic real number.
#[derive(Debug, Clone, PartialEq)]
pub struct AReal {
    value: RealAlgebraicNumber,
}
impl AReal {
    fn from_alg(value: RealAlgebraicNumber) -> Self {
        Self { value }
    }

    /// Raises an `AReal` value to a fractional power.
    ///
    /// # Parameters
    ///
    /// - `frac`: Tuple containing `(numerator, denominator)` of the power to
    ///   which the number should be raised.
    ///   
    /// # Returns
    ///
    /// A new number raised to the given power.
    pub fn pow(&self, frac: (i32, i32)) -> Self {
        Self::from_alg(self.value.clone().pow(frac))
    }

    /// Returns `sqrt(2)` as an `AReal` constant.
    pub fn sqrt2() -> Self {
        AReal::from(2i32).pow((1, 2))
    }
}

impl Add for AReal {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        AReal::from_alg(self.value + rhs.value)
    }
}
impl Sub for AReal {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        AReal::from_alg(self.value - rhs.value)
    }
}
impl Mul for AReal {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        AReal::from_alg(self.value * rhs.value)
    }
}
impl Div for AReal {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        AReal::from_alg(self.value / rhs.value)
    }
}
impl Neg for AReal {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AReal::from_alg(-self.value)
    }
}
impl AddAssign for AReal {
    fn add_assign(&mut self, rhs: Self) {
        self.value.add_assign(rhs.value);
    }
}
impl Field for AReal {
    fn zero() -> Self {
        AReal::from(0i32)
    }
    fn one() -> Self {
        AReal::from(1i32)
    }
    fn inv(&self) -> Option<Self> {
        if self.value.is_zero() {
            None
        } else {
            Some(AReal::from_alg(self.value.recip()))
        }
    }
}

macro_rules! impl_from {
    ($($t:ty),*) => {
        $(
            impl From<$t> for AReal {
                fn from(value: $t) -> Self {
                    AReal::from_alg(RealAlgebraicNumber::from(value))
                }
            }
        )*
    };
}
impl_from!(i8, i16, i32, i64, i128, isize, u8, u32, u64, u128, usize);

macro_rules! impl_from_ratio {
    ($($t:ty),*) => {
        $(
            impl From<Ratio<$t>> for AReal {
                fn from(value: Ratio<$t>) -> Self {
                    AReal::from_alg(RealAlgebraicNumber::from(value))
                }
            }
        )*
    };
}
impl_from_ratio!(i8, i16, i32, i64, i128, isize, u8, u32, u64, u128, usize);

#[cfg(test)]
pub mod tests {
    use crate::field::tests::prop_field;

    use super::*;
    use proptest::prelude::*;

    /// Strategy to generate any i128 value.
    fn strategy_i128() -> BoxedStrategy<i128> {
        any::<i128>().boxed()
    }

    /// Strategy to generate any non-zero i128 value.
    fn strategy_i128_nonzero() -> BoxedStrategy<i128> {
        prop_oneof![i128::MIN..-1, 1..i128::MAX].boxed()
    }

    /// Strategy to generate any valid `Ratio<i128>` value.
    fn strategy_ratio() -> BoxedStrategy<Ratio<i128>> {
        (strategy_i128(), strategy_i128_nonzero())
            .prop_map(|(numerator, denominator)| {
                Ratio::new(numerator, denominator)
            })
            .boxed()
    }

    /// Strategy to generate a rational `AReal`.
    pub fn strategy_rational_areal() -> BoxedStrategy<AReal> {
        strategy_ratio().prop_map(|r| AReal::from(r)).boxed()
    }

    /// Test for equality between `AReal` values.
    pub fn eq_areal(x: &AReal, y: &AReal) -> bool {
        x.eq(y)
    }

    /// Test that `AReal` forms a field.
    #[test]
    fn test_areal_field() {
        prop_field(strategy_rational_areal(), eq_areal);
    }
}
