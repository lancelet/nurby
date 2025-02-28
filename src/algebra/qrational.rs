use crate::algebra::Field;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
use malachite::{
    base::num::basic::traits::Zero,
    base::num::conversion::traits::RoundingFrom,
    base::rounding_modes::RoundingMode, rational::Rational,
};

/// Quadratic rational numbers.
///
/// These numbers are a super-set of rationals, and can also represent
/// rational factors of $\sqrt 2$. This means that they can exactly represent
/// values such as $\sqrt 2$, $1 / \sqrt 2$, $4 + 3\sqrt 2$ and so on.
///
/// The internal representation is:
///
/// $$ r + q \sqrt 2 $$
///
/// where $r$ and $q$ are both real rationals (ie. $r, q \in \mathbb{Q}$).
///
/// # Operations Summary
///
/// Addition:
///
/// $$
/// \left(r_1 + q_1 \sqrt 2\right) +
/// \left(r_2 + q_2 \sqrt 2\right) =
/// \left[(r_1 + r_2) + (q_1 + q_2) \sqrt 2\right]
/// $$
///
/// Subtraction:
///
/// $$
/// \left(r_1 + q_1 \sqrt 2\right) -
/// \left(r_2 + q_2 \sqrt 2\right) =
/// \left[(r_1 - r_2) + (q_1 - q_2) \sqrt 2\right]
/// $$
///
/// Multiplication:
///
/// $$
/// \left(r_1 + q_1 \sqrt 2\right) \times
/// \left(r_2 + q_2 \sqrt 2\right) =
/// \left[
/// \left(r_1 r_2 + 2 q_1 q_2\right)
/// \left(r_1 q_2 + r_2 q_1\right)\sqrt 2
/// \right]
/// $$
///
/// Inverse:
///
/// $$
/// \mathrm{inv}\left(r + q\sqrt 2\right) =
/// \frac{\left(-r + q \sqrt 2\right)}{-r^2 + 2q^2}
/// $$
///
/// Division:
///
/// $$
/// a \div b = a \times \mathrm{inv}(b)
/// $$
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QRational {
    r: Rational,
    q: Rational,
}
impl QRational {
    /// Positive square root of 2.
    pub const SQRT2: Self = QRational {
        r: Rational::const_from_unsigned(0),
        q: Rational::const_from_unsigned(1),
    };
    /// Returns an `f64` value which approximates this `QRational`.
    pub fn round_to_f64(&self) -> f64 {
        let r: f64 =
            f64::rounding_from(self.r.clone(), RoundingMode::Nearest).0;
        let q: f64 =
            f64::rounding_from(self.q.clone(), RoundingMode::Nearest).0;
        r + q * (2.0f64.sqrt())
    }
    fn new(r: Rational, q: Rational) -> Self {
        Self { r, q }
    }
    fn invert(&self) -> Self {
        let r2 = self.r.clone() * self.r.clone();
        let q2 = self.q.clone() * self.q.clone();
        let denom = -r2 + Rational::from(2) * q2;

        let r = -self.r.clone() / denom.clone();
        let q = self.q.clone() / denom;

        Self::new(r, q)
    }
}

impl Add for QRational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        QRational::new(self.r + rhs.r, self.q + rhs.q)
    }
}
impl<'a> Add<&'a Self> for QRational {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + rhs.clone()
    }
}
impl Sub for QRational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        QRational::new(self.r - rhs.r, self.q - rhs.q)
    }
}
impl Mul for QRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let r = self.r.clone() * rhs.r.clone()
            + Rational::from(2) * self.q.clone() * rhs.q.clone();
        let q = self.r * rhs.q + rhs.r * self.q;
        QRational::new(r, q)
    }
}
impl Div for QRational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.invert()
    }
}
impl Neg for QRational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        QRational::new(-self.r, -self.q)
    }
}
impl AddAssign for QRational {
    fn add_assign(&mut self, rhs: Self) {
        self.r += rhs.r;
        self.q += rhs.q;
    }
}
impl SubAssign for QRational {
    fn sub_assign(&mut self, rhs: Self) {
        self.r -= rhs.r;
        self.q -= rhs.q;
    }
}
impl MulAssign for QRational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}
impl DivAssign for QRational {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}
impl Field for QRational {
    const ZERO: Self = QRational {
        r: Rational::const_from_unsigned(0),
        q: Rational::const_from_unsigned(0),
    };
    const ONE: Self = QRational {
        r: Rational::const_from_unsigned(1),
        q: Rational::const_from_unsigned(0),
    };
    fn inv(&self) -> Option<Self> {
        if *self == QRational::ZERO {
            None
        } else {
            Some(self.invert())
        }
    }
}

macro_rules! impl_from_integer {
    ($($t:ty),*) => {
        $(impl From<$t> for QRational {
            fn from(value: $t) -> Self {
                QRational {
                    r: Rational::from(value),
                    q: Rational::ZERO,
                }
            }
        })*
    };
}
impl_from_integer!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::field::tests::{approx_eq_f64_inf, prop_field_axioms};
    use proptest::prelude::*;

    #[test]
    fn example_01() {
        let a = QRational::from(1);
        let b = QRational::from(2);
        let c = (a / b).round_to_f64();
        assert_eq!(0.5f64, c);
    }

    #[test]
    fn example_02() {
        let a = (QRational::from(1) / QRational::SQRT2).round_to_f64();
        let r = 1.0 / (2.0f64.sqrt());
        assert!(approx_eq_f64_inf(a, r, 1e-12, 1e-12));
    }

    #[test]
    fn example_03() {
        let a = (QRational::from(2) + QRational::from(4)).round_to_f64();
        assert_eq!(6.0f64, a);
    }

    /// Strategy to generate a `QRational` from `i32` values.
    fn strategy_qrational_i32s() -> BoxedStrategy<QRational> {
        let s = any::<i32>();
        (s.clone(), s.clone(), s.clone(), s.clone())
            .prop_map(|(rn, rd, qn, qd)| {
                // Denominators cannot be zero.
                let rd = if rd == 0 { 1 } else { rd };
                let qd = if qd == 0 { 1 } else { qd };

                let r = Rational::from(rn) / Rational::from(rd);
                let q = Rational::from(qn) / Rational::from(qd);
                QRational::new(r, q)
            })
            .boxed()
    }

    #[test]
    fn test_qrational_field_axioms() {
        prop_field_axioms(strategy_qrational_i32s(), QRational::eq, 2048);
    }
}
