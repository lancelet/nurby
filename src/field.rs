use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// Mathematical field.
///
/// See: https://en.wikipedia.org/wiki/Field_(mathematics)
pub trait Field:
    Sized
    + Clone
    + PartialEq
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn inv(&self) -> Option<Self>;
}

/// Field implementation for f32.
impl Field for f32 {
    const ZERO: Self = 0f32;
    const ONE: Self = 1f32;
    fn inv(&self) -> Option<Self> {
        let reciprocal = 1f32 / self;
        if self.is_finite() && *self != 0f32 && reciprocal.is_finite() {
            Some(reciprocal)
        } else {
            None
        }
    }
}

/// Field implementation for f64.
impl Field for f64 {
    const ZERO: Self = 0f64;
    const ONE: Self = 1f64;
    fn inv(&self) -> Option<Self> {
        let reciprocal = 1f64 / self;
        if self.is_finite() && *self != 0f64 && reciprocal.is_finite() {
            Some(reciprocal)
        } else {
            None
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use core::fmt::Debug;
    use proptest::prelude::*;

    /// Returns a function that checks two `f32` values for approximate
    /// equality using relative tolerance.
    ///
    /// The returned function checks that: `|a-b| <= reltol * max(|a|,|b|)`.
    ///
    /// # Parameters
    ///
    /// - `reltol`: relative tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f32` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f32_rel(reltol: f32) -> impl Fn(&f32, &f32) -> bool {
        move |a, b| {
            let diff = (a - b).abs();
            let largest = a.abs().max(b.abs());
            if largest == 0f32 {
                true
            } else {
                diff <= reltol.abs() * largest
            }
        }
    }

    /// Returns a function that checks two `f32` values for approximate
    /// equality using relative tolerance.
    ///
    /// The returned function checks that: `|a-b| <= abstol`.
    ///
    /// # Parameters
    ///
    /// - `abstol`: absolute tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f32` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f32_abs(abstol: f32) -> impl Fn(&f32, &f32) -> bool {
        move |a, b| (a - b).abs() < abstol.abs()
    }

    /// Returns a function that checks two `f32` "special" values for
    /// approximate equality.
    ///
    /// This handles:
    ///   - Infinite values
    pub fn approx_eq_f32_special() -> impl Fn(&f32, &f32) -> bool {
        move |a, b| {
            a.is_infinite() && b.is_infinite() && (a.signum() == b.signum())
        }
    }

    /// Returns a function that checks two `f32` values for approximate
    /// equality using both absolute and relative tolerance.
    ///
    /// The returned function checks that EITHER:
    /// 1. The two numbers match up to a given absolute tolerance, OR
    /// 2. The two numbers match up to a given relative tolerance.
    ///
    /// # Parameters
    ///
    /// - `reltol`: relative tolerance
    /// - `abstol`: absolute tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f32` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f32_absrel(
        abstol: f32,
        reltol: f32,
    ) -> impl Fn(&f32, &f32) -> bool {
        move |a, b| {
            approx_eq_f32_abs(abstol)(a, b) || approx_eq_f32_rel(reltol)(a, b)
        }
    }

    /// Returns a function that checks two `f32` values for approximate
    /// equality using both absolute and relative tolerance, and allows
    /// certain "special" values to be equal.
    ///
    /// The returned function checks that EITHER:
    /// 1. The two numbers match up to a given absolute tolerance, OR
    /// 2. The two numbers match up to a given relative tolerance.
    ///
    /// # Parameters
    ///
    /// - `reltol`: relative tolerance
    /// - `abstol`: absolute tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f32` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f32_absrelspecial(
        abstol: f32,
        reltol: f32,
    ) -> impl Fn(&f32, &f32) -> bool {
        move |a, b| {
            approx_eq_f32_absrel(abstol, reltol)(a, b)
                || approx_eq_f32_special()(a, b)
        }
    }

    /// Returns a function that checks two `f64` values for approximate
    /// equality using relative tolerance.
    ///
    /// The returned function checks that: `|a-b| <= reltol * max(|a|,|b|)`.
    ///
    /// # Parameters
    ///
    /// - `reltol`: relative tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f64` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f64_rel(reltol: f64) -> impl Fn(&f64, &f64) -> bool {
        move |a, b| {
            let diff = (a - b).abs();
            let largest = a.abs().max(b.abs());
            if largest == 0f64 {
                true
            } else {
                diff <= reltol.abs() * largest
            }
        }
    }

    /// Returns a function that checks two `f64` values for approximate
    /// equality using relative tolerance.
    ///
    /// The returned function checks that: `|a-b| <= abstol`.
    ///
    /// # Parameters
    ///
    /// - `abstol`: absolute tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f64` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f64_abs(abstol: f64) -> impl Fn(&f64, &f64) -> bool {
        move |a, b| (a - b).abs() < abstol.abs()
    }

    /// Returns a function that checks two `f64` values for approximate
    /// equality using both absolute and relative tolerance.
    ///
    /// The returned function checks that EITHER:
    /// 1. The two numbers match up to a given absolute tolerance, OR
    /// 2. The two numbers match up to a given relative tolerance.
    ///
    /// # Parameters
    ///
    /// - `reltol`: relative tolerance
    /// - `abstol`: absolute tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f64` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f64_absrel(
        abstol: f64,
        reltol: f64,
    ) -> impl Fn(&f64, &f64) -> bool {
        move |a, b| {
            approx_eq_f64_abs(abstol)(a, b) || approx_eq_f64_rel(reltol)(a, b)
        }
    }

    /// Returns a function that checks two `f64` "special" values for
    /// approximate equality.
    ///
    /// This handles:
    ///   - Infinite values
    pub fn approx_eq_f64_special() -> impl Fn(&f64, &f64) -> bool {
        move |a, b| {
            a.is_infinite() && b.is_infinite() && (a.signum() == b.signum())
        }
    }

    /// Returns a function that checks two `f64` values for approximate
    /// equality using both absolute and relative tolerance, and allows
    /// certain "special" values to be equal.
    ///
    /// The returned function checks that EITHER:
    /// 1. The two numbers match up to a given absolute tolerance, OR
    /// 2. The two numbers match up to a given relative tolerance.
    ///
    /// # Parameters
    ///
    /// - `reltol`: relative tolerance
    /// - `abstol`: absolute tolerance
    ///
    /// # Returns
    ///
    /// Function that accepts two `f64` values and returns `true` if they are
    /// approximately equal; `false` otherwise.
    pub fn approx_eq_f64_absrelspecial(
        abstol: f64,
        reltol: f64,
    ) -> impl Fn(&f64, &f64) -> bool {
        move |a, b| {
            approx_eq_f64_absrel(abstol, reltol)(a, b)
                || approx_eq_f64_special()(a, b)
        }
    }

    /// Property test checking that addition is associative.
    pub fn prop_addition_associative<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Add<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy.clone(), z in strategy)| {
                let r1 = x.clone() + (y.clone() + z.clone());
                let r2 = (x + y) + z;
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected x + (y + z) ≈ (x + y) + z, but got:\n\
                     x + (y + z) = {:?}\n\
                     (x + y) + z = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test checking that multiplication is associative.
    pub fn prop_multiplication_associative<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Mul<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy.clone(), z in strategy)| {
                let r1 = x.clone() * (y.clone() * z.clone());
                let r2 = (x * y) * z;
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected x * (y * z) ≈ (x * y) * z, but got:\n\
                     x * (y * z) = {:?}\n\
                     (x * y) * z = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test checking that addition is commutative.
    pub fn prop_addition_commutative<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Add<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy)| {
                let r1 = x.clone() + y.clone();
                let r2 = y + x;
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected x + y ≈ y + x, but got:\n\
                     x + y = {:?}\n\
                     y + x = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test checking that multiplication is commutative.
    pub fn prop_multiplication_commutative<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Mul<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy)| {
                let r1 = x.clone() * y.clone();
                let r2 = y * x;
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected x * y ≈ y * x, but got:\n\
                     x * y = {:?}\n\
                     y * x = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test checking that the supplied value is a valid additive
    /// identity.
    pub fn prop_additive_identity<F, S, C>(strategy: S, approx_eq: C, zero: F)
    where
        F: Add<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy)| {
                let r = x.clone() + zero.clone();
                prop_assert!(
                    approx_eq(&x, &r),
                    "Expected x + 0 ≈ x, but got:\n\
                     x + 0 = {:?}\n\
                     x     = {:?}\n",
                     r, x
                );
            }
        )
    }

    /// Property test checking that `ONE` is a valid multiplicative identity.
    pub fn prop_additive_identity_field<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        prop_additive_identity(strategy, approx_eq, F::ZERO);
    }

    /// Property test checking that the supplied value is a valid
    /// multiplicative identity.
    pub fn prop_multiplicative_identity<F, S, C>(
        strategy: S,
        approx_eq: C,
        one: F,
    ) where
        F: Mul<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy)| {
                let r = x.clone() * one.clone();
                prop_assert!(
                    approx_eq(&x, &r),
                    "Expected x * 1 ≈ x, but got:\n\
                     x * 1 = {:?}\n\
                     x     = {:?}\n",
                     r, x
                );
            }
        )
    }

    /// Property test checking that `ONE` is a valid multiplicative identity.
    pub fn prop_multiplicative_identity_field<F, S, C>(
        strategy: S,
        approx_eq: C,
    ) where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        prop_multiplicative_identity(strategy, approx_eq, F::ONE);
    }

    /// Property test that `neg` calculates an additive inverse.
    pub fn prop_additive_inverse<F, S, C>(strategy: S, approx_eq: C, zero: F)
    where
        F: Add<Output = F> + Neg<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy)| {
                let r = x.clone() + x.clone().neg();
                prop_assert!(
                    approx_eq(&zero, &r),
                    "Expected x + (-x) ≈ 0, but got:\n\
                     x + (-x) = {:?}\n",
                     r
                );
            }
        )
    }

    /// Property test that `neg` calculates an additive inverse.
    pub fn prop_additive_inverse_field<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        prop_additive_inverse(strategy, approx_eq, F::ZERO);
    }

    /// Property test that `inv` (where defined) calculates a multiplicative
    /// inverse.
    pub fn prop_multiplicative_inverse<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy)| {
                if let Some(recip) = x.clone().inv() {
                    let r = x.clone() * recip;
                    prop_assert!(
                        approx_eq(&F::ONE, &r),
                        "Expected x * inv(x) ≈ 1, but got:\n\
                        x * inv(x) = {:?}\n",
                        r
                    );
                }
            }
        )
    }

    /// Property test for distributivity of multiplication over addition.
    pub fn prop_distributivity<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Add<Output = F> + Mul<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy.clone(), z in strategy)| {
                let r1 = x.clone() * (y.clone() + z.clone());
                let r2 = (x.clone() * y) + (x.clone() * z);
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected x * (y + z) ≈ (x * y) + (x * z), but got:\n\
                     x * (y + z)       = {:?}\n\
                     (x * y) + (x * z) = {:?}\n",
                     r1, r2
                );
            }
        )
    }

    /// Property test that addition and subtraction are inverses.
    pub fn prop_add_sub_inverse<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Add<Output = F> + Sub<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy)| {
                let r1 = (x.clone() + y.clone()) - y.clone();
                let r2 = (x.clone() + y.clone()) - x.clone();
                prop_assert!(
                    approx_eq(&r1, &x),
                    "Expected (x + y) - y ≈ x, but got:\n\
                     (x + y) - y = {:?}\n\
                     x           = {:?}\n",
                     r1, x
                );
                prop_assert!(
                    approx_eq(&r2, &y),
                    "Expected (x + y) - x ≈ y, but got:\n\
                     (x + y) - x = {:?}\n\
                     y           = {:?}\n",
                     r2, y
                );
            }
        )
    }

    /// Property test that multiplication and division are inverses.
    pub fn prop_mul_div_inverse<F, S, C>(strategy: S, approx_eq: C)
    where
        F: Mul<Output = F> + Div<Output = F> + Debug + Clone,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool,
    {
        proptest!(
            |(x in strategy.clone(), y in strategy)| {
                let r1 = (x.clone() * y.clone()) / y.clone();
                let r2 = (x.clone() * y.clone()) / x.clone();
                prop_assert!(
                    approx_eq(&r1, &x),
                    "Expected (x * y) / y ≈ x, but got:\n\
                     (x * y) / y = {:?}\n\
                     x           = {:?}\n",
                     r1, x
                );
                prop_assert!(
                    approx_eq(&r2, &y),
                    "Expected (x * y) / x ≈ y, but got:\n\
                     (x * y) / x = {:?}\n\
                     y           = {:?}\n",
                     r2, y
                );
            }
        )
    }

    /// Default approximate equality for `f32``.
    ///
    /// This is suitable for many situations, but certainly not all.
    pub fn approx_eq_f32(a: &f32, b: &f32) -> bool {
        approx_eq_f32_absrelspecial(1e-6, 1e-5)(a, b)
    }

    /// Default approximate equality for `f64``.
    ///
    /// This is suitable for many situations, but certainly not all.
    pub fn approx_eq_f64(a: &f64, b: &f64) -> bool {
        approx_eq_f64_absrelspecial(1e-12, 1e-12)(a, b)
    }

    /// Strategy to generate a finite `f32` value.
    fn strategy_f32_finite() -> BoxedStrategy<f32> {
        any::<f32>()
            .prop_filter("Must be finite", |f| f.is_finite())
            .boxed()
    }

    /// Strategy to generates "nice" `f32` values.
    fn strategy_f32_nice() -> BoxedStrategy<f32> {
        let min: f32 = 1e-5;
        let max: f32 = 1e5;
        prop_oneof![
            50 => -max..-min,
            50 => min..max,
            1 => Just(0f32)
        ]
        .boxed()
    }

    /// Strategy to generates "nice" `f32` values that exclude zero.
    fn strategy_f32_nice_nonzero() -> BoxedStrategy<f32> {
        let min: f32 = 1e-5;
        let max: f32 = 1e5;
        prop_oneof![-max..-min, min..max,].boxed()
    }

    /// Strategy to generate a finite `f64` value.
    fn strategy_f64_finite() -> BoxedStrategy<f64> {
        any::<f64>()
            .prop_filter("Must be finite", |f| f.is_finite())
            .boxed()
    }

    /// Strategy to generates "nice" `f64` values.
    fn strategy_f64_nice() -> BoxedStrategy<f64> {
        let min: f64 = 1e-5;
        let max: f64 = 1e5;
        prop_oneof![
            50 => -max..-min,
            50 => min..max,
            1 => Just(0f64)
        ]
        .boxed()
    }

    /// Strategy to generates "nice" `f64` values that exclude zero.
    fn strategy_f64_nice_nonzero() -> BoxedStrategy<f64> {
        let min: f64 = 1e-5;
        let max: f64 = 1e5;
        prop_oneof![-max..-min, min..max,].boxed()
    }

    /// Test that `f32` satisfies the requirements to be a `Field`.
    ///
    /// Due to floating-point issues, some of the properties require relaxed
    /// ranges of floating-point values and/or relaxed comparisons.
    #[test]
    fn test_f32_field() {
        let strategy = strategy_f32_finite();
        let approx_eq = approx_eq_f32;

        prop_addition_associative(strategy.clone(), approx_eq.clone());
        prop_multiplication_associative(strategy.clone(), approx_eq.clone());
        prop_addition_commutative(strategy.clone(), approx_eq.clone());
        prop_multiplication_commutative(strategy.clone(), approx_eq.clone());
        prop_additive_identity_field(strategy.clone(), approx_eq.clone());
        prop_multiplicative_identity_field(strategy.clone(), approx_eq.clone());
        prop_additive_inverse_field(strategy.clone(), approx_eq.clone());
        prop_multiplicative_inverse(strategy.clone(), approx_eq.clone());
        prop_distributivity(
            strategy_f32_nice(),
            approx_eq_f32_absrelspecial(1e-4, 1e-3),
        );
        prop_add_sub_inverse(
            strategy_f32_nice(),
            approx_eq_f32_absrelspecial(1e-4, 1e-3),
        );
        prop_mul_div_inverse(
            strategy_f32_nice_nonzero(),
            approx_eq_f32_absrelspecial(1e-4, 1e-3),
        );
    }

    /// Test that `f64` satisfies the requirements to be a `Field`.
    ///
    /// Due to floating-point issues, some of the properties require relaxed
    /// ranges of floating-point values and/or relaxed comparisons.
    #[test]
    fn test_f64_field() {
        let strategy = strategy_f64_finite();
        let approx_eq = approx_eq_f64;

        prop_addition_associative(strategy.clone(), approx_eq.clone());
        prop_multiplication_associative(strategy.clone(), approx_eq.clone());
        prop_addition_commutative(strategy.clone(), approx_eq.clone());
        prop_multiplication_commutative(strategy.clone(), approx_eq.clone());
        prop_additive_identity_field(strategy.clone(), approx_eq.clone());
        prop_multiplicative_identity_field(strategy.clone(), approx_eq.clone());
        prop_additive_inverse_field(strategy.clone(), approx_eq.clone());
        prop_multiplicative_inverse(strategy.clone(), approx_eq.clone());
        prop_distributivity(strategy_f64_nice(), approx_eq.clone());
        prop_add_sub_inverse(strategy_f64_nice(), approx_eq.clone());
        prop_mul_div_inverse(strategy_f64_nice_nonzero(), approx_eq.clone());
    }
}
