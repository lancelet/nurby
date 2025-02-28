use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Mathematical field.
///
/// See: <https://en.wikipedia.org/wiki/Field_(mathematics)>
///
/// A `Field` represents the scalar values that NURBS curves and surfaces
/// are computed on. Examples of a field include:
/// - `f32` (approximate)
/// - `f64` (approximate)
/// - [`QRational`][crate::algebra::QRational] (exact if it fits in memory)
///
/// Not all NURBS operations are possible on `Field`s, but they form the
/// initial representation.
pub trait Field:
    Sized
    + Debug
    + Clone
    + Display
    + PartialEq
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Additive identity of the field.
    const ZERO: Self;

    /// Multiplicative identity of the field.
    const ONE: Self;

    /// Returns the multiplicative inverse of a value of the field.
    fn inv(&self) -> Option<Self>;
}

/// Field implementation for f32.
impl Field for f32 {
    const ZERO: Self = 0f32;
    const ONE: Self = 1f32;
    fn inv(&self) -> Option<Self> {
        let reciprocal = 1f32 / self;
        if reciprocal.is_finite() {
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
    use proptest::{prelude::*, test_runner::Config, test_runner::TestRunner};

    /// Checks for approximate `f32` equality, with equality of infinities.
    ///
    /// This function checks if values `a` and `b` satisfy one of:
    ///
    /// 1. **Relative equality**: The difference between `a` and `b` is within
    ///    a relative tolerance:
    ///    ```
    ///    |a - b| <= reltol * max(|a|, |b|)
    ///    ```
    ///    This is useful for comparing large floating-point values.
    ///
    /// 2. **Absolute equality**: The absolute difference between `a` and `b`
    ///    is within a given absolute tolerance:
    ///    ```
    ///    |a - b| <= abstol
    ///    ```
    ///    This is useful when `a` and `b` are close to zero.
    ///
    /// 3. **Equality of infinities**: If both `a` and `b` are infinite with the
    ///    same sign (`+∞` or `-∞`), they are considered approximately equal.
    ///
    /// **Note:** NaN (Not-a-Number) values are **never** considered equal,
    /// even to themselves.
    ///
    /// If any of the three types of equality are satisfied then the function
    /// returns `true`.
    ///
    /// # Parameters
    ///
    /// - `a`: One value to test.
    /// - `b`: The other value to test.
    /// - `abstol`: Absolute tolerance value.
    /// - `reltol`: Relative tolerance value.
    ///
    /// # Returns
    ///
    /// `true` if `a ≈ b`, `false` otherwise.
    pub fn approx_eq_f32_inf(a: f32, b: f32, abstol: f32, reltol: f32) -> bool {
        let diff = (a - b).abs();
        let largest = f32::max(a.abs(), b.abs());
        (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || diff <= reltol.abs() * largest
            || diff < abstol.abs()
    }

    /// Checks for approximate `f64` equality, with equality of infinities.
    ///
    /// This function checks if values `a` and `b` satisfy one of:
    ///
    /// 1. **Relative equality**: The difference between `a` and `b` is within
    ///    a relative tolerance:
    ///    ```
    ///    |a - b| <= reltol * max(|a|, |b|)
    ///    ```
    ///    This is useful for comparing large floating-point values.
    ///
    /// 2. **Absolute equality**: The absolute difference between `a` and `b`
    ///    is within a given absolute tolerance:
    ///    ```
    ///    |a - b| <= abstol
    ///    ```
    ///    This is useful when `a` and `b` are close to zero.
    ///
    /// 3. **Equality of infinities**: If both `a` and `b` are infinite with the
    ///    same sign (`+∞` or `-∞`), they are considered approximately equal.
    ///
    /// **Note:** NaN (Not-a-Number) values are **never** considered equal,
    /// even to themselves.
    ///
    /// If any of the three types of equality are satisfied then the function
    /// returns `true`.
    ///
    /// # Parameters
    ///
    /// - `a`: One value to test.
    /// - `b`: The other value to test.
    /// - `abstol`: Absolute tolerance value.
    /// - `reltol`: Relative tolerance value.
    ///
    /// # Returns
    ///
    /// `true` if `a ≈ b`, `false` otherwise.
    pub fn approx_eq_f64_inf(a: f64, b: f64, abstol: f64, reltol: f64) -> bool {
        let diff = (a - b).abs();
        let largest = f64::max(a.abs(), b.abs());
        (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || diff <= reltol.abs() * largest
            || diff < abstol.abs()
    }

    /// A utility macro to clone multiple variables before evaluating an
    /// expression.
    ///
    /// This macro is useful when working with closures that capture variables
    /// by reference, but you need owned copies instead.
    ///
    /// # Example
    ///
    /// ```
    /// let s = String::from("hello");
    /// let n = 42;
    ///
    /// let closure = cloned!(s n => move || {
    ///     println!("Cloned values: {}, {}", s, n);
    /// });
    ///
    /// closure();
    /// ```
    ///
    /// This expands to:
    ///
    /// ```rust
    /// let s = s.clone();
    /// let n = n.clone();
    /// let closure = move || {
    ///     println!("Cloned values: {}, {}", s, n);
    /// };
    /// ```
    ///
    /// # Details
    ///
    /// - The variables listed before `=>` are cloned.
    /// - The final expression after `=>` is evaluated with the cloned
    ///   variables.
    /// - The macro ensures that cloning happens before evaluating the
    ///   expression.
    ///
    /// # Note
    ///
    /// - This macro assumes that all provided variables implement [`Clone`].
    /// - If a variable is not used in the final expression, it will be marked
    ///   as unused but not cause a warning.
    ///
    /// # Expansion Example
    ///
    /// ```rust
    /// // Given:
    /// cloned!(a b => some_expression);
    ///
    /// // Expands to:
    /// {
    ///     let a = a.clone();
    ///     let b = b.clone();
    ///     some_expression
    /// }
    /// ```
    #[macro_export]
    macro_rules! cloned {
        ($($var:ident)+ => $expr:expr) => {{
            #[allow(unused_variables)]
            {
                $(let $var = $var.clone();)+
                $expr
            }
        }};
    }

    /// Asserts that two values are approximately equal using a custom
    /// comparison function.
    ///
    /// This macro provides a way to check for approximate equality by
    /// delegating the comparison to a user-specified function. It is useful
    /// when working with floating-point numbers or any data type where exact
    /// equality is too strict.
    ///
    /// # Arguments
    ///
    /// - `$lhs_label`: A string representing the name of the left-hand side
    ///   value (for error messages).
    /// - `$rhs_label`: A string representing the name of the right-hand side
    ///   value (for error messages).
    /// - `$lhs_value`: The left-hand side value to compare.
    /// - `$rhs_value`: The right-hand side value to compare.
    /// - `$approx_eq`: A function or closure that takes references to both
    ///   values and returns `true` if they are approximately equal.
    ///
    /// # Example
    ///
    /// ```
    /// fn approx_eq(a: &f64, b: &f64) -> bool {
    ///     (a - b).abs() < 1e-6
    /// }
    ///
    /// let x = 3.141592;
    /// let y = 3.141593;
    ///
    /// assert_approx_eq!("x", "y", x, y, approx_eq);
    /// ```
    ///
    /// This expands to:
    ///
    /// ```rust
    /// assert!(
    ///     approx_eq(&x, &y),
    ///     "Expected x ≈ y, but got:
    ///     x = 3.141592
    ///     y = 3.141593"
    /// );
    /// ```
    ///
    /// # Notes
    ///
    /// - The `$approx_eq` function is responsible for defining what
    ///   "approximately equal" means.
    /// - If the assertion fails, the macro prints a detailed message showing
    ///   the values and labels.
    ///
    /// # Use Cases
    ///
    /// - Comparing floating-point numbers.
    /// - Checking approximate equality of complex data structures.
    /// - Custom comparisons with domain-specific tolerances.
    ///
    /// # Alternative
    ///
    /// If you need a more general assertion, consider using `assert!` directly
    /// with a custom error message.
    #[macro_export]
    macro_rules! assert_approx_eq {
        ($lhs_label:expr,
         $rhs_label:expr,
         $lhs_value:expr,
         $rhs_value:expr,
         $approx_eq:expr) => {
            assert!(
                $approx_eq(&$lhs_value, &$rhs_value),
                "Expected {} ≈ {}, but got:\n\
                {} = {:?}\n\
                {} = {:?}\n",
                $lhs_label,
                $rhs_label,
                $lhs_label,
                $lhs_value,
                $rhs_label,
                $rhs_value
            );
        };
    }

    /// Evaluates expressions in the same variable context and asserts they are
    /// approximately equal.
    ///
    /// This macro ensures that two expressions evaluate in the same
    /// environment, clones the necessary variables to avoid borrow conflicts,
    /// and checks if their results are approximately equal using a custom
    /// comparison function.
    ///
    /// # Arguments
    ///
    /// - `$($var:ident)+` - A list of variables that should be cloned for both
    ///   expressions.
    /// - `$lhs` - The left-hand side expression to evaluate.
    /// - `$rhs` - The right-hand side expression to evaluate.
    /// - `$approx_eq` - A function or closure that takes references to both
    ///   results and returns `true` if they are approximately equal.
    ///
    /// # Behavior
    ///
    /// - The variables in `$($var)+` are cloned before evaluating `$lhs` and
    ///   `$rhs`.
    /// - Both expressions are evaluated separately, ensuring no borrowing
    ///   conflicts.
    /// - The results are then compared using the provided `$approx_eq`
    ///   function.
    /// - If the comparison fails, an assertion error is raised, displaying the
    ///   evaluated values.
    ///
    /// # Example
    ///
    /// ```
    /// fn approx_eq(a: &f64, b: &f64) -> bool {
    ///     (a - b).abs() < 1e-6
    /// }
    ///
    /// let x = 3.141592;
    /// let y = 3.141593;
    ///
    /// check_approx_eq!(x y => x * y, y * x, approx_eq);
    /// ```
    ///
    /// This expands to:
    ///
    /// ```rust
    /// let lhs_result = {
    ///     let x = x.clone();
    ///     let y = y.clone();
    ///     x * y
    /// };
    /// let rhs_result = {
    ///     let x = x.clone();
    ///     let y = y.clone();
    ///     y * x
    /// };
    ///
    /// assert!(
    ///     approx_eq(&lhs_result, &rhs_result),
    ///     "Expected x * y ≈ y * x, but got:
    ///     x * y = {:?}
    ///     y * x = {:?}",
    ///     lhs_result,
    ///     rhs_result
    /// );
    /// ```
    ///
    /// # Notes
    ///
    /// - The macro uses [`cloned!`] to avoid variable borrow conflicts.
    /// - It relies on [`assert_approx_eq!`] to perform the final comparison.
    /// - This is useful when comparing expressions involving floating-point
    ///   numbers or any domain where exact equality is too strict.
    ///
    /// # Alternative
    ///
    /// If you only need to compare values (not expressions), use
    /// [`assert_approx_eq!`].
    #[macro_export]
    macro_rules! check_approx_eq {
        ($($var:ident)+ => $lhs:expr, $rhs:expr, $approx_eq:expr) => {{
            let lhs_result = cloned!($($var)+ => { $lhs });
            let rhs_result = cloned!($($var)+ => { $rhs });

            // Compare using assert_approx_eq
            assert_approx_eq!(
                stringify!($lhs),
                stringify!($rhs),
                lhs_result,
                rhs_result,
                $approx_eq);
        }};
    }

    /// Checks that addition is associative.
    pub fn check_addition_associative<F, C>(x: &F, y: &F, z: &F, approx_eq: C)
    where
        F: Add<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x y z => x + (y + z), (x + y) + z, approx_eq);
    }

    /// Checks that multiplication is associative.
    pub fn check_multiplication_associative<F, C>(
        x: &F,
        y: &F,
        z: &F,
        approx_eq: C,
    ) where
        F: Mul<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x y z => x * (y * z), (x * y) * z, approx_eq);
    }

    /// Checks that addition is commutative.
    pub fn check_addition_commutative<F, C>(x: &F, y: &F, approx_eq: C)
    where
        F: Add<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x y => x + y, y + x, approx_eq);
    }

    /// Checks that multiplication is commutative.
    pub fn check_multiplication_commutative<F, C>(x: &F, y: &F, approx_eq: C)
    where
        F: Mul<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x y => x * y, y * x, approx_eq);
    }

    /// Check that the supplied `zero` is an additive identity.
    pub fn check_additive_identity<F, C>(x: &F, zero: &F, approx_eq: C)
    where
        F: Add<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x zero => x + zero, x, approx_eq);
    }

    /// Check that the supplied `one` is a multiplicative identity.
    pub fn check_multiplicative_identity<F, C>(x: &F, one: &F, approx_eq: C)
    where
        F: Mul<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x one => x * one, x, approx_eq);
    }

    /// Check that `neg` returns an additive inverse.
    pub fn check_additive_inverse<F, C>(x: &F, zero: &F, approx_eq: C)
    where
        F: Add<Output = F> + Neg<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x zero => x.clone() + (-x), zero, approx_eq);
    }

    /// Check that `inv` (when it returns a value) calculates a multiplicative
    /// inverse.
    pub fn check_multiplicative_inverse<F, C>(x: &F, one: &F, approx_eq: C)
    where
        F: Field + Debug,
        C: Fn(&F, &F) -> bool,
    {
        if let Some(reciprocal) = x.clone().inv() {
            let r = x.clone() * reciprocal;
            assert_approx_eq!("x * x.inv()", "1", r, one, approx_eq);
        }
    }

    /// Check distributivity of multiplication over addition.
    pub fn check_distributivity_mul_over_add<F, C>(
        x: &F,
        y: &F,
        z: &F,
        approx_eq: C,
    ) where
        F: Add<Output = F> + Mul<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(
            x y z =>
            x * (y + z),
            (x.clone() * y) + (x * z),
            approx_eq
        );
    }

    /// Check that addition and subtraction are inverses.
    pub fn check_add_sub_inverse<F, C>(x: &F, y: &F, approx_eq: C)
    where
        F: Add<Output = F> + Sub<Output = F> + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(x y => (x + y.clone()) - y, x, approx_eq);
    }

    /// Check that multiplication and division are inverses.
    pub fn check_mul_div_inverse<F, C>(x: &F, y: &F, zero: &F, approx_eq: C)
    where
        F: Mul<Output = F> + Div<Output = F> + Debug + Clone + PartialEq,
        C: Fn(&F, &F) -> bool,
    {
        if y != zero {
            check_approx_eq!(x y => (x * y.clone()) / y, x, approx_eq);
        }
    }

    /// Check that `AddAssign` produces similar results to `Add`.
    pub fn check_addassign_add_compatible<F, C>(x: &F, y: &F, approx_eq: C)
    where
        F: Add<Output = F> + AddAssign + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(
            x y =>
            {
                let mut xm = x;
                xm += y;
                xm
            },
            x + y,
            approx_eq
        );
    }

    /// Check that `SubAssign` produces similar results to `Sub`.
    pub fn check_subassign_sub_compatible<F, C>(x: &F, y: &F, approx_eq: C)
    where
        F: Sub<Output = F> + SubAssign + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(
            x y =>
            {
                let mut xm = x;
                xm -= y;
                xm
            },
            x - y,
            approx_eq
        );
    }

    /// Check that `MulAssign` produces similar results to `Mul`.
    pub fn check_mulassign_mul_compatible<F, C>(x: &F, y: &F, approx_eq: C)
    where
        F: Mul<Output = F> + MulAssign + Debug + Clone,
        C: Fn(&F, &F) -> bool,
    {
        check_approx_eq!(
            x y =>
            {
                let mut xm = x;
                xm *= y;
                xm
            },
            x * y,
            approx_eq
        );
    }

    /// Check that `DivAssign` produces similar results to `Div`.
    pub fn check_divassign_div_compatible<F, C>(
        x: &F,
        y: &F,
        zero: &F,
        approx_eq: C,
    ) where
        F: Div<Output = F> + DivAssign + Debug + Clone + PartialEq,
        C: Fn(&F, &F) -> bool,
    {
        if y != zero {
            check_approx_eq!(
                x y =>
                {
                    let mut xm = x;
                    xm /= y;
                    xm
                },
                x / y,
                approx_eq
            );
        }
    }

    /// Returns a strategy to generate "reasonable" `f32` values.
    ///
    /// These are values which avoid "large" values and values that are too
    /// small, and too close to zero.
    ///
    /// This "reasonable" range is chosen to satisfy the field axioms as
    /// tested.
    ///
    /// # Parameters
    ///
    /// - `include_zero`: `true` if an exact zero should be included among
    ///   the generated values.
    ///   
    /// # Returns
    ///
    /// A boxed strategy for `f32` values.
    pub fn reasonable_f32(include_zero: bool) -> BoxedStrategy<f32> {
        let mag_max = 1e3f32;
        let mag_min = 1e-1f32;
        let range_neg = -mag_max..-mag_min;
        let range_pos = mag_min..mag_max;
        if include_zero {
            prop_oneof![
                64 => range_neg,
                1 => Just(0f32),
                64 => range_pos
            ]
            .boxed()
        } else {
            prop_oneof![range_neg, range_pos].boxed()
        }
    }

    /// Returns a strategy to generate "reasonable" `f64` values.
    ///
    /// These are values which avoid "large" values and values that are too
    /// small, and too close to zero.
    ///
    /// This "reasonable" range is chosen to satisfy the field axioms as
    /// tested.
    ///
    /// # Parameters
    ///
    /// - `include_zero`: `true` if an exact zero should be included among
    ///   the generated values.
    ///   
    /// # Returns
    ///
    /// A boxed strategy for `f64` values.
    pub fn reasonable_f64(include_zero: bool) -> BoxedStrategy<f64> {
        let mag_max = 1e16f64;
        let mag_min = 1e-8f64;
        let range_neg = -mag_max..-mag_min;
        let range_pos = mag_min..mag_max;
        if include_zero {
            prop_oneof![
                64 => range_neg,
                1 => Just(0f64),
                64 => range_pos
            ]
            .boxed()
        } else {
            prop_oneof![range_neg, range_pos].boxed()
        }
    }

    /// Property test of all `Field` axioms.
    ///
    /// This also checks:
    ///   - Addition and subtraction are inverses.
    ///   - Multiplication and division are inverses.
    ///   - Operations that can be x-assign are checked against the raw op.
    pub fn prop_field_axioms<F, S, C>(strategy: S, approx_eq: C, n_cases: u32)
    where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone,
        C: Fn(&F, &F) -> bool + Clone,
    {
        let ae = approx_eq;
        let mut runner = TestRunner::new(Config {
            cases: n_cases,
            ..Config::default()
        });

        runner
            .run(
                &(strategy.clone(), strategy.clone(), strategy.clone()),
                |(x, y, z)| {
                    check_addition_associative(&x, &y, &z, ae.clone());
                    check_multiplication_associative(&x, &y, &z, ae.clone());
                    check_addition_commutative(&x, &y, ae.clone());
                    check_multiplication_commutative(&x, &y, ae.clone());
                    check_additive_identity(&x, &F::ZERO, ae.clone());
                    check_multiplicative_identity(&x, &F::ONE, ae.clone());
                    check_additive_inverse(&x, &F::ZERO, ae.clone());
                    check_multiplicative_inverse(&x, &F::ONE, ae.clone());
                    check_distributivity_mul_over_add(&x, &y, &z, ae.clone());
                    check_add_sub_inverse(&x, &y, ae.clone());
                    check_mul_div_inverse(&x, &y, &F::ZERO, ae.clone());
                    // Compatibility of operations
                    check_addassign_add_compatible(&x, &y, ae.clone());
                    check_subassign_sub_compatible(&x, &y, ae.clone());
                    check_mulassign_mul_compatible(&x, &y, ae.clone());
                    check_divassign_div_compatible(
                        &x,
                        &y,
                        &F::ZERO,
                        ae.clone(),
                    );
                    Ok(())
                },
            )
            .unwrap();
    }

    /// Test that `f32` approximately satisfies the field axioms.
    #[test]
    fn test_f32_field_axioms() {
        let abstol = 1e-2;
        let reltol = 1e-1;
        let strategy = reasonable_f32(true);
        let approx_eq =
            |a: &f32, b: &f32| approx_eq_f32_inf(*a, *b, abstol, reltol);
        let n_cases = 2048;
        prop_field_axioms(strategy, approx_eq, n_cases);
    }

    /// Test that `f64` approximately satisfies the field axioms.
    #[test]
    fn test_f64_field_axioms() {
        let abstol = 1e-12;
        let reltol = 1e-8;
        let strategy = reasonable_f64(true);
        let approx_eq =
            |a: &f64, b: &f64| approx_eq_f64_inf(*a, *b, abstol, reltol);
        let n_cases = 2048;
        prop_field_axioms(strategy, approx_eq, n_cases);
    }
}
