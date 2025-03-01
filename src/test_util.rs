use num_traits::Float;
use proptest::prelude::*;

/// Checks for approximate `Float` equality.
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
/// If either of these two types of equality are satisfied then the function
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
pub fn approx_eq<F>(a: F, b: F, abstol: F, reltol: F) -> bool
where
    F: Float,
{
    debug_assert!(
        abstol >= F::zero() && reltol >= F::zero(),
        "approx_eq: abstol and reltol must be >= 0"
    );
    let diff = (a - b).abs();
    let largest_mag = F::max(a.abs(), b.abs());
    diff <= reltol * largest_mag || diff <= abstol
}

/// Returns a strategy to generate "reasonable" `f32` values.
///
/// These are values which avoid "large" values and values that are too
/// small, and too close to zero. This "reasonable" range is chosen to
/// pass properties that test multiple `f64` values in bulk.
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
/// small, and too close to zero. This "reasonable" range is chosen to
/// pass properties that test multiple `f64` values in bulk.
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

/// Asserts that two values are approximately equal, using a supplied
/// comparison function.
#[macro_export]
macro_rules! assert_approx_eq {
    ($lhs:expr, $rhs:expr, $approx_eq:expr) => {
        let lhs_result = { $lhs };
        let rhs_result = { $rhs };

        assert!(
            $approx_eq(lhs_result, rhs_result),
            "Expected {} = {}, but got:\n  {} = {:?}\n  {} = {:?}\n",
            stringify!($lhs),
            stringify!($rhs),
            stringify!($lhs),
            lhs_result,
            stringify!($rhs),
            rhs_result
        );
    };
}
