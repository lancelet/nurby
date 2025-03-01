use super::Field;

/// Mathematical ordered field.
///
/// See: <https://en.wikipedia.org/wiki/Ordered_field>
pub trait OrderedField: Field + PartialOrd {}

/// OrderedField implementation for f32.
impl OrderedField for f32 {}

/// OrderedField implementation for f64.
impl OrderedField for f64 {}

/// Clamp a value to a range.
///
/// $$
/// \begin{cases}
/// \mathrm{min} & \textrm{if } \mathrm{value} < \mathrm{min} \\\\
/// \mathrm{max} & \textrm{if } \mathrm{value} > \mathrm{max} \\\\
/// \mathrm{value} & \textrm{otherwise}
/// \end{cases}
/// $$
///
/// This clamps the `value` to the range `[min, max]`. If the value is less
/// than `min` then `min` is returned, if the value is greater than `max` then
/// `max` is returned, otherwise the original values is returned unchanged.
///
/// # Parameters
///
/// - `min`: Minimum value (inclusive).
/// - `max`: Maximum value (inclusive).
/// - `value`: Value to clamp.
///
/// # Returns
///
/// A value clamped to the range `[min, max]`.
pub fn clamp<F>(min: &F, max: &F, value: &F) -> F
where
    F: OrderedField,
{
    debug_assert!(min <= max, "clamp: called with min > max");
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
    .clone()
}
