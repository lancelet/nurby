use crate::{
    ControlPoint,
    algebra::{Field, VectorSpace},
};

/// Rational Bézier curve.
pub struct RationalBezierCurve<F: Field, V: VectorSpace<F>> {
    control_points: Vec<ControlPoint<F, V>>,
}
impl<F: Field, V: VectorSpace<F>> RationalBezierCurve<F, V> {
    /// Creates a new `RationalBezierCurve`.
    ///
    /// # Parameters
    ///
    /// - `control_points`: Vector of rational control points.
    ///
    /// # Returns
    ///
    /// A new `RationalBezierCurve`.
    pub fn new(control_points: Vec<ControlPoint<F, V>>) -> Self {
        assert!(
            control_points.len() >= 2,
            "A RationalBezierCurve needs at least 2 control points."
        );
        Self { control_points }
    }

    /// Returns the degree of the curve.
    ///
    /// The degree is given by `number_of_control_points - 1`.
    pub fn degree(&self) -> usize {
        self.control_points.len() - 1
    }
}
