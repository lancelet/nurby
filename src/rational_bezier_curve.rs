use crate::{
    ControlPoint,
    algebra::{OrderedField, ProjectiveEmbedding, VectorSpace, clamp, interp},
};

/// Rational Bézier curve.
pub struct RationalBezierCurve<F, V>
where
    F: OrderedField,
    V: VectorSpace<F>,
{
    control_points: Vec<ControlPoint<F, V>>,
}
impl<F, V> RationalBezierCurve<F, V>
where
    F: OrderedField,
    V: VectorSpace<F>,
{
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

    /// Evaluates the curve at a parameter value using the de Casteljau
    /// recursive algorithm.
    ///
    /// # Parameters
    ///
    /// - `u`: The parameter value. This is clamped to the range
    ///   $0 \leq u \leq 1$.
    ///
    /// # Returns
    ///
    /// The vector value which results from evaluating the curve at `u`.
    pub fn decasteljau_eval<E>(&self, u: &F) -> V
    where
        E: ProjectiveEmbedding<F, V>,
    {
        // Clamp the input to the range `[0, 1]`.
        let u = clamp(&F::ZERO, &F::ONE, &u);

        // Clone control points in homogeneous form.
        let mut pts: Vec<E::Homogeneous> = self
            .control_points
            .iter()
            .map(|p| p.to_homogeneous::<E>())
            .collect();

        // Perform deCasteljau evaluation.
        for j in (0..self.degree()).rev() {
            for i in 0..j {
                pts[i] = interp(&pts[i], &pts[i + 1], &u);
            }
        }

        // Project final point.
        E::project(&pts[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /*
    proptest! {
        #[test]
        fn prop_circle(u in )
    }
    */
}
