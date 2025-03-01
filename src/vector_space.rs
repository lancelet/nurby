use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};
use num_traits::Float;

/// Vector space.
///
/// See: <https://en.wikipedia.org/wiki/Vector_space>
pub trait VectorSpace<F>:
    Sized
    + Debug
    + Display
    + Copy
    + Clone
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<F, Output = Self>
    + Div<F, Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign<F>
    + DivAssign<F>
where
    F: Float + Debug + Display,
{
    fn zero() -> Self;
    fn elements(&self) -> Vec<F>;
}

/// Interpolate linearly between vectors.
///
/// This function linearly interpolates between `p0` and `p1`, using the
/// value `u`. To obtain points in the line segment `[p0, p1]`, `u`
/// should be in the range `[0, 1]`. If `u` is outside this range, then
/// the line linearly extrapolating the line `p0` and `p1` is obtained. For
/// a value of `u = 0`, the value `p0` is returned, while for a value
/// of `u = 1`, the value `p1` is returned.
///
/// # Parameters
///
/// - `p0`: Origin point, corresponding to `u = 0`.
/// - `p1`: Destination point, corresponding to `u = 1`.
/// - `u`: Interpolation value; from `[0, 1]` for interpolation, and
///   outside that range for extrapolation.
///
/// # Returns
///
/// Interpolated value.
pub fn interp<F, V>(p0: V, p1: V, u: F) -> V
where
    F: Float + Debug + Display,
    V: VectorSpace<F>,
{
    p0 * (F::one() - u) + p1 * u
}

/// Defines a projective embedding for a vector space.
///
/// This trait allows a vector space to be embedded into a higher-dimensional
/// homogeneous space, provided that such an embedding exists for the types
/// available.
///
/// It is separate from [VectorSpace] because not all vector spaces support
/// projective embeddings.
pub trait ProjectiveEmbedding<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F>,
{
    /// The homogeneous representation of this vector space.
    ///
    /// This type should correspond to a vector space with one additional
    /// dimension. For example, for [Vec2D], `Homogeneous` should be [Vec3D].
    type Homogeneous: VectorSpace<F>;

    /// Converts a vector into its homogeneous form by appending a weight.
    ///
    /// This function maps a vector from its current space into the homogeneous
    /// space by adding a weight component.
    ///
    /// # Parameters
    /// - `v`: The vector to embed.
    /// - `w`: The weight to append.
    ///
    /// # Returns
    /// A homogeneous vector with the additional dimension.
    fn embed(v: V, w: F) -> Self::Homogeneous;

    /// Returns the vector component of the homogeneous coordinate, WITHOUT
    /// normalizing (ie. no division by the weight).
    fn get_vector_from_homogeneous(h: &Self::Homogeneous) -> V;

    /// Returns the weight component of the homogeneous coordinate.
    fn get_weight_from_homogeneous(h: &Self::Homogeneous) -> F;

    /// Converts a homogeneous vector back to the original vector space.
    ///
    /// This function performs a perspective division / dehomogenization by the
    /// homogeneous coordinate (if applicable) to return the vector in its
    /// original space.
    ///
    /// # Parameters
    /// - `h`: The homogeneous vector to project.
    ///
    /// # Returns
    /// The original vector in the lower-dimensional space.
    fn project(h: Self::Homogeneous) -> V;
}

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2D<F>
where
    F: Float + Debug + Display,
{
    x: F,
    y: F,
}
impl<F> Vec2D<F>
where
    F: Float + Debug + Display,
{
    pub fn new(x: F, y: F) -> Self {
        Vec2D { x, y }
    }
}
impl<F> Add for Vec2D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl<F> Sub for Vec2D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x - rhs.x, self.y - rhs.y)
    }
}
impl<F> Mul<F> for Vec2D<F>
where
    F: Float + Debug + Display,
{
    type Output = Vec2D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec2D::new(self.x * rhs, self.y * rhs)
    }
}
impl<F> Div<F> for Vec2D<F>
where
    F: Float + Debug + Display,
{
    type Output = Vec2D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec2D::new(self.x / rhs, self.y / rhs)
    }
}
impl<F> Neg for Vec2D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec2D::new(-self.x, -self.y)
    }
}
impl<F> AddAssign for Vec2D<F>
where
    F: Float + Debug + Display,
{
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}
impl<F> SubAssign for Vec2D<F>
where
    F: Float + Debug + Display,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
    }
}
impl<F> MulAssign<F> for Vec2D<F>
where
    F: Float + Debug + Display,
{
    fn mul_assign(&mut self, rhs: F) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
    }
}
impl<F> DivAssign<F> for Vec2D<F>
where
    F: Float + Debug + Display,
{
    fn div_assign(&mut self, rhs: F) {
        self.x = self.x / rhs;
        self.y = self.y / rhs;
    }
}
impl<F> Display for Vec2D<F>
where
    F: Float + Debug + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
impl<F> ProjectiveEmbedding<F, Vec2D<F>> for Vec2D<F>
where
    F: Float + Debug + Display,
{
    type Homogeneous = Vec3D<F>;
    fn embed(v: Vec2D<F>, w: F) -> Self::Homogeneous {
        Vec3D::new(v.x, v.y, w)
    }
    fn project(h: Self::Homogeneous) -> Vec2D<F> {
        let w = h.z;
        Vec2D::new(h.x / w, h.y / w)
    }
    fn get_vector_from_homogeneous(h: &Self::Homogeneous) -> Vec2D<F> {
        Vec2D::new(h.x, h.y)
    }
    fn get_weight_from_homogeneous(h: &Self::Homogeneous) -> F {
        h.z
    }
}
impl<F> VectorSpace<F> for Vec2D<F>
where
    F: Float + Debug + Display,
{
    fn zero() -> Self {
        let z = F::zero();
        Self::new(z, z)
    }
    fn elements(&self) -> Vec<F> {
        vec![self.x.clone(), self.y.clone()]
    }
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3D<F>
where
    F: Float + Debug + Display,
{
    x: F,
    y: F,
    z: F,
}
impl<F> Vec3D<F>
where
    F: Float + Debug + Display,
{
    pub fn new(x: F, y: F, z: F) -> Self {
        Vec3D { x, y, z }
    }
}
impl<F> Add for Vec3D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl<F: Float> Sub for Vec3D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl<F: Float> AddAssign for Vec3D<F>
where
    F: Float + Debug + Display,
{
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
        self.z = self.z + rhs.z;
    }
}
impl<F> SubAssign for Vec3D<F>
where
    F: Float + Debug + Display,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
        self.z = self.z - rhs.z;
    }
}
impl<F> Mul<F> for Vec3D<F>
where
    F: Float + Debug + Display,
{
    type Output = Vec3D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec3D::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl<F> Div<F> for Vec3D<F>
where
    F: Float + Debug + Display,
{
    type Output = Vec3D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec3D::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}
impl<F> Neg for Vec3D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec3D::new(-self.x, -self.y, -self.z)
    }
}
impl<F> MulAssign<F> for Vec3D<F>
where
    F: Float + Debug + Display,
{
    fn mul_assign(&mut self, rhs: F) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
        self.z = self.z * rhs;
    }
}
impl<F> DivAssign<F> for Vec3D<F>
where
    F: Float + Debug + Display,
{
    fn div_assign(&mut self, rhs: F) {
        self.x = self.x / rhs;
        self.y = self.y / rhs;
        self.z = self.z / rhs;
    }
}
impl<F> Display for Vec3D<F>
where
    F: Float + Debug + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl<F> ProjectiveEmbedding<F, Vec3D<F>> for Vec3D<F>
where
    F: Float + Debug + Display,
{
    type Homogeneous = Vec4D<F>;
    fn embed(v: Vec3D<F>, w: F) -> Self::Homogeneous {
        Vec4D::new(v.x, v.y, v.z, w)
    }
    fn project(h: Self::Homogeneous) -> Vec3D<F> {
        Vec3D::new(h.x / h.w, h.y / h.w, h.z / h.w)
    }
    fn get_vector_from_homogeneous(h: &Self::Homogeneous) -> Vec3D<F> {
        Vec3D::new(h.x, h.y, h.z)
    }
    fn get_weight_from_homogeneous(h: &Self::Homogeneous) -> F {
        h.w
    }
}
impl<F> VectorSpace<F> for Vec3D<F>
where
    F: Float + Debug + Display,
{
    fn zero() -> Self {
        let z = F::zero();
        Self::new(z, z, z)
    }
    fn elements(&self) -> Vec<F> {
        vec![self.x.clone(), self.y.clone(), self.z.clone()]
    }
}

/// 4D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4D<F>
where
    F: Float + Debug + Display,
{
    x: F,
    y: F,
    z: F,
    w: F,
}
impl<F> Vec4D<F>
where
    F: Float + Debug + Display,
{
    pub fn new(x: F, y: F, z: F, w: F) -> Self {
        Vec4D { x, y, z, w }
    }
}
impl<F> Add for Vec4D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec4D::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}
impl<F> Sub for Vec4D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec4D::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}
impl<F> AddAssign for Vec4D<F>
where
    F: Float + Debug + Display,
{
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
        self.z = self.z + rhs.z;
        self.w = self.w + rhs.w;
    }
}
impl<F> SubAssign for Vec4D<F>
where
    F: Float + Debug + Display,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
        self.z = self.z - rhs.z;
        self.w = self.w - rhs.w;
    }
}
impl<F> Mul<F> for Vec4D<F>
where
    F: Float + Debug + Display,
{
    type Output = Vec4D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec4D::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}
impl<F> Div<F> for Vec4D<F>
where
    F: Float + Debug + Display,
{
    type Output = Vec4D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec4D::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}
impl<F> Neg for Vec4D<F>
where
    F: Float + Debug + Display,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec4D::new(-self.x, -self.y, -self.z, -self.w)
    }
}
impl<F> MulAssign<F> for Vec4D<F>
where
    F: Float + Debug + Display,
{
    fn mul_assign(&mut self, rhs: F) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
        self.z = self.z * rhs;
        self.w = self.w * rhs;
    }
}
impl<F> DivAssign<F> for Vec4D<F>
where
    F: Float + Debug + Display,
{
    fn div_assign(&mut self, rhs: F) {
        self.x = self.x / rhs;
        self.y = self.y / rhs;
        self.z = self.z / rhs;
        self.w = self.w / rhs;
    }
}
impl<F> Display for Vec4D<F>
where
    F: Float + Debug + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}
impl<F: Float> VectorSpace<F> for Vec4D<F>
where
    F: Float + Debug + Display,
{
    fn zero() -> Self {
        let z = F::zero();
        Self::new(z, z, z, z)
    }
    fn elements(&self) -> Vec<F> {
        vec![
            self.x.clone(),
            self.y.clone(),
            self.z.clone(),
            self.w.clone(),
        ]
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{
        assert_approx_eq,
        test_util::{approx_eq, reasonable_f32, reasonable_f64},
    };

    use super::*;
    use proptest::{
        prelude::*,
        test_runner::{Config, TestRunner},
    };

    pub fn strategy_vec2d<F, SF>(strategy_f: SF) -> BoxedStrategy<Vec2D<F>>
    where
        F: Float + Debug + Display,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (strategy_f.clone(), strategy_f)
            .prop_map(|(x, y)| Vec2D::new(x, y))
            .boxed()
    }

    pub fn strategy_vec3d<F, SF>(strategy_f: SF) -> BoxedStrategy<Vec3D<F>>
    where
        F: Float + Debug + Display,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (strategy_f.clone(), strategy_f.clone(), strategy_f.clone())
            .prop_map(|(x, y, z)| Vec3D::new(x, y, z))
            .boxed()
    }

    pub fn strategy_vec4d<F, SF>(strategy_f: SF) -> BoxedStrategy<Vec4D<F>>
    where
        F: Float + Debug + Display,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (
            strategy_f.clone(),
            strategy_f.clone(),
            strategy_f.clone(),
            strategy_f.clone(),
        )
            .prop_map(|(x, y, z, w)| Vec4D::new(x, y, z, w))
            .boxed()
    }

    pub fn elementwise_approx_eq<F, V, C>(a: V, b: V, approx_eq: C) -> bool
    where
        F: Float + Debug + Display,
        V: VectorSpace<F>,
        C: Fn(F, F) -> bool,
    {
        let a_elems = a.elements();
        let b_elems = b.elements();
        assert!(a_elems.len() == b_elems.len());
        a_elems
            .into_iter()
            .zip(b_elems.into_iter())
            .all(|(ae, be)| approx_eq(ae, be))
    }

    pub fn check_vector_space_axioms<F, V, C>(
        x: V,
        y: V,
        z: V,
        a: F,
        b: F,
        eq: C,
    ) where
        F: Float + Debug + Display,
        V: VectorSpace<F>,
        C: Fn(V, V) -> bool,
    {
        // ---- Basic Axioms

        // Associativity of vector addition.
        assert_approx_eq!(x + (y + z), (x + y) + z, eq);

        // Commutativity of vector addition.
        assert_approx_eq!(x + y, y + x, eq);

        // Identity element of vector addition (zero).
        assert_approx_eq!(x + V::zero(), x, eq);

        // Inverse elements of vector addition (neg).
        assert_approx_eq!(x + x.neg(), V::zero(), eq);

        // Compatibility of scalar multiplication with field multiplication
        assert_approx_eq!((x * b) * a, x * (a * b), eq);

        // Identity element of scalar multiplication
        assert_approx_eq!(x * F::one(), x, eq);

        // Distributivity of scalar multiplication with respect to vector
        // addition
        assert_approx_eq!((x + y) * a, x * a + y * a, eq);

        // Distributivity of scalar multiplication with respect to field
        // addition
        assert_approx_eq!(x * (a + b), x * a + x * b, eq);

        // ---- Inverse Operations

        // Addition and subtraction are inverses
        assert_approx_eq!((x + y) - y, x, eq);

        // Multiplication by a scalar and division by a scalar are inverses.
        if a != F::zero() {
            assert_approx_eq!((x * a) / a, x, eq);
        }

        // ---- Compatibility of XxxAssign Operations

        // Add and AddAssign
        assert_approx_eq!(
            x + y,
            {
                let mut xx = x;
                xx += y;
                xx
            },
            eq
        );

        // Sub and SubAssign
        assert_approx_eq!(
            x - y,
            {
                let mut xx = x;
                xx -= y;
                xx
            },
            eq
        );

        // Mul and MulAssign
        assert_approx_eq!(
            x * a,
            {
                let mut xx = x;
                xx *= a;
                xx
            },
            eq
        );

        // Div and DivAssign
        if a != F::zero() {
            assert_approx_eq!(
                x / a,
                {
                    let mut xx = x;
                    xx /= a;
                    xx
                },
                eq
            );
        }
    }

    pub fn prop_vector_space_axioms<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
        n_cases: u32,
    ) where
        F: Float + Debug + Display,
        V: VectorSpace<F>,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(V, V) -> bool + Clone,
    {
        let mut runner = TestRunner::new(Config::with_cases(n_cases));
        runner
            .run(
                &(
                    strategy_v.clone(),
                    strategy_v.clone(),
                    strategy_v.clone(),
                    strategy_f.clone(),
                    strategy_f.clone(),
                ),
                |(x, y, z, a, b)| {
                    check_vector_space_axioms(x, y, z, a, b, approx_eq.clone());
                    Ok(())
                },
            )
            .unwrap()
    }

    pub fn check_projective_embedding<F, V, C>(x: V, w: F, eq: C)
    where
        F: Float + Debug + Display,
        V: VectorSpace<F> + ProjectiveEmbedding<F, V> + PartialEq,
        C: Fn(V, V) -> bool,
    {
        // Construct the embedding into the projective / homogeneous space and
        // check that its elements are correct.
        let h = V::embed(x, w);
        let mut expected_h_elements = x.elements();
        expected_h_elements.push(w);
        assert_eq!(expected_h_elements, h.elements());

        // Check that get_vector and get_weight return correct values.
        assert_eq!(x, V::get_vector_from_homogeneous(&h));
        assert_eq!(w, V::get_weight_from_homogeneous(&h));

        // Project / de-homogenise from the projective space.
        if !w.is_zero() {
            assert_approx_eq!(x / w, V::project(h), eq);
        }
    }

    pub fn prop_projective_embedding<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
        n_cases: u32,
    ) where
        F: Float + Debug + Display,
        V: VectorSpace<F> + ProjectiveEmbedding<F, V> + PartialEq,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(V, V) -> bool + Clone,
    {
        let mut runner = TestRunner::new(Config::with_cases(n_cases));
        runner
            .run(&(strategy_v.clone(), strategy_f.clone()), |(x, w)| {
                check_projective_embedding(x, w, approx_eq.clone());
                Ok(())
            })
            .unwrap()
    }

    pub fn check_interp<F, V, CF, CV>(p0: V, p1: V, u: F, eq_f: CF, eq_v: CV)
    where
        F: Float + Debug + Display,
        V: VectorSpace<F>,
        CF: Fn(F, F) -> bool,
        CV: Fn(V, V) -> bool,
    {
        // When u = 0, the result is p0.
        assert_approx_eq!(interp(p0, p1, F::zero()), p0, eq_v);

        // When u = 1, the result is p1.
        assert_approx_eq!(interp(p0, p1, F::one()), p1, eq_v);

        // For the given value of u, we interpolate.
        let r = interp(p0, p1, u);
        assert!(
            p0.elements()
                .into_iter()
                .zip(p1.elements().into_iter())
                .map(|(e0, e1)| e0 * (F::one() - u) + e1 * u)
                .zip(r.elements())
                .all(|(ei, ri)| eq_f(ei, ri))
        );
    }

    pub fn prop_check_interp<F, V, SF, SV, CF, CV>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq_f: CF,
        approx_eq_v: CV,
        n_cases: u32,
    ) where
        F: Float + Debug + Display,
        V: VectorSpace<F>,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        CF: Fn(F, F) -> bool + Clone,
        CV: Fn(V, V) -> bool + Clone,
    {
        let mut runner = TestRunner::new(Config::with_cases(n_cases));
        runner
            .run(
                &(strategy_v.clone(), strategy_v.clone(), strategy_f.clone()),
                |(p0, p1, u)| {
                    check_interp(
                        p0,
                        p1,
                        u,
                        approx_eq_f.clone(),
                        approx_eq_v.clone(),
                    );
                    Ok(())
                },
            )
            .unwrap()
    }

    #[test]
    pub fn test_vec2d_f32_vector_space_axioms() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_vector_space_axioms(
            reasonable_f32(true),
            strategy_vec2d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f32_interp() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_check_interp(
            reasonable_f32(true),
            strategy_vec2d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f32_projective_embedding() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_projective_embedding(
            reasonable_f32(true),
            strategy_vec2d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f32_vector_space_axioms() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_vector_space_axioms(
            reasonable_f32(true),
            strategy_vec3d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f32_interp() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_check_interp(
            reasonable_f32(true),
            strategy_vec3d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f32_projective_embedding() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_projective_embedding(
            reasonable_f32(true),
            strategy_vec3d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f32_vector_space_axioms() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_vector_space_axioms(
            reasonable_f32(true),
            strategy_vec4d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f32_interp() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_check_interp(
            reasonable_f32(true),
            strategy_vec4d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_vector_space_axioms() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_vector_space_axioms(
            reasonable_f64(true),
            strategy_vec2d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_interp() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_check_interp(
            reasonable_f64(true),
            strategy_vec4d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_projective_embedding() {
        let abstol = 1e-3f64;
        let reltol = 1e-1f64;

        prop_projective_embedding(
            reasonable_f64(true),
            strategy_vec2d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f64_vector_space_axioms() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_vector_space_axioms(
            reasonable_f64(true),
            strategy_vec3d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f64_interp() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_check_interp(
            reasonable_f64(true),
            strategy_vec3d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f64_projective_embedding() {
        let abstol = 1e-3f64;
        let reltol = 1e-1f64;

        prop_projective_embedding(
            reasonable_f64(true),
            strategy_vec3d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f64_vector_space_axioms() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_vector_space_axioms(
            reasonable_f64(true),
            strategy_vec4d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f64_interp() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_check_interp(
            reasonable_f64(true),
            strategy_vec4d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_display() {
        assert_eq!("(2.2, 4.2)", Vec2D::new(2.2f64, 4.2f64).to_string());
    }

    #[test]
    pub fn test_vec3d_f64_display() {
        assert_eq!(
            "(2.2, 4.2, 6.2)",
            Vec3D::new(2.2f64, 4.2f64, 6.2f64).to_string()
        );
    }

    #[test]
    pub fn test_vec4d_f64_display() {
        assert_eq!(
            "(2.2, 4.2, 6.2, 8.2)",
            Vec4D::new(2.2f64, 4.2f64, 6.2f64, 8.2f64).to_string()
        );
    }

    #[test]
    pub fn test_interp_example() {
        let p0 = Vec2D::new(1f32, 2f32);
        let p1 = Vec2D::new(3f32, 5f32);

        // end points
        assert_eq!(p0, interp(p0, p1, 0.0));
        assert_eq!(p1, interp(p0, p1, 1.0));
        // mid point
        assert_eq!(Vec2D::new(2f32, 3.5f32), interp(p0, p1, 0.5));
        // extrapolation
        assert_eq!(Vec2D::new(5f32, 8f32), interp(p0, p1, 2.0));
    }
}
