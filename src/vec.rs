use crate::field::Field;
use std::ops::{Add, AddAssign, Neg, Sub};

/// Vector space.
///
/// See: https://en.wikipedia.org/wiki/Vector_space
pub trait Vec<F: Field>:
    Sized
    + Clone
    + PartialEq
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    fn scalar_mul(&self, value: F) -> Self;
    fn scalar_div(&self, value: F) -> Self;
}

/// 2D vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec2D<F: Field> {
    x: F,
    y: F,
}
impl<F: Field> Vec2D<F> {
    pub fn new(x: F, y: F) -> Self {
        Vec2D { x, y }
    }
}
impl<F: Field> Add for Vec2D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl<F: Field> Sub for Vec2D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x - rhs.x, self.y - rhs.y)
    }
}
impl<F: Field> Neg for Vec2D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec2D::new(-self.x, -self.y)
    }
}
impl<F: Field> AddAssign for Vec2D<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}
impl<F: Field> Vec<F> for Vec2D<F> {
    const ZERO: Self = Vec2D {
        x: F::ZERO,
        y: F::ZERO,
    };
    fn scalar_mul(&self, value: F) -> Self {
        Vec2D::new(self.x.clone() * value.clone(), self.y.clone() * value)
    }
    fn scalar_div(&self, value: F) -> Self {
        Vec2D::new(self.x.clone() / value.clone(), self.y.clone() / value)
    }
}

/// 3D vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec3D<F: Field> {
    x: F,
    y: F,
    z: F,
}
impl<F: Field> Vec3D<F> {
    pub fn new(x: F, y: F, z: F) -> Self {
        Vec3D { x, y, z }
    }
}
impl<F: Field> Add for Vec3D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl<F: Field> Sub for Vec3D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl<F: Field> AddAssign for Vec3D<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl<F: Field> Neg for Vec3D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec3D::new(-self.x, -self.y, -self.z)
    }
}
impl<F: Field> Vec<F> for Vec3D<F> {
    const ZERO: Self = Vec3D {
        x: F::ZERO,
        y: F::ZERO,
        z: F::ZERO,
    };
    fn scalar_mul(&self, value: F) -> Self {
        Vec3D::new(
            self.x.clone() * value.clone(),
            self.y.clone() * value.clone(),
            self.z.clone() * value,
        )
    }
    fn scalar_div(&self, value: F) -> Self {
        Vec3D::new(
            self.x.clone() / value.clone(),
            self.y.clone() / value.clone(),
            self.z.clone() / value,
        )
    }
}

#[cfg(test)]
pub mod tests {
    use crate::field::tests::{
        approx_eq_f32_absrel, approx_eq_f64_absrel, prop_addition_associative,
        prop_addition_commutative, prop_additive_identity,
        prop_additive_inverse,
    };

    use super::*;
    use core::fmt::Debug;
    use proptest::prelude::*;

    /// Property test checking the multiplicative identity of a vector.
    pub fn prop_multiplicative_identity<F, V, S, C>(strategy: S, approx_eq: C)
    where
        F: Field,
        V: Vec<F> + Debug,
        S: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        proptest!(
            |(x in strategy)| {
                let r = x.scalar_mul(F::ONE);
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

    /// Property test checking the compatibility of scalar multiplication.
    pub fn prop_compatible_scalar_multiplication<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
    ) where
        F: Field + Debug,
        V: Vec<F> + Debug,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        proptest!(
            |(a in strategy_f.clone(), b in strategy_f, v in strategy_v)| {
                let r1 = v.clone().scalar_mul(a.clone()).scalar_mul(b.clone());
                let r2 = v.clone().scalar_mul(a * b);
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected (v * a) * b ≈ v * (a * b), but got:\n\
                    (v * a) * b = {:?}\n\
                    v * (a * b) = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test checking distributivity of scalar multiplication with
    /// respect to vector addition.
    pub fn prop_distributivity_wrt_vectors<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
    ) where
        F: Field + Debug,
        V: Vec<F> + Debug,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        proptest!(
            |(a in strategy_f, u in strategy_v.clone(), v in strategy_v)| {
                let r1 = (u.clone() + v.clone()).scalar_mul(a.clone());
                let r2 = u.scalar_mul(a.clone()) + v.scalar_mul(a);
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected (u + v) * a ≈ u * a + u * v, but got:\n\
                    (u + v) * a   = {:?}\n\
                    u * a + u * v = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test checking distributivity of scalar multiplication with
    /// respect to field addition.
    pub fn prop_distributivity_wrt_field<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
    ) where
        F: Field + Debug,
        V: Vec<F> + Debug,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        proptest!(
            |(a in strategy_f.clone(), b in strategy_f, v in strategy_v)| {
                let r1 = v.clone().scalar_mul(a.clone() + b.clone());
                let r2 = v.clone().scalar_mul(a) + v.scalar_mul(b);
                prop_assert!(
                    approx_eq(&r1, &r2),
                    "Expected v * (a + b) ≈ v * a + v * b, but got:\n\
                    v * (a + b)   = {:?}\n\
                    v * a + v * b = {:?}\n",
                    r1, r2
                );
            }
        )
    }

    /// Property test to check that scalar multiplication and scalar division
    /// are inverses.
    pub fn prop_scalar_mul_div_inverse<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
    ) where
        F: Field + Debug,
        V: Vec<F> + Debug,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        proptest!(
            |(a in strategy_f, v in strategy_v)| {
                let r = v.scalar_mul(a.clone()).scalar_div(a);
                prop_assert!(
                    approx_eq(&r, &v),
                    "Expected v * a / a ≈ v, but got:\n\
                    v         = {:?}\n\
                    v * a / a = {:?}\n",
                    v, r
                );
            }
        )
    }

    /// Property test checking the vector axioms.
    pub fn prop_vec<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
    ) where
        F: Field + Debug,
        V: Vec<F> + Debug,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        prop_addition_associative(strategy_v.clone(), approx_eq.clone());
        prop_addition_commutative(strategy_v.clone(), approx_eq.clone());
        prop_additive_identity(strategy_v.clone(), approx_eq.clone(), V::ZERO);
        prop_additive_inverse(strategy_v.clone(), approx_eq.clone(), V::ZERO);
        prop_multiplicative_identity(strategy_v.clone(), approx_eq.clone());
        prop_compatible_scalar_multiplication(
            strategy_f.clone(),
            strategy_v.clone(),
            approx_eq.clone(),
        );
        prop_distributivity_wrt_vectors(
            strategy_f.clone(),
            strategy_v.clone(),
            approx_eq.clone(),
        );
        prop_distributivity_wrt_field(
            strategy_f.clone(),
            strategy_v.clone(),
            approx_eq.clone(),
        );
        prop_scalar_mul_div_inverse(
            strategy_f.clone(),
            strategy_v.clone(),
            approx_eq.clone(),
        );
    }

    fn strategy_vec2d<F, S>(strategy: S) -> BoxedStrategy<Vec2D<F>>
    where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone + 'static,
    {
        (strategy.clone(), strategy.clone())
            .prop_map(|(x, y)| Vec2D::new(x, y))
            .boxed()
    }

    fn strategy_f32() -> BoxedStrategy<f32> {
        let max: f32 = 1e5;
        let min: f32 = 1e-2;
        prop_oneof![-max..-min, min..max].boxed()
    }

    fn strategy_f64() -> BoxedStrategy<f64> {
        let max: f64 = 1e5;
        let min: f64 = 1e-2;
        prop_oneof![-max..-min, min..max].boxed()
    }

    fn strategy_vec2d_f32() -> BoxedStrategy<Vec2D<f32>> {
        strategy_vec2d(strategy_f32())
    }

    fn strategy_vec2d_f64() -> BoxedStrategy<Vec2D<f64>> {
        strategy_vec2d(strategy_f64())
    }

    fn approxeq_vec2d_f32(a: &Vec2D<f32>, b: &Vec2D<f32>) -> bool {
        let abstol = 1e-4;
        let reltol = 1e-3;
        approx_eq_f32_absrel(abstol, reltol)(&a.x, &b.x)
            && approx_eq_f32_absrel(abstol, reltol)(&a.y, &b.y)
    }

    fn approxeq_vec2d_f64(a: &Vec2D<f64>, b: &Vec2D<f64>) -> bool {
        let abstol = 1e-8;
        let reltol = 1e-6;
        approx_eq_f64_absrel(abstol, reltol)(&a.x, &b.x)
            && approx_eq_f64_absrel(abstol, reltol)(&a.y, &b.y)
    }

    fn strategy_vec3d<F, S>(strategy: S) -> BoxedStrategy<Vec3D<F>>
    where
        F: Field + Debug,
        S: Strategy<Value = F> + Clone + 'static,
    {
        (strategy.clone(), strategy.clone(), strategy.clone())
            .prop_map(|(x, y, z)| Vec3D::new(x, y, z))
            .boxed()
    }

    fn strategy_vec3d_f32() -> BoxedStrategy<Vec3D<f32>> {
        let max: f32 = 1e5;
        let min: f32 = 1e-2;
        let field_strategy = prop_oneof![-max..-min, min..max].boxed();
        strategy_vec3d(field_strategy)
    }

    fn strategy_vec3d_f64() -> BoxedStrategy<Vec3D<f64>> {
        let max: f64 = 1e5;
        let min: f64 = 1e-2;
        let field_strategy = prop_oneof![-max..-min, min..max].boxed();
        strategy_vec3d(field_strategy)
    }

    fn approxeq_vec3d_f32(a: &Vec3D<f32>, b: &Vec3D<f32>) -> bool {
        let abstol = 1e-4;
        let reltol = 1e-3;
        approx_eq_f32_absrel(abstol, reltol)(&a.x, &b.x)
            && approx_eq_f32_absrel(abstol, reltol)(&a.y, &b.y)
            && approx_eq_f32_absrel(abstol, reltol)(&a.z, &b.z)
    }

    fn approxeq_vec3d_f64(a: &Vec3D<f64>, b: &Vec3D<f64>) -> bool {
        let abstol = 1e-8;
        let reltol = 1e-6;
        approx_eq_f64_absrel(abstol, reltol)(&a.x, &b.x)
            && approx_eq_f64_absrel(abstol, reltol)(&a.y, &b.y)
            && approx_eq_f64_absrel(abstol, reltol)(&a.z, &b.z)
    }

    /// Test that a Vec2D<f32> is a valid vector.
    #[test]
    fn test_vec2d_f32() {
        prop_vec(strategy_f32(), strategy_vec2d_f32(), approxeq_vec2d_f32);
    }

    /// Test that a Vec2D<f64> is a valid vector.
    #[test]
    fn test_vec2d_f64() {
        prop_vec(strategy_f64(), strategy_vec2d_f64(), approxeq_vec2d_f64);
    }

    /// Test that a Vec3D<f32> is a valid vector.
    #[test]
    fn test_vec3d_f32() {
        prop_vec(strategy_f32(), strategy_vec3d_f32(), approxeq_vec3d_f32);
    }

    /// Test that a Vec3D<f64> is a valid vector.
    #[test]
    fn test_vec3d_f64() {
        prop_vec(strategy_f64(), strategy_vec3d_f64(), approxeq_vec3d_f64);
    }
}
