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
    fn scalar_mul(&self, value: &F) -> Self;
    fn scalar_div(&self, value: &F) -> Self;
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
    fn scalar_mul(&self, value: &F) -> Self {
        Vec2D::new(
            self.x.clone() * value.clone(),
            self.y.clone() * value.clone(),
        )
    }
    fn scalar_div(&self, value: &F) -> Self {
        Vec2D::new(
            self.x.clone() / value.clone(),
            self.y.clone() / value.clone(),
        )
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
    fn scalar_mul(&self, value: &F) -> Self {
        Vec3D::new(
            self.x.clone() * value.clone(),
            self.y.clone() * value.clone(),
            self.z.clone() * value.clone(),
        )
    }
    fn scalar_div(&self, value: &F) -> Self {
        Vec3D::new(
            self.x.clone() / value.clone(),
            self.y.clone() / value.clone(),
            self.z.clone() / value.clone(),
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

    pub fn prop_multiplicative_identity<F, V, S, C>(strategy: S, approx_eq: C)
    where
        F: Field,
        V: Vec<F> + Debug,
        S: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        proptest!(
            |(x in strategy)| {
                let r = x.scalar_mul(&F::ONE);
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

    pub fn prop_vec<F, V, S, C>(strategy: S, approx_eq: C)
    where
        F: Field,
        V: Vec<F> + Debug,
        S: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        prop_addition_associative(strategy.clone(), approx_eq.clone());
        prop_addition_commutative(strategy.clone(), approx_eq.clone());
        prop_additive_identity(strategy.clone(), approx_eq.clone(), V::ZERO);
        prop_additive_inverse(strategy.clone(), approx_eq.clone(), V::ZERO);
        prop_multiplicative_identity(strategy.clone(), approx_eq.clone());
        // TODO: Compatibility of scalar multiplication
        // TODO: Distributivity of scalar multiplication wrt vector addition
        // TODO: Distributivity of scalar multiplication wrt field addition
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

    fn strategy_vec2d_f32() -> BoxedStrategy<Vec2D<f32>> {
        let max: f32 = 1e5;
        let min: f32 = 1e-2;
        let field_strategy = prop_oneof![-max..-min, min..max].boxed();
        strategy_vec2d(field_strategy)
    }

    fn strategy_vec2d_f64() -> BoxedStrategy<Vec2D<f64>> {
        let max: f64 = 1e5;
        let min: f64 = 1e-2;
        let field_strategy = prop_oneof![-max..-min, min..max].boxed();
        strategy_vec2d(field_strategy)
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

    #[test]
    fn test_vec2d_f32() {
        prop_vec(strategy_vec2d_f32(), approxeq_vec2d_f32);
    }

    #[test]
    fn test_vec2d_f64() {
        prop_vec(strategy_vec2d_f64(), approxeq_vec2d_f64);
    }

    #[test]
    fn test_vec3d_f32() {
        prop_vec(strategy_vec3d_f32(), approxeq_vec3d_f32);
    }

    #[test]
    fn test_vec3d_f64() {
        prop_vec(strategy_vec3d_f64(), approxeq_vec3d_f64);
    }
}
