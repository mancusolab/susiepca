from __future__ import annotations

from typing import Optional, Union

from plum import dispatch

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnla
import lineax as lx

from jax._src.dtypes import JAXType  # ug...
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike, Float, Num, Shaped


_sparse_mean = sparse.sparsify(jnp.mean)


@jax.jit
@sparse.sparsify
def _get_mean_terms(geno: ArrayLike, covar: ArrayLike) -> Array:
    m, n = covar.shape
    dtype = covar.dtype
    rcond = jnp.finfo(dtype).eps * max(n, m)
    u, s, vt = jnla.svd(covar, full_matrices=False)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]

    safe_s = jnp.where(mask, s, 1).astype(covar.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)[:, jnp.newaxis]
    uTb = jnp.matmul(u.conj().T, geno, precision=lax.Precision.HIGHEST)

    beta = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)

    return beta


def _get_dense_var(geno: sparse.JAXSparse, dense_dtype: JAXType):
    # def _inner(_, variant):
    #     var_idx = jnp.mean(variant **2) - jnp.mean(variant) ** 2
    #     return _, var_idx
    #
    # _, var_geno = lax.scan(_inner, 0.0, geno.T)
    var_geno = (
        _sparse_mean(geno**2, axis=0, dtype=dense_dtype)
        - _sparse_mean(geno, axis=0, dtype=dense_dtype) ** 2
    )

    return var_geno.todense()


class _MatrixLinearOperator(lx.AbstractLinearOperator):
    """Wraps a 2-dimensional JAX array into a linear operator.

    If the matrix has shape `(a, b)` then matrix-vector multiplication (`self.mv`) is
    defined in the usual way: as performing a matrix-vector that accepts a vector of
    shape `(a,)` and returns a vector of shape `(b,)`.
    """

    matrix: Float[Array, "a b"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self, matrix: Shaped[Array, "a b"], tags: Union[object, frozenset[object]] = ()
    ):
        """**Arguments:**

        - `matrix`: a two-dimensional JAX array. For an array with shape `(a, b)` then
            this operator can perform matrix-vector products on a vector of shape
            `(b,)` to return a vector of shape `(a,)`.
        - `tags`: any tags indicating whether this matrix has any particular properties,
            like symmetry or positive-definite-ness. Note that these properties are
            unchecked and you may get incorrect values elsewhere if these tags are
            wrong.
        """
        if jnp.ndim(matrix) != 2:
            raise ValueError(
                "`MatrixLinearOperator(matrix=...)` should be 2-dimensional."
            )
        if not jnp.issubdtype(matrix, jnp.inexact):
            matrix = matrix.astype(jnp.float32)
        self.matrix = matrix
        self.tags = lx._operator._frozenset(tags)

    def mv(self, vector):
        return jnp.matmul(self.matrix, vector, precision=lax.Precision.HIGHEST)

    def as_matrix(self):
        return self.matrix

    def transpose(self):
        if lx._tags.symmetric_tag in self.tags:
            return self
        return _MatrixLinearOperator(self.matrix.T, lx._tags.transpose_tags(self.tags))

    def in_structure(self):
        _, in_size = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(in_size,), dtype=self.matrix.dtype)

    def out_structure(self):
        out_size, _ = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(out_size,), dtype=self.matrix.dtype)


class _SparseMatrixOperator(lx.AbstractLinearOperator):
    matrix: sparse.JAXSparse

    def __init__(self, matrix):
        self.matrix = matrix

    def mv(self, vector: ArrayLike):
        return sparse.sparsify(jnp.matmul)(
            self.matrix, vector, precision=lax.Precision.HIGHEST
        )

    def as_matrix(self) -> Float[Array, "n p"]:
        # raise ValueError("Refusing to materialise sparse matrix.")
        # Or you could do:
        return self.matrix.todense()

    def transpose(self) -> "_SparseMatrixOperator":
        return _SparseMatrixOperator(self.matrix.T)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        _, in_size = self.matrix.shape
        return jax.ShapeDtypeStruct((in_size,), self.matrix.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        out_size, _ = self.matrix.shape
        return jax.ShapeDtypeStruct((out_size,), self.matrix.dtype)


class SparseMatrix(lx.AbstractLinearOperator):
    data: lx.AbstractLinearOperator

    @dispatch
    def __init__(self, data: lx.AbstractLinearOperator):
        self.data = data

    @dispatch
    def __init__(
        self,
        data: sparse.JAXSparse,
        covar: Optional[ArrayLike] = None,
        scale: bool = False,
    ):
        n, p = data.shape
        geno_op = _SparseMatrixOperator(data)
        dtype = data.dtype

        if covar is None:
            covar = jnp.ones((n, 1), dtype=dtype)
            beta = _sparse_mean(data, axis=0, dtype=dtype).todense()
            beta = beta.reshape((1, p))
        else:
            beta = _get_mean_terms(data, covar)

        center_op = _MatrixLinearOperator(covar) @ _MatrixLinearOperator(beta)

        if scale:
            wgt = jnp.sqrt(_get_dense_var(data, dtype))
            scale_op = lx.DiagonalLinearOperator(1.0 / wgt)
            self.data = (geno_op - center_op) @ scale_op
        else:
            self.data = geno_op - center_op

    @property
    def dense_dtype(self) -> JAXType:
        return self.out_structure().dtype

    @dispatch
    def __matmul__(self, other: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
        return self.data @ other

    @dispatch
    def __matmul__(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.mv(vector)

    @dispatch
    def __matmul__(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "p k"]:
        return self.mm(matrix)

    @dispatch
    def __rmatmul__(
        self, other: lx.AbstractLinearOperator
    ) -> lx.AbstractLinearOperator:
        return other @ self.data

    @dispatch
    def __rmatmul__(self, vector: Num[ArrayLike, " n"]) -> Float[Array, " p"]:
        return self.T.mv(vector.T).T

    @dispatch
    def __rmatmul__(self, matrix: Num[ArrayLike, "k n"]) -> Float[Array, "k p"]:
        return self.T.mm(matrix.T).T

    def mv(self, vector: Num[ArrayLike, " p"]) -> Float[Array, " n"]:
        return self.data.mv(vector)

    def mm(self, matrix: Num[ArrayLike, "p k"]) -> Float[Array, "n k"]:
        return jax.vmap(self.data.mv, (1,), 1)(matrix)

    def as_matrix(self) -> Float[Array, "n p"]:
        return self.data.as_matrix()

    def transpose(self) -> "SparseMatrix":
        return SparseMatrix(self.data.T)

    @property
    def shape(self):
        n, *_ = self.out_structure().shape
        p, *_ = self.in_structure().shape

        return n, p

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.data.in_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.data.out_structure()


@lx.is_symmetric.register(_MatrixLinearOperator)
@lx.is_symmetric.register(_SparseMatrixOperator)
@lx.is_symmetric.register(SparseMatrix)
def _(op):
    return False


@lx.is_negative_semidefinite.register(_MatrixLinearOperator)
@lx.is_negative_semidefinite.register(_SparseMatrixOperator)
@lx.is_negative_semidefinite.register(SparseMatrix)
def _(op):
    return False


@lx.linearise.register(_MatrixLinearOperator)
@lx.linearise.register(_SparseMatrixOperator)
@lx.linearise.register(SparseMatrix)
def _(op):
    return op
