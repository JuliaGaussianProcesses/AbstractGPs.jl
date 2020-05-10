
"""
     update_chol(chol::Cholesky, C12::AbstractMatrix, C22::AbstractMatrix)

Let `C` be the positive definite matrix comprising blocks
```julia
C = [C11 C12;
     C21 C22]
```
with upper-triangular cholesky factorisation comprising blocks
```julia
U = [U11 U12;
     0   U22]
```
where `U11` and `U22` are themselves upper-triangular, and `U11 = cholesky(C11).U`.
update_chol computes the updated Cholesky given original `chol`, `C12`, and `C22`.

# Arguments
     - chol::Cholesky: The original cholesky decomposition
     - C12::AbstractMatrix: matrix of size (size(chol.U, 1), size(C22, 1))
     - C22::AbstractMatrix: positive-definite matrix
"""
function update_chol(chol::Cholesky, C12::AbstractMatrix, C22::AbstractMatrix)
    U12 = chol.U' \ C12
    U22 = cholesky(Symmetric(C22 - U12'U12)).U
    return Cholesky([chol.U U12; zero(U12') U22], 'U', 0)
end

