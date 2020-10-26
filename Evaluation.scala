import breeze.linalg._

/*
* Evaluation.scala
* Last Modified: 5/3/2018
* Modified By: Andrew Roberts
*
* Defines an object which defines function to evaluate the quality of matrix approximations
* Uses the relative frobenius norm as the error metric
*/

object Eval {

	def frobenius_norm(X: DenseMatrix[Double]) : Double = {
		/*
		* Calculates the frobenius norm of the input matrix
		*
		* Arguments: 
		*			- X: mxn matrix stored as a Breeze DenseMatrix[Double]
		*
		* Returns:
		*			- Double: Frobenius norm of X
		*
		*/

		return math.sqrt(trace(X.t * X))
	}

	def rel_error(X: DenseMatrix[Double], X_approx: DenseMatrix[Double]) : Double = {
		/*
		* Computes the approximation error between two matrices
		* Approximation error defined as the relative frobenius norm
		*
		* Arguments: 
		*			- X: mxn DenseMatrix[Double], the original matrix
		*			- X_approx: mxn DenseMatrix[Double], the approximation of X
		*
		* Returns: 
		*			- Double: The relative approximation error
		*/

		return frobenius_norm(X - X_approx) / frobenius_norm(X)
	}

}
