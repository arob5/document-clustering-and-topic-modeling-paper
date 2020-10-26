import breeze.linalg._

/*
* SVD.scala
* Last Modified: 5/3/2018
* Modified By: Andrew Roberts
*
* Defines an SVD object, implementing functions that use the Breeze Singular Value Decomposition (SVD) to provide a 
* framework to conduct dimensionality reduction and Latent Semantic analysis
*/


object SVD {

	def get_svd_factors(X: DenseMatrix[Double], 
		full_matrices: Boolean = false) : (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
		/*
		* Computes the factors of the Singular Value Decomposition (SVD) of an input matrix
		*
		* Arguments: 
		*			- X: mxn input matrix stored as Breeze DenseMatrix[Double]
		*			- full_matrices: Boolean; if true, returns matrices without excess rows/cols of zeros removed
		*	
		* Returns:
		*			- Tuple: If full_matrices==true, elements are U(mxm), S(mxm), Vt(nxn) 
		*					 If full_matrices==false and n>m, elements are U(mxm), S(mxm), Vt(mxn)
		*					 If full_matrices==false and m>n, elements are U(mxn), S(nxn), Vt(nxn)
		*/					

		val svd_factors = svd(X)
		var U = svd_factors.U
		var S = diag(svd_factors.S)
		var Vt = svd_factors.Vt


		if(full_matrices) 
			return (U, S, Vt)
		else {
			val r = min(X.rows, X.cols)
			return (U(::, 0 until r), S, Vt(0 until r, ::))
		}
	}
	
	def svd_compression(U: DenseMatrix[Double], S: DenseMatrix[Double], Vt: DenseMatrix[Double], k: Int) : DenseMatrix[Double] = {
		/*
		* Using SVD factors from get_svd_factors, uses first k singular vectors/values to calculate an approximation of the original matrix
		*
		* Arguments: 
		*			-U, S, Vt; the DenseMatrix[Double] factors returned from get_svd_factors
		*			-k: Approximation rank
		*
		* Returns:
		*			-Approximated rank k matrix as Breeze DenseMatrix[Double]
		*/
		
		U(::, 0 until k) * S(0 until k, 0 until k) * Vt(0 until k, ::)

	}
	
}
