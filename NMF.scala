import breeze.linalg._
import scala.math

/*
* NMF.scala
* Last Modified: 5/3/2018
* Modified By: Andrew Roberts
*
* This file contains the NMF object, with implementations of various functions to perform the 
* Non-Negative Matrix Factorization (NMF) on a Breeze DenseMatrix. This is a relatively simple
* implementation using the multiplicative update rules derived by Lee and Seung
*/

object NMF {
	def nmf(X: DenseMatrix[Double], r: Int, n_itr: Int ) : (DenseMatrix[Double], DenseMatrix[Double]) = {
		/*
		* Factors matrix into a product of two matrices, using standard NMF multiplicative update rules
		* 
		* Arguments:
		*			- X: mxn matrix stored as a Breeze DenseMatrix[Double]
		*			- r: Rank of low-dim basis (i.e., # cols in W, # rows in H)
		*				 Typically, if X is mxn, then r << min(m, n)
		*			- n_itr: Number of iterations algorithm will run
		*
		* Returns:
		*			- Tuple: First element is W (mxr); second element is H (rxn)
		*/			

		val epsilon = java.lang.Double.MIN_VALUE
		var W = DenseMatrix.rand(X.rows, r)
		var H = DenseMatrix.rand(r, X.cols)
	
		for(i <- 0 until n_itr) {
			W = W *:* ((X * H.t) / ((W * H * H.t) + epsilon))
			H = H *:* ((W.t * X) / ((W.t * W * H) + epsilon))	
		}

		return (W, H)
	}

	def normalize_cols(W: DenseMatrix[Double]) : DenseMatrix[Double] = {
		/*
		* Normalizes columns of matrix so that column-wise sums are all 1
		*
		* Arguments: 
		*			- W: mxn matrix stored as a Breeze DenseMatrix[Double]
		* 
		* Returns: 
		*			- DenseMatrix[Double]: W with normalized columns
		*/

		W(*, ::) :/ sum(W, Axis._0).t
	}

}
