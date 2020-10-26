import breeze.numerics._
import breeze.linalg._
import java.io._

/* 
* Main.scala
* Last Modified: 5/11/2018
* Modified By: Andrew Roberts
*
* This file reads the preprocessed BBC term-document matrix from a CSV file and calls functions from NMF.scala
* and SVD.scala and KMEANS.scala to analyze the BBC corpus. Both algorithms calculate low-dimensional bases in the document space
* and save the resulting bases as CSV files. These files are used in subsequent analysis to determine the five 
* primary article topics in the BBC corpus. 
*
* IMPORTANT: This file should be run with a command such as "sbt -J-Xmx2g run" to expand the heap space; The default
*            memory limits are not sufficient to load the term-document matrix
*
*/

object Main extends App {

	
	// Read term-document matrix (tf-idf already applied)
	val X = csvread(new File("bbc_matrix_preprocessed.csv"))
	val rank = 5

	// Perform NMF on dataset
	val nmf_factors = NMF.nmf(X, rank, 100)
	println("NMF Approximation Error Rank 5:")
	println(Eval.rel_error(X, nmf_factors._1 * nmf_factors._2))
	csvwrite(new File("NMF Basis Rank 5.csv"), nmf_factors._1, separator=',')

	// Perform SVD on dataset 
	val svd_factors = SVD.get_svd_factors(X, false)
	println("SVD Approximation Error Rank 5:")
	println(Eval.rel_error(X, SVD.svd_compression(svd_factors._1, svd_factors._2, svd_factors._3, rank)))
	csvwrite(new File("SVD Basis Rank 5.csv"), svd_factors._1(::, 0 until rank), separator=',')

	// Perform KMeans on dataset
	labels = KMeans.cluster(X, 5)

	/*
		
	EXAMPLE CODE: HOW TO USE SVD, NMF, KMEANS OBJECTS

	// NMF Example
	println("NMF Example: \n")
	var A = DenseMatrix.rand(1000, 100)
	var B = DenseMatrix.rand(100, 600)
	var C = A * B
	
	val nmf_factors = NMF.nmf(C, 100, 1000)
	println("NMF Approximation Error Rank 100:")
	println(NMF.rel_error(C, nmf_factors._1, nmf_factors._2))

	// SVD Example
	println("SVD Example: \n")
	val C = DenseMatrix.rand(20, 10)
	val svd_factors = SVD.get_svd_factors(C, false) 

	val C_approx = SVD.svd_compression(svd_factors._1, svd_factors._2, svd_factors._3, 100)
	
	println("SVD Approximation Error Full Rank:")
	println(SVD.compute_relative_err(C, svd_factors._1*svd_factors._2*svd_factors._3))
	println("SVD Approximation Error Rank 100:")
	println(SVD.compute_relative_err(C, C_approx))

	// KMeans Example	
	val X = DenseMatrix((0.0, 10.0, 11.0, 2.0, .1, 13.0), (.05, 10.0, 12.0, 1.0, .7, 9.0))
	println(KMeans.cluster(X, 2))

	*/
}
